# Copyright 2025 XunhaoLai. All rights reserved.

import logging
from typing import Callable, List, Optional, Tuple

import torch

from .common.index import topk_index_reduce
from .common.utils import get_cu_seqblocks
from .decode.flash_with_topk_idx import flash_decode_with_topk_idx
from .decode.topk_sparse import flash_decode_with_gqa_share_sparse
from .prefill.flash_with_topk_idx import flash_prefill_with_topk_index
from .prefill.topk_sparse import flash_prefill_with_gqa_share_sparse

logger = logging.getLogger(__name__)


def minimax_sparse_prefill(
    q: torch.Tensor,  # [total_extend_tokens, num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    sink: Optional[torch.Tensor],  # [num_q_heads, qk_head_dim]
    idx_q: torch.Tensor,  # [total_extend_tokens, num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim] (paged index)
    idx_v_cache: Optional[
        torch.Tensor
    ],  # [max_slots, 1, idx_head_dim] (paged index); None when disable_index_value
    idx_sink: Optional[torch.Tensor],  # [num_idx_heads, idx_head_dim]
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch_size, ]
    cu_seqlens: torch.Tensor,  # [batch_size + 1, ] (Q-side cumulative)
    seq_lens: torch.Tensor,  # [batch_size, ] total K length (prefix + chunk)
    prefix_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    use_msa: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
    seqlens_cpu: Optional[List[int]] = None,
    cached_topk_idx: Optional[torch.Tensor] = None,
    return_topk_idx: bool = False,
):
    """Run MiniMax-M3 sparse prefill.

    Index cache (ATOM #1354): when ``cached_topk_idx`` is given, skip Step 1
    (the flash-index attention + top-k selection) and reuse the provided top-k
    indices for Step 3's sparse attention. When ``return_topk_idx`` is True, the
    reduced top-k tensor is returned as a third element so the caller can cache
    it for later skip layers. Only valid for ``disable_index_value`` layers
    (idx_o is None there, so skipping the indexer has no output side effect).

    ``cu_seqblocks_q``, ``max_seqblock_q``, and ``all_seqblock_q`` are optional
    precomputed query-block metadata shared by the index and value sparse
    kernels. Supplying them avoids recomputing the same block layout twice.
    ``seqlens_cpu`` (host copy of ``torch.diff(cu_seqlens)``) is forwarded to
    ``get_cu_seqblocks`` to avoid a per-layer device sync when it recomputes.
    """
    if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k, seqlens_cpu
        )

    use_atom_env = False
    try:
        from sglang.srt.environ import envs

        use_atom_env = envs.SGLANG_MINIMAX_M3_ATOM_PREFILL.get()
    except Exception:
        use_atom_env = False

    if use_atom_env and idx_k_cache.dim() == 5:
        from .atom_prefill import vectorized_5d_index_cache_to_nhd

        idx_k_cache = vectorized_5d_index_cache_to_nhd(idx_k_cache)
        if idx_v_cache is not None and idx_v_cache.dim() == 5:
            idx_v_cache = vectorized_5d_index_cache_to_nhd(idx_v_cache)

    # All seqlen is less than topk, use full attention
    if cached_topk_idx is not None:
        # Index cache hit: reuse a prior sparse layer's reduced top-k, skipping
        # Step 1 (flash-index attention + top-k) and Step 2 (reduce). idx_o is
        # unused downstream for disable_index_value layers.
        idx_o = None
        topk_idx = cached_topk_idx
    else:
        # Step 1: Flash attention with topk index (using index head)
        idx_o, topk_idx = flash_prefill_with_topk_index(
            q=idx_q,
            k_cache=idx_k_cache,
            v_cache=idx_v_cache,
            sink=idx_sink,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_size_q=block_size_q,
            block_size_k=block_size_k,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            sm_scale=idx_sm_scale,
            score_type=score_type,
            disable_index_value=disable_index_value,
            cu_seqblocks_q=cu_seqblocks_q,
            max_seqblock_q=max_seqblock_q,
            all_seqblock_q=all_seqblock_q,
        )
        # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads
        num_idx_heads = idx_q.shape[1]
        num_kv_heads = k_cache.shape[1]
        idx_group_size = num_idx_heads // num_kv_heads
        if idx_group_size > 1:
            topk_idx = topk_index_reduce(
                topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
            )
    # Reduced top-k to be cached by the caller for subsequent skip layers.
    reduced_topk_idx = topk_idx
    # Step 3: Sparse attention using topk index (main head). The MSA path only
    # replaces this step; the indexer above is unchanged. MSA has no attn-sink
    # input, so keep the Triton path when sink is present.
    use_atom_prefill = False
    try:
        if use_atom_env:
            from .atom_prefill import can_use_atom_prefill

            use_atom_prefill = can_use_atom_prefill(
                q, k_cache, v_cache, sink, block_size_k
            )
            if not use_atom_prefill:
                logger.warning(
                    "SGLANG_MINIMAX_M3_ATOM_PREFILL is set, but the current "
                    "MiniMax-M3 prefill batch/cache layout is unsupported; "
                    "falling back to the default sparse prefill path."
                )
    except Exception as exc:
        logger.warning(
            "Failed to initialize MiniMax-M3 ATOM prefill path; falling back: %s",
            exc,
        )
        use_atom_prefill = False

    if use_atom_prefill:
        from .atom_prefill import atom_gluon_sparse_prefill

        o = atom_gluon_sparse_prefill(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            topk_idx=topk_idx,
            req_to_token=req_to_token,
            req_pool_indices=slot_ids,
            cu_seqlens=cu_seqlens,
            prefix_lens=prefix_lens,
            block_size_k=block_size_k,
            sm_scale=sm_scale,
        )
    elif use_msa and sink is None:
        from .msa import msa_sparse_prefill_main

        o = msa_sparse_prefill_main(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            topk_idx=topk_idx,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            block_size_k=block_size_k,
            sm_scale=sm_scale,
        )
    else:
        o = flash_prefill_with_gqa_share_sparse(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            sink=sink,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            topk_idx=topk_idx,
            block_size_q=block_size_q,
            block_size_k=block_size_k,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=max_seqlen_q,
            sm_scale=sm_scale,
            cu_seqblocks_q=cu_seqblocks_q,
            max_seqblock_q=max_seqblock_q,
        )
    if return_topk_idx:
        return idx_o, o, reduced_topk_idx
    return idx_o, o


def minimax_sparse_decode(
    q: torch.Tensor,  # [batch_size, num_q_heads, qk_head_dim]
    sink: Optional[torch.Tensor],  # [num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    idx_q: torch.Tensor,  # [batch_size, num_idx_heads, idx_head_dim], num_idx_heads >= num_kv_heads
    idx_sink: Optional[torch.Tensor],  # [num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim] (paged)
    idx_v_cache: Optional[
        torch.Tensor
    ],  # [max_slots, 1, idx_head_dim] (paged); None when disable_index_value
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch_size, ]
    seq_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen: int,  # max of seq_lens, passed from caller to avoid sync during CUDA graph capture
    block_size_q: int,  # useless for now, will always be 1
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    dense_main_attn_fn: Optional[Callable] = None,
    page_size: int = 1,
    use_msa: bool = False,
    msa_kv_indices: Optional[
        torch.Tensor
    ] = None,  # per-forward MSA page table (cached)
    msa_plan=None,  # per-forward MSA fmha_sm100 plan (cached)
) -> Tuple[torch.Tensor, torch.Tensor]:
    use_atom_env = False
    try:
        from sglang.srt.environ import envs

        use_atom_env = envs.SGLANG_MINIMAX_M3_ATOM_PREFILL.get()
    except Exception:
        use_atom_env = False

    if use_atom_env and idx_k_cache.dim() == 5:
        from .atom_prefill import vectorized_5d_index_cache_to_nhd

        idx_k_cache = vectorized_5d_index_cache_to_nhd(idx_k_cache)
        if idx_v_cache is not None and idx_v_cache.dim() == 5:
            idx_v_cache = vectorized_5d_index_cache_to_nhd(idx_v_cache)

    # Step 1: Flash decode with topk index (using index head). When the dense main
    # attention is used, the indexer emits the page table directly (fused
    # transform) instead of block ids, plus the per-query effective KV length.
    idx_o, topk_idx, real_seq_lens = flash_decode_with_topk_idx(
        q=idx_q,
        sink=idx_sink,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
        slot_ids=slot_ids,
        block_size=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        use_dense_main_attn=dense_main_attn_fn is not None,
        page_size=page_size,
    )
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if dense_main_attn_fn is not None:
        # topk_idx is the page table; real_seq_lens is the per-query cache_seqlens
        assert idx_group_size == 1
        o = dense_main_attn_fn(q, topk_idx, real_seq_lens)
    else:
        # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads
        if idx_group_size > 1:
            topk_idx = topk_index_reduce(
                topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
            )
        # Step 3: Sparse attention using topk index (main head). Decode stays on
        # SGLang's Triton sparse path, which reads a vectorized_5d main cache in
        # place. It used to gather the *entire* pool back to NHD here on every
        # sparse layer of every decode step — a batch-independent O(pool) cost
        # (~770 MB x 57 layers per step at a 1.5M-token pool) that dominated
        # decode and made 5D ~8x slower than nhd.
        if use_msa and sink is None:
            if k_cache.dim() == 5:
                # msa_sparse_decode_main has no 5D reader yet; keep the gather
                # confined to that path rather than silently mis-reading.
                from sglang.srt.layers.attention.utils import (
                    launch_gather_shuffle_5d_to_linear,
                )

                total_slots = k_cache.shape[0] * k_cache.shape[3]
                all_slots = torch.arange(
                    total_slots, dtype=torch.int64, device=q.device
                )
                k_cache, v_cache = launch_gather_shuffle_5d_to_linear(
                    k_cache, v_cache, all_slots
                )
            from .msa import msa_sparse_decode_main

            o = msa_sparse_decode_main(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                topk_idx=topk_idx,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                seq_lens=seq_lens,
                block_size_k=block_size_k,
                sm_scale=sm_scale,
                kv_indices=msa_kv_indices,
                plan=msa_plan,
            )
        else:
            o = flash_decode_with_gqa_share_sparse(
                q=q,
                sink=sink,
                k_cache=k_cache,
                v_cache=v_cache,
                req_to_token=req_to_token,
                seq_lens=seq_lens,
                slot_ids=slot_ids,
                block_size=block_size_k,
                topk_idx=topk_idx,
                sm_scale=sm_scale,
            )
    return idx_o, o
