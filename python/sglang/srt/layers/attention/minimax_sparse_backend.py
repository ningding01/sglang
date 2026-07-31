from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import (
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    get_minimax_sparse_score_type,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_decode,
    minimax_sparse_prefill,
)
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def build_minimax_linear_verify_rows(
    req_pool_indices: torch.Tensor,
    prefix_seq_lens: torch.Tensor,
    draft_token_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand a topk=1 verify chain into request-major decode rows."""
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}")
    if req_pool_indices.shape != prefix_seq_lens.shape:
        raise ValueError(
            "req_pool_indices and prefix_seq_lens must have the same shape, got "
            f"{req_pool_indices.shape} and {prefix_seq_lens.shape}"
        )

    batch_size = prefix_seq_lens.shape[0]
    row_req_pool_indices = req_pool_indices.repeat_interleave(draft_token_num)
    local_query_offsets = torch.arange(
        1,
        draft_token_num + 1,
        dtype=prefix_seq_lens.dtype,
        device=prefix_seq_lens.device,
    ).repeat(batch_size)
    row_seq_lens = prefix_seq_lens.repeat_interleave(
        draft_token_num
    ) + local_query_offsets
    return row_req_pool_indices, row_seq_lens


class MiniMaxSparseAttnBackend(AttentionBackend):
    def __init__(self, runner: ModelRunner):
        assert isinstance(runner.token_to_kv_pool, MiniMaxSparseKVPool)
        self.kv_pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.max_context_len = int(runner.model_config.context_len)

        hf_config = runner.model_config.hf_config
        sparse_cfg = get_minimax_sparse_attention_config(hf_config)
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.dense_layer_ids, self.sparse_layer_ids = get_minimax_sparse_layer_ids(
            sparse_cfg
        )
        self.disable_value_layer_ids: set[int] = set(
            get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
        )
        self.score_type: str = get_minimax_sparse_score_type(sparse_cfg)
        # assert self.idx_head_dim == head_dim

        # max_seqlen for the current forward pass, stored as a plain Python int
        # so that it is safe to use inside CUDA graphs (no .item() at graph time).
        # Populated by init_forward_metadata* before each forward.
        self._max_seqlen_q: int = 1
        self._max_seqlen_k: int = 1

        self.block_size_q = 1
        self.block_size_k = sparse_cfg["sparse_block_size"]
        if "sparse_init_block" in sparse_cfg:
            self.init_blocks = sparse_cfg["sparse_init_block"]
        else:
            init_tokens = sparse_cfg["sparse_init_tokens"]
            self.init_blocks = (
                init_tokens + self.block_size_k - 1
            ) // self.block_size_k
        if "sparse_local_block" in sparse_cfg:
            self.local_blocks = sparse_cfg["sparse_local_block"]
        else:
            local_tokens = sparse_cfg["sparse_local_tokens"]
            self.local_blocks = (
                local_tokens + self.block_size_k - 1
            ) // self.block_size_k + 1
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]

        # NVIDIA Blackwell (SM100): use MiniMax's MSA kernel (fmha_sm100) only
        # for the main sparse-attention step when the kernel constraints hold.
        # The lightning indexer remains unchanged; missing fmha_sm100 keeps the
        # existing Triton path.
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
            msa_available,
        )

        # MSA (fmha_sm100) is bf16/fp16-only. With an fp8 main KV cache
        # (--kv-cache-dtype fp8_*) keep the sparse path on Triton (it dequants fp8 on
        # load) rather than feeding fp8 bytes to the bf16 kernel; mirrors vLLM's
        # select_main_impl_cls (fp8 KV -> Triton, never MSA).
        _main_kv_is_fp8 = self.kv_pool.main_pool.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        self.use_msa = (
            not envs.SGLANG_DISABLE_MSA.get()
            and msa_available()
            and self.block_size_k == 128
            and self.kv_pool.page_size == self.block_size_k
            and self.topk_blocks in (4, 8, 16, 32)
            and not _main_kv_is_fp8
        )
        # Per-forward MSA decode metadata (page table + fmha plan), shared by every
        # sparse layer of a forward; (re)built in init_forward_metadata_out_graph.
        self._msa_dec_meta = None
        self._verify_row_req_pool_indices = None
        self._verify_row_seq_lens = None
        self._cuda_graph_verify_row_req_pool_indices = None
        self._cuda_graph_verify_row_seq_lens = None
        self._cuda_graph_verify_local_offsets = None
        if self.use_msa:
            from sglang.srt.layers.dp_attention import get_attention_tp_size

            # Per-rank head counts for the decode plan (== runtime q.shape[1] /
            # k_cache.shape[1]); needed in out_graph where q/k_cache aren't available.
            self.num_q_heads = (
                runner.model_config.num_attention_heads // get_attention_tp_size()
            )
            # KV head count lives on the main sub-pool (== runtime k_cache.shape[1]).
            self.num_kv_heads = self.kv_pool.main_pool.head_num
            # CUDA-graph decode: one persistent plan + page-table buffer per batch
            # size, refreshed in place each step (worklist is length-independent).
            self._msa_nb_max = (
                self.max_context_len + self.block_size_k - 1
            ) // self.block_size_k
            self._msa_cg: dict[int, tuple] = {}

        self.page_size = self.kv_pool.page_size
        self.use_dense_sparse_decode = (
            envs.SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE.get()
            and self.block_size_k % self.page_size == 0
        )
        # MSA fmha_sm100 decode is NOT cuda-graph-safe: captured & replayed it returns
        # wrong results that compound across replays (silent ~14% GSM8K loss on B200;
        # masked early by radix-cache prefix reuse, then cliffs under sustained load).
        # Use the MSA decode kernel only when decode does NOT run under cuda graph;
        # otherwise route the decode step through the cuda-graph-safe Triton sparse path.
        # MSA still serves prefill (run eager — prefill cuda graph is disabled), where
        # its long-context speedup matters.
        #
        # Decide from the resolved cuda_graph_config — the same source
        # init_decode_cuda_graph uses to decide capture — not the legacy disable_*
        # server_args flags: the two can disagree under config-native flags, and a
        # mismatch could capture the unsafe MSA decode kernel into a graph.
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            check_cuda_graph_backend,
        )

        _sa = getattr(runner, "server_args", None)
        _decode_cuda_graph = not check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        )
        self._use_msa_decode = self.use_msa and not _decode_cuda_graph

        # MSA + speculative decode + cuda graph is unsupported: spec verify
        # (TARGET_VERIFY) batches route to forward_extend and are captured into the
        # decode graph, which both dereferences extend metadata absent in the capture
        # batch and would record the MSA prefill kernel into a graph. Fail loudly at
        # startup instead of crashing mid-capture.
        if (
            self.use_msa
            and _decode_cuda_graph
            and getattr(_sa, "speculative_algorithm", None) is not None
        ):
            raise NotImplementedError(
                "MiniMax-M3 MSA attention does not support speculative decoding under "
                "CUDA graph. Use --disable-cuda-graph, set SGLANG_DISABLE_MSA=1, or "
                "disable speculative decoding."
            )
        # MSA owns the main decode step unless dense-sparse-decode does; the dense
        # path only engages when k_cache.shape[1] == 1 (see forward_decode).
        self._msa_owns_decode = self._use_msa_decode and not (
            self.use_dense_sparse_decode and self.kv_pool.main_pool.head_num == 1
        )
        # The page table + effective KV length are allocated and returned by the
        # fused decode top-k kernel each layer, so the backend keeps no metadata.
        self.dense_backend: Optional[AttentionBackend] = None

        # Index cache (ATOM #1354): share indexer top-k across groups of
        # consecutive sparse layers. freq=N -> each group of N sparse layers
        # computes top-k once (first layer) and the other N-1 reuse it. Prefill
        # only for now (decode runs under cuda graph; the per-forward host dict
        # would not be graph-safe). Only the group's source layer computes.
        self.index_topk_freq = max(int(envs.SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ.get()), 1)
        self.index_cache_enabled = self.index_topk_freq > 1
        # Map each sparse layer_id -> (group_key, is_source). Source layers compute
        # and store; non-source layers in a group reuse the stored top-k. Groups run
        # over the sparse-layer ordinal (position among sparse layers only).
        self._topk_group_of_layer: dict[int, int] = {}
        self._topk_is_source: dict[int, bool] = {}
        for ordinal, lid in enumerate(self.sparse_layer_ids):
            group = ordinal // self.index_topk_freq
            self._topk_group_of_layer[lid] = group
            self._topk_is_source[lid] = (ordinal % self.index_topk_freq) == 0
        # Per-forward cache {group_key: reduced_topk_idx}; cleared each forward.
        self._topk_cache: dict = {}
        # Separate from _topk_cache: verify's top-k is shaped over
        # batch * draft_token_num rows, so it must never be mixed with a
        # prefill-shaped entry for the same group.
        self._verify_topk_cache: dict = {}

        logger.info(
            f"[MiniMaxSparse] Backend initialized "
            f"(score_type={self.score_type!r}, "
            f"main_attn={'MSA' if self.use_msa else 'triton'}, "
            f"index_topk_freq={self.index_topk_freq}, "
            f"disable_value_layers={sorted(self.disable_value_layer_ids)})"
        )

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # cuda-graph replay views are a SimpleNamespace without extend_seq_lens_cpu,
        # and TARGET_VERIFY sets it to None despite is_extend() — getattr covers both.
        # New forward -> invalidate the cached per-forward MSA decode metadata.
        self._msa_dec_meta = None
        # New forward -> drop the per-forward index-cache top-k (prefill + verify).
        if self.index_cache_enabled:
            self._topk_cache = {}
            self._verify_topk_cache = {}
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is not None:
            self._max_seqlen_q = int(max(extend_lens))
        else:
            self._max_seqlen_q = 1
        if in_capture and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
        ):
            self._max_seqlen_k = self.max_context_len
        elif forward_batch.forward_mode.is_target_verify():
            verify_input = forward_batch.spec_info
            if verify_input.topk != 1:
                raise NotImplementedError(
                    "MiniMax-M3 sparse attention currently supports EAGLE target "
                    "verification only with speculative_eagle_topk=1."
                )
            self._max_seqlen_k = int(
                forward_batch.seq_lens_cpu.max().item()
                + verify_input.draft_token_num
            )
        else:
            self._max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())

        if forward_batch.forward_mode.is_target_verify():
            verify_input = forward_batch.spec_info
            if verify_input.topk != 1:
                raise NotImplementedError(
                    "MiniMax-M3 sparse attention currently supports EAGLE target "
                    "verification only with speculative_eagle_topk=1."
                )
            num_rows = forward_batch.seq_lens.shape[0] * verify_input.draft_token_num
            if self._cuda_graph_verify_row_req_pool_indices is None:
                (
                    self._verify_row_req_pool_indices,
                    self._verify_row_seq_lens,
                ) = build_minimax_linear_verify_rows(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    verify_input.draft_token_num,
                )
            else:
                if num_rows > self._cuda_graph_verify_row_req_pool_indices.shape[0]:
                    raise RuntimeError(
                        "MiniMax-M3 target verify rows exceed CUDA graph capacity: "
                        f"{num_rows} > "
                        f"{self._cuda_graph_verify_row_req_pool_indices.shape[0]}"
                    )
                self._verify_row_req_pool_indices = (
                    self._cuda_graph_verify_row_req_pool_indices[:num_rows]
                )
                self._verify_row_seq_lens = self._cuda_graph_verify_row_seq_lens[
                    :num_rows
                ]

        # Build the MSA decode plan + page table here (eager, outside graph capture)
        # so forward_decode — captured into the graph — only runs device-side ops.
        # Runs at capture, replay, and eager, refreshing the persistent buffers the
        # captured graph reads. Skipped when the dense-sparse-decode path owns decode.
        if self._msa_owns_decode and forward_batch.forward_mode.is_decode_or_idle():
            self._prepare_msa_decode_meta(forward_batch)

    def _prepare_msa_decode_meta(self, forward_batch: ForwardBatch):
        """Refresh the persistent per-batch-size MSA decode plan + page table in place."""
        from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
            build_msa_decode_cg_plan,
            update_msa_decode_cg_meta,
        )

        bs = forward_batch.seq_lens.shape[0]
        if bs == 0:
            return
        entry = self._msa_cg.get(bs)
        if entry is None:
            device = forward_batch.seq_lens.device
            plan = build_msa_decode_cg_plan(
                self.num_q_heads,
                self.num_kv_heads,
                self.block_size_k,
                self.topk_blocks,
                bs,
                device=device,
            )
            kv_indices_buf = torch.zeros(
                bs * self._msa_nb_max, dtype=torch.int32, device=device
            )
            entry = (plan, kv_indices_buf)
            self._msa_cg[bs] = entry
        plan, kv_indices_buf = entry
        update_msa_decode_cg_meta(
            plan,
            kv_indices_buf,
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self.block_size_k,
            self.topk_blocks,
            self.num_q_heads,
            self.num_kv_heads,
        )
        self._msa_dec_meta = (kv_indices_buf, plan)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        if not forward_batch.forward_mode.is_target_verify():
            return
        if self._cuda_graph_verify_row_req_pool_indices is None:
            return

        draft_token_num = forward_batch.spec_info.draft_token_num
        batch_size = forward_batch.seq_lens.shape[0]
        num_rows = batch_size * draft_token_num
        req_rows = self._cuda_graph_verify_row_req_pool_indices[:num_rows].view(
            batch_size, draft_token_num
        )
        seq_rows = self._cuda_graph_verify_row_seq_lens[:num_rows].view(
            batch_size, draft_token_num
        )
        req_rows.copy_(forward_batch.req_pool_indices[:, None])
        seq_rows.copy_(
            forward_batch.seq_lens[:, None]
            + self._cuda_graph_verify_local_offsets[:draft_token_num][None, :]
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        device = self.req_to_token.device
        self._cuda_graph_verify_row_req_pool_indices = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=device
        )
        self._cuda_graph_verify_row_seq_lens = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=device
        )
        num_tokens_per_bs = max_num_tokens // max_bs
        self._cuda_graph_verify_local_offsets = torch.arange(
            1,
            num_tokens_per_bs + 1,
            dtype=torch.int64,
            device=device,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @staticmethod
    def _is_sparse_kv_cached_by_fusion(
        forward_batch: ForwardBatch, layer_id: int
    ) -> bool:
        layer_ids = forward_batch.minimax_m3_precached_sparse_layers
        return layer_ids is not None and layer_id in layer_ids

    def index_topk_skipped(self, layer_id: int, disable_value: bool) -> bool:
        """Whether this sparse layer reuses another layer's top-k (index cache).

        When True, the layer never runs the indexer (no flash-index attention,
        no top-k), so its index Q/K norm+rope is dead work the model can skip.
        Only valid for disable_value layers (idx_o is None there). Applies to
        prefill and to EAGLE target verify; plain decode still computes its own
        top-k because its fused dense path also needs real_seq_lens, which the
        cache does not carry.
        """
        return (
            self.index_cache_enabled
            and disable_value
            and not self._topk_is_source.get(layer_id, True)
        )

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if forward_batch.forward_mode.is_idle():
            idx_q = kwargs.get("idx_q")
            num_idx_heads = idx_q.shape[1]
            disable_value = layer.layer_id in self.disable_value_layer_ids
            idx_out: Optional[torch.Tensor] = (
                None
                if disable_value
                else q.new_zeros(q.shape[0], num_idx_heads * self.idx_head_dim)
            )
            out = q.new_zeros(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
            return idx_out, out
        else:
            return super().forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ):
        disable_value = layer.layer_id in self.disable_value_layer_ids
        kv_cached_by_fusion = self._is_sparse_kv_cached_by_fusion(
            forward_batch, layer.layer_id
        )
        if not kv_cached_by_fusion:
            self.kv_pool.set_fused_kv_index_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                idx_k,
                None if disable_value else idx_v,
            )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        if forward_batch.forward_mode.is_target_verify():
            return self._forward_target_verify(
                q,
                k_cache,
                v_cache,
                idx_q,
                idx_k_cache,
                idx_v_cache,
                forward_batch,
                disable_value,
            )

        cu_seqlens = torch.cat(
            [
                torch.zeros(
                    1, dtype=torch.int32, device=forward_batch.extend_seq_lens.device
                ),
                forward_batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
            ]
        )
        seq_lens = forward_batch.seq_lens.to(torch.int32)  # prefix + extend
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
        else:
            prefix_lens = torch.zeros_like(seq_lens)

        # In DP attention mode, q may be padded beyond the actual token count
        # for collective communication alignment. Trim to actual tokens so
        # the sparse attention kernel sees consistent shapes.
        #
        # Source the token count from CPU-side metadata when available so we do
        # not force a GPU->CPU sync (cu_seqlens[-1].item()) on every sparse
        # layer of every prefill. extend_seq_lens_cpu is a plain list of ints
        # (ForwardBatch sets it from extend_seq_lens.cpu()), so sum() is a host
        # op and the result is identical to cu_seqlens[-1]. Fall back to the
        # device tensor only when CPU metadata is absent.
        if forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
        else:
            actual_num_tokens = int(cu_seqlens[-1].item())
        original_num_tokens = q.shape[0]
        if actual_num_tokens < original_num_tokens:
            q = q[:actual_num_tokens]
            idx_q = idx_q[:actual_num_tokens]

        # Index cache (ATOM #1354): only for disable_value layers (idx_o is None,
        # so skipping the indexer has no output side effect). A group's source
        # layer computes + stores the reduced top-k; the other layers reuse it.
        use_index_cache = self.index_cache_enabled and disable_value
        cached_topk_idx = None
        want_topk = False
        if use_index_cache:
            group = self._topk_group_of_layer[layer.layer_id]
            if self._topk_is_source[layer.layer_id]:
                want_topk = True  # compute and store for this group
            else:
                cached_topk_idx = self._topk_cache.get(group)
                # Miss (e.g. source layer chunked differently) -> recompute safely.

        result = minimax_sparse_prefill(
            q,
            k_cache,
            v_cache,
            None,
            idx_q,
            idx_k_cache,
            idx_v_cache,
            None,
            self.req_to_token,
            forward_batch.req_pool_indices,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            self._max_seqlen_q,
            self._max_seqlen_k,
            self.block_size_q,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            use_msa=self.use_msa,
            # Host seq-lens let get_cu_seqblocks avoid a per-layer .item() sync.
            seqlens_cpu=forward_batch.extend_seq_lens_cpu,
            cached_topk_idx=cached_topk_idx,
            return_topk_idx=want_topk,
        )
        if want_topk:
            idx_o, o, reduced_topk_idx = result
            self._topk_cache[group] = reduced_topk_idx
        else:
            idx_o, o = result

        # Pad output back to original size for DP communication
        if actual_num_tokens < original_num_tokens:
            pad_len = original_num_tokens - actual_num_tokens
            o = torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)
            if idx_o is not None:
                idx_o = torch.cat(
                    [idx_o, idx_o.new_zeros(pad_len, *idx_o.shape[1:])], dim=0
                )

        return (
            (
                None
                if idx_o is None
                else idx_o.reshape(original_num_tokens, -1).contiguous()
            ),
            o.reshape(original_num_tokens, -1).contiguous(),
        )

    def _forward_target_verify(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        disable_value: bool,
    ):
        draft_token_num = forward_batch.spec_info.draft_token_num
        batch_size = forward_batch.seq_lens.shape[0]
        expected_num_tokens = batch_size * draft_token_num
        if q.shape[0] != expected_num_tokens:
            raise RuntimeError(
                "MiniMax-M3 target verify expects a uniform request-major token "
                f"layout: {q.shape[0]} != {batch_size} * {draft_token_num}"
            )

        row_req_pool_indices = self._verify_row_req_pool_indices
        row_seq_lens = self._verify_row_seq_lens
        if row_req_pool_indices is None or row_seq_lens is None:
            raise RuntimeError("MiniMax-M3 target verify metadata was not initialized")

        # Index cache on the verify path. Verify runs the indexer over
        # batch * draft_token_num rows, so it is where the per-layer indexer cost
        # hurts most: profiling EAGLE3 steps=3 at 75K showed it as the largest
        # decode-side amplifier (2.39x vs plain decode, the worst of any
        # decode-only kernel). Sharing one group's reduced top-k across its
        # sparse layers cuts that by index_topk_freq. Same approximation the
        # prefill path already ships, and the same one ATOM applies to decode.
        use_index_cache = self.index_cache_enabled and disable_value
        cached_topk_idx = None
        want_topk = False
        if use_index_cache:
            group = self._topk_group_of_layer[layer.layer_id]
            if self._topk_is_source[layer.layer_id]:
                want_topk = True
            else:
                cached_topk_idx = self._verify_topk_cache.get(group)

        result = minimax_sparse_decode(
            q,
            None,
            k_cache,
            v_cache,
            idx_q,
            None,
            idx_k_cache,
            idx_v_cache,
            self.req_to_token,
            row_req_pool_indices,
            row_seq_lens,
            self._max_seqlen_k,
            1,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            page_size=self.page_size,
            use_msa=False,
            cached_topk_idx=cached_topk_idx,
            return_topk_idx=want_topk,
        )
        if want_topk:
            idx_o, o, reduced_topk_idx = result
            self._verify_topk_cache[group] = reduced_topk_idx
        else:
            idx_o, o = result
        return (
            None if idx_o is None else idx_o.reshape(q.shape[0], -1).contiguous(),
            o.reshape(q.shape[0], -1).contiguous(),
        )

    def _dense_sparse_main_decode(
        self,
        q: torch.Tensor,  # [bs, num_q_heads, head_dim]
        page_table: torch.Tensor,  # [bs, max_sparse_pages] int32 (from the indexer)
        real_seq_lens: torch.Tensor,  # [bs] int32, effective KV length per query
        k_cache: torch.Tensor,  # [max_slots, 1, head_dim]
        v_cache: torch.Tensor,  # [max_slots, 1, head_dim]
        layer,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        if isinstance(self.dense_backend, TRTLLMHAAttnBackend):
            import flashinfer

            ps = self.page_size
            nkv = 1
            head_dim = q.size(-1)
            # [max_slots, nkv, D] -> [num_pages, page_size, nkv, D]
            #                     -> [num_pages, nkv, page_size, D] (HND, trtllm default)
            kc = k_cache.view(-1, ps, nkv, head_dim).permute(0, 2, 1, 3)
            vc = v_cache.view(-1, ps, nkv, head_dim).permute(0, 2, 1, 3)
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache(  # type: ignore
                query=q.contiguous(),
                kv_cache=(kc, vc),
                workspace_buffer=self.dense_backend.workspace_buffer,
                block_tables=page_table,
                seq_lens=real_seq_lens,
                max_seq_len=self.topk_blocks * self.block_size_k,
                bmm1_scale=layer.scaling,
                bmm2_scale=1.0,
            )
        raise NotImplementedError(
            "dense sparse decode currently supports trtllm_mha only (fa3 is TODO)"
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        **kwargs,
    ):
        assert len(kwargs) == 0
        disable_value = layer.layer_id in self.disable_value_layer_ids
        self.kv_pool.set_fused_kv_index_buffer(
            layer,
            forward_batch.out_cache_loc,
            k,
            v,
            idx_k,
            None if disable_value else idx_v,
        )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        attn_fn = None
        if self.use_dense_sparse_decode and k_cache.shape[1] == 1:

            def attn_fn(main_q, page_table, real_seq_lens):
                return self._dense_sparse_main_decode(
                    main_q,
                    page_table,
                    real_seq_lens,
                    k_cache,
                    v_cache,
                    layer,
                    forward_batch,
                )

        # The MSA decode page table + plan are built once per forward in
        # init_forward_metadata_out_graph (eager, outside graph capture) and shared
        # across all sparse layers; here we just consume the cached metadata.
        msa_kv_indices = msa_plan = None
        if self._use_msa_decode and attn_fn is None:
            if self._msa_dec_meta is not None:
                msa_kv_indices, msa_plan = self._msa_dec_meta
            elif q.shape[0] > 0:
                # Rebuilding the plan inline would run host-side code inside
                # CUDA-graph capture; fail loudly instead.
                raise RuntimeError(
                    "MSA decode metadata missing: init_forward_metadata_out_graph "
                    "did not prepare the plan for this forward (gate mismatch)."
                )

        idx_o, o = minimax_sparse_decode(
            q,
            None,
            k_cache,
            v_cache,
            idx_q,
            None,
            idx_k_cache,
            idx_v_cache,
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self._max_seqlen_k,
            1,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            dense_main_attn_fn=attn_fn,
            page_size=self.page_size,
            use_msa=self._use_msa_decode,
            msa_kv_indices=msa_kv_indices,
            msa_plan=msa_plan,
        )
        return (
            None if idx_o is None else idx_o.reshape(q.shape[0], -1).contiguous(),
            o.reshape(q.shape[0], -1).contiguous(),
        )


class MiniMaxHybridAttnBackend(AttentionBackend):
    """Combines a dense backend and a sparse backend, routing by call site."""

    # aiter dense gluon decode ps-reduce covers at most 64 partitions * 256 tokens.
    _GLUON_DECODE_MAX_CTX = 64 * 256  # 16384

    def __init__(
        self,
        dense_backend: AttentionBackend,
        sparse_backend: MiniMaxSparseAttnBackend,
        sparse_layer_ids: list[int],
        dense_decode_backend: Optional[AttentionBackend] = None,
    ):
        self.dense = dense_backend
        self.sparse = sparse_backend
        self.sparse_layer_ids = sparse_layer_ids
        # Fallback backend for LONG-CONTEXT dense DECODE only (triton), used when
        # context exceeds what aiter's gluon decode can cover (see forward()).
        # None -> all dense decode uses self.dense.
        self.dense_decode = dense_decode_backend
        # Let the sparse decode reuse the dense paged backend (page table + workspace).
        self.sparse.dense_backend = dense_backend

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # delegate so the dense (FlashInfer) backend keeps its own eager init.
        self.sparse.init_forward_metadata(forward_batch)
        self.dense.init_forward_metadata(forward_batch)
        # Only init the triton decode fallback when it will actually be used
        # (long-context dense decode). Initing it unconditionally perturbs shared
        # runner scratch and regresses the normal aiter path. See _needs_dense_decode.
        if self._needs_dense_decode(forward_batch):
            self.dense_decode.init_forward_metadata(forward_batch)

    def _needs_dense_decode(self, forward_batch: ForwardBatch) -> bool:
        return (
            self.dense_decode is not None
            and forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.seq_lens_cpu is not None
            and int(forward_batch.seq_lens_cpu.max()) > self._GLUON_DECODE_MAX_CTX
        )

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self.sparse.init_forward_metadata_out_graph(forward_batch, in_capture)
        self.dense.init_forward_metadata_out_graph(forward_batch, in_capture)
        if self._needs_dense_decode(forward_batch):
            self.dense_decode.init_forward_metadata_out_graph(forward_batch, in_capture)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self.sparse.init_forward_metadata_in_graph(forward_batch)
        self.dense.init_forward_metadata_in_graph(forward_batch)
        if self._needs_dense_decode(forward_batch):
            self.dense_decode.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.dense.init_cuda_graph_state(max_bs, max_num_tokens)
        self.sparse.init_cuda_graph_state(max_bs, max_num_tokens)
        if self.dense_decode is not None:
            # The triton long-context decode fallback also needs its cuda-graph
            # buffers when decode runs under cuda graph.
            self.dense_decode.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.sparse.get_cuda_graph_seq_len_fill_value()

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

        # Dense layers delegate to the stock backend (e.g. flashinfer). Under DP
        # attention the per-rank token block is padded to an even length
        # (prepare_mlp_sync_batch -> ceil_align(num_tokens, attn_cp_size * 2)), but
        # flashinfer builds qo_indptr from extend_seq_lens, so q.shape[0] (padded)
        # != qo_indptr[-1] (real) and the paged-prefill kernel raises. Trim q to
        # the real token count and re-pad the output; k/v stay untrimmed so the
        # KV-cache write stays aligned with out_cache_loc. Prefill-only.
        mode = forward_batch.forward_mode
        # Long-context dense DECODE fallback: aiter's gluon decode ps-reduce only
        # covers <= 16384 tokens (64 partitions); beyond that it crashes. Route
        # just those steps to the triton decode backend. Short decode stays on
        # aiter (self.dense) — fast and KV-consistent with aiter prefill.
        if (
            mode.is_decode_or_idle()
            and self.dense_decode is not None
            and forward_batch.seq_lens_cpu is not None
            and int(forward_batch.seq_lens_cpu.max()) > self._GLUON_DECODE_MAX_CTX
        ):
            return self.dense_decode.forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        if mode.is_extend() and forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
            original_num_tokens = q.shape[0]
            if actual_num_tokens < original_num_tokens:
                o = self.dense.forward(
                    q[:actual_num_tokens],
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache,
                    **kwargs,
                )
                pad_len = original_num_tokens - actual_num_tokens
                return torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)

        return self.dense.forward(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        if (
            self.dense_decode is not None
            and forward_batch.seq_lens_cpu is not None
            and int(forward_batch.seq_lens_cpu.max()) > self._GLUON_DECODE_MAX_CTX
        ):
            return self.dense_decode.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.dense.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )
