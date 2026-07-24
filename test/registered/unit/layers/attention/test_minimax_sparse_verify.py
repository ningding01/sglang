import unittest

import torch

from sglang.srt.layers.attention.minimax_sparse_backend import (
    build_minimax_linear_verify_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiniMaxSparseVerifyRows(CustomTestCase):
    def test_request_major_rows(self):
        row_req_pool_indices, row_seq_lens = build_minimax_linear_verify_rows(
            torch.tensor([7, 3], dtype=torch.int64),
            torch.tensor([17, 33], dtype=torch.int32),
            draft_token_num=4,
        )

        torch.testing.assert_close(
            row_req_pool_indices,
            torch.tensor([7, 7, 7, 7, 3, 3, 3, 3], dtype=torch.int64),
        )
        torch.testing.assert_close(
            row_seq_lens,
            torch.tensor([18, 19, 20, 21, 34, 35, 36, 37], dtype=torch.int32),
        )

    def test_single_verify_token(self):
        row_req_pool_indices, row_seq_lens = build_minimax_linear_verify_rows(
            torch.tensor([5, 9], dtype=torch.int32),
            torch.tensor([1, 128], dtype=torch.int64),
            draft_token_num=1,
        )

        torch.testing.assert_close(
            row_req_pool_indices, torch.tensor([5, 9], dtype=torch.int32)
        )
        torch.testing.assert_close(
            row_seq_lens, torch.tensor([2, 129], dtype=torch.int64)
        )

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_minimax_linear_verify_rows(
                torch.tensor([0]), torch.tensor([1]), draft_token_num=0
            )

        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            build_minimax_linear_verify_rows(
                torch.tensor([0, 1]), torch.tensor([1]), draft_token_num=2
            )


if __name__ == "__main__":
    unittest.main()
