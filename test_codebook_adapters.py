"""Small correctness tests for the optional codebook adapters."""

from __future__ import annotations

import unittest

import torch

from quantizers.base_codebook import CodebookContext
from quantizers.codebook_factory import get_codebook
from quantizers.rbvt import apply_rbvt


class CodebookAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.weight = torch.randn(5, 19)
        self.mean = torch.randn(19) * 0.1
        self.variance = torch.rand(19) + 0.1

    def test_quant_result_is_rbvt_compatible(self):
        for name in ("leanquant", "squeezellm"):
            for bits in (3, 4):
                with self.subTest(name=name, bits=bits):
                    codebook = get_codebook(
                        name,
                        bits,
                        group_size=-1,
                    )
                    centers = torch.randn(5, 1, 2**bits).sort(dim=-1).values
                    codebook.set_context(
                        CodebookContext(
                            precomputed_centers=centers,
                        )
                    )
                    result = codebook.quantize(self.weight, row_chunk=3)

                    self.assertEqual(result.W_dequant.shape, self.weight.shape)
                    self.assertEqual(result.indices.shape, self.weight.shape)
                    self.assertEqual(result.block_codebooks.shape, (5, 1, 2**bits))
                    self.assertEqual(result.block_size, 19)
                    self.assertTrue(
                        torch.all(
                            result.block_codebooks[:, :, 1:]
                            >= result.block_codebooks[:, :, :-1]
                        )
                    )

                    rbvt_weight, stats = apply_rbvt(
                        W_fp=self.weight,
                        qres=result,
                        mu=self.mean,
                        sigma_ii=self.variance,
                        row_chunk=2,
                    )
                    self.assertEqual(rbvt_weight.shape, self.weight.shape)
                    self.assertGreaterEqual(stats.flips, 0)

    def test_adapter_requires_upstream_centers(self):
        codebook = get_codebook("squeezellm", 3)
        codebook.set_context(CodebookContext())
        with self.assertRaisesRegex(RuntimeError, "upstream repository"):
            codebook.quantize(self.weight)


if __name__ == "__main__":
    unittest.main()
