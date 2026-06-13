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
        activations = torch.randn(64, 19)
        self.hessian = 2.0 / activations.shape[0] * activations.t().matmul(activations)
        self.sensitivity = torch.rand_like(self.weight)

    def test_quant_result_is_rbvt_compatible(self):
        for name in ("leanquant", "squeezellm"):
            for bits in (3, 4):
                with self.subTest(name=name, bits=bits):
                    codebook = get_codebook(
                        name,
                        bits,
                        group_size=-1,
                        fit_row_chunk=2,
                    )
                    codebook.set_context(
                        CodebookContext(
                            activation_mean=self.mean,
                            activation_variance=self.variance,
                            hessian=self.hessian if name == "leanquant" else None,
                            sensitivity=(
                                self.sensitivity if name == "squeezellm" else None
                            ),
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

    def test_squeezellm_accepts_explicit_sensitivity(self):
        codebook = get_codebook(
            "squeezellm",
            3,
            group_size=-1,
            fit_row_chunk=2,
        )
        codebook.set_context(
            CodebookContext(
                activation_mean=self.mean,
                activation_variance=self.variance,
                sensitivity=self.sensitivity,
            )
        )
        result = codebook.quantize(self.weight, row_chunk=4)
        self.assertEqual(result.block_codebooks.shape, (5, 1, 8))
        self.assertEqual(result.block_size, 19)

    def test_squeezellm_rejects_activation_proxy(self):
        codebook = get_codebook("squeezellm", 3)
        codebook.set_context(
            CodebookContext(
                activation_mean=self.mean,
                activation_variance=self.variance,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "Fisher"):
            codebook.quantize(self.weight)


if __name__ == "__main__":
    unittest.main()
