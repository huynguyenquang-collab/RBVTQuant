"""LeanQuant loss-error-aware codebook adapter for RBVTQuant."""

from __future__ import annotations

import torch

from .base_codebook import BaseCodebook, CodebookContext


class LeanQuantCodebook(BaseCodebook):
    """Approximate LeanQuant grids from diagonal activation Hessian statistics.

    LeanQuant weights k-means samples by ``diag(Hinv) ** -exponent``. With the
    diagonal Hessian available in RBVTQuant, this is equivalent to weighting by
    ``diag(H) ** exponent`` up to a row-wise constant.
    """

    def __init__(
        self,
        bits: int,
        group_size: int = -1,
        n_iters: int = 20,
        fit_row_chunk: int = 32,
        exponent: float = 4.0,
    ):
        if exponent <= 0.0:
            raise ValueError(f"LeanQuant exponent must be positive, got {exponent}")
        super().__init__(
            bits=bits,
            group_size=group_size,
            n_iters=n_iters,
            fit_row_chunk=fit_row_chunk,
        )
        self.name = f"leanquant{bits}"
        self.exponent = exponent

    def initial_centers(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_levels <= 8:
            alpha = torch.linspace(
                0.0,
                1.0,
                self.num_levels,
                device=values.device,
                dtype=torch.float32,
            )
            lower = values.amin(dim=1, keepdim=True)
            upper = values.amax(dim=1, keepdim=True)
            return lower + (upper - lower) * alpha.unsqueeze(0)
        return super().initial_centers(values, weights)

    def sample_weights(
        self,
        values: torch.Tensor,
        context: CodebookContext,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
    ) -> torch.Tensor:
        second_moment = context.activation_second_moment(
            size=self._input_size(context, column_end),
            device=values.device,
        )
        diagonal = second_moment[column_start:column_end].clamp_min(self.eps)
        diagonal = diagonal / diagonal.mean().clamp_min(self.eps)
        return diagonal.pow(self.exponent)

    @staticmethod
    def _input_size(context: CodebookContext, fallback: int) -> int:
        for statistic in (context.activation_mean, context.activation_variance):
            if statistic is not None:
                return statistic.numel()
        return fallback

    def __repr__(self) -> str:
        return f"{super().__repr__()[:-1]}, exponent={self.exponent})"
