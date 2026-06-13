"""SqueezeLLM dense sensitivity-weighted LUT adapter for RBVTQuant."""

from __future__ import annotations

import torch

from .base_codebook import BaseCodebook, CodebookContext


class SqueezeLLMCodebook(BaseCodebook):
    """Dense-only SqueezeLLM codebook using Fisher or activation sensitivity."""

    def __init__(
        self,
        bits: int,
        group_size: int = -1,
        n_iters: int = 20,
        fit_row_chunk: int = 32,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            n_iters=n_iters,
            fit_row_chunk=fit_row_chunk,
        )
        self.name = f"squeezellm{bits}"

    def sample_weights(
        self,
        values: torch.Tensor,
        context: CodebookContext,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
    ) -> torch.Tensor:
        if context.sensitivity is not None:
            sensitivity = context.sensitivity
            if sensitivity.ndim != 2:
                raise ValueError(
                    "SqueezeLLM sensitivity must be a weight-shaped matrix, "
                    f"got {tuple(sensitivity.shape)}"
                )
            weights = sensitivity[
                row_start:row_end,
                column_start:column_end,
            ].to(device=values.device, dtype=torch.float32)
            if weights.shape != values.shape:
                raise ValueError(
                    f"Sensitivity slice {tuple(weights.shape)} does not match "
                    f"weight slice {tuple(values.shape)}"
                )
        else:
            input_size = self._input_size(context, column_end)
            second_moment = context.activation_second_moment(
                size=input_size,
                device=values.device,
            )
            # Under a factorized Fisher approximation, the output-channel
            # factor is constant within each row and does not affect k-means.
            weights = second_moment[column_start:column_end]

        return weights * values.ne(0).to(torch.float32)

    @staticmethod
    def _input_size(context: CodebookContext, fallback: int) -> int:
        for statistic in (context.activation_mean, context.activation_variance):
            if statistic is not None:
                return statistic.numel()
        return fallback
