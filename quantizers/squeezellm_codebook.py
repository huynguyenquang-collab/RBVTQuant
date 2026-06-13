"""Dense-only SqueezeLLM Fisher-weighted LUT reproduced from upstream."""

from __future__ import annotations

import numpy as np
import torch

from .base_codebook import BaseCodebook, CodebookContext


class SqueezeLLMCodebook(BaseCodebook):
    """One Fisher-weighted sklearn KMeans LUT per output channel."""

    def __init__(
        self,
        bits: int,
        group_size: int = -1,
        n_iters: int = 50,
        fit_row_chunk: int = 32,
    ):
        if group_size != -1:
            raise ValueError(
                "Upstream dense-only SqueezeLLM uses one LUT per output channel; "
                "group_size must be -1"
            )
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
        if context.sensitivity is None:
            raise RuntimeError(
                "SqueezeLLM requires the upstream Fisher gradient-square tensor; "
                "activation proxies are not supported"
            )
        sensitivity = context.sensitivity
        if sensitivity.ndim != 2:
            raise ValueError(
                "SqueezeLLM sensitivity must be weight-shaped, "
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
        weights = weights * values.ne(0).to(torch.float32)
        empty = weights.sum(dim=1, keepdim=True) == 0
        return torch.where(empty, torch.ones_like(weights), weights)

    def fit_centers(
        self,
        values: torch.Tensor,
        context: CodebookContext,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
    ) -> torch.Tensor:
        weights = self.sample_weights(
            values,
            context,
            row_start,
            row_end,
            column_start,
            column_end,
        )
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for upstream-compatible SqueezeLLM LUTs"
            ) from exc

        values_np = values.detach().float().cpu().numpy()
        weights_np = weights.detach().float().cpu().numpy()
        centers = []
        for row, sample_weight in zip(values_np, weights_np):
            fitted = KMeans(
                n_clusters=self.num_levels,
                random_state=0,
                n_init="auto",
                max_iter=50,
            ).fit(row[:, None], sample_weight=sample_weight)
            centers.append(fitted.cluster_centers_.reshape(-1))
        result = torch.from_numpy(np.stack(centers)).to(
            device=values.device,
            dtype=torch.float32,
        )
        return torch.sort(result, dim=1).values
