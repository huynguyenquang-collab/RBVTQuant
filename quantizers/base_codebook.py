"""Common interface for learned scalar codebooks used by RBVTQuant."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from .base_quantizer import BaseQuantizer, QuantResult


@dataclass
class CodebookContext:
    """Calibration statistics available while fitting a layer codebook."""

    activation_mean: Optional[torch.Tensor] = None
    activation_variance: Optional[torch.Tensor] = None
    sensitivity: Optional[torch.Tensor] = None

    def activation_second_moment(
        self,
        size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.activation_mean is None and self.activation_variance is None:
            return torch.ones(size, device=device, dtype=torch.float32)

        if self.activation_mean is None:
            mean = torch.zeros(size, device=device, dtype=torch.float32)
        else:
            mean = self.activation_mean.to(device=device, dtype=torch.float32)

        if self.activation_variance is None:
            variance = torch.zeros(size, device=device, dtype=torch.float32)
        else:
            variance = self.activation_variance.to(device=device, dtype=torch.float32)

        if mean.numel() != size or variance.numel() != size:
            raise ValueError(
                "Activation statistics must match the layer input dimension: "
                f"expected {size}, got mean={mean.numel()}, variance={variance.numel()}"
            )
        return (variance + mean.square()).clamp_min(0.0)


class BaseCodebook(BaseQuantizer, ABC):
    """Base quantizer for per-row, optionally grouped, learned codebooks."""

    def __init__(
        self,
        bits: int,
        group_size: int = -1,
        n_iters: int = 20,
        fit_row_chunk: int = 32,
        eps: float = 1e-12,
    ):
        if bits not in (3, 4):
            raise ValueError(f"Codebook bits must be 3 or 4, got {bits}")
        if group_size == 0 or group_size < -1:
            raise ValueError(f"group_size must be -1 or positive, got {group_size}")
        if n_iters <= 0:
            raise ValueError(f"n_iters must be positive, got {n_iters}")
        if fit_row_chunk <= 0:
            raise ValueError(f"fit_row_chunk must be positive, got {fit_row_chunk}")

        # The actual block size is resolved from the input width when group_size=-1.
        super().__init__(bits=bits, block_size=group_size)
        self.group_size = group_size
        self.num_levels = 2**bits
        self.n_iters = n_iters
        self.fit_row_chunk = fit_row_chunk
        self.eps = eps
        self.context = CodebookContext()
        self._q_levels = torch.linspace(-1.0, 1.0, self.num_levels)

    @property
    def q_levels(self) -> torch.Tensor:
        return self._q_levels

    def set_context(self, context: CodebookContext | None) -> "BaseCodebook":
        self.context = context or CodebookContext()
        return self

    @abstractmethod
    def sample_weights(
        self,
        values: torch.Tensor,
        context: CodebookContext,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
    ) -> torch.Tensor:
        """Return non-negative k-means weights for a row group."""

    def initial_centers(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Initialize centers with deterministic weighted quantiles."""

        order = torch.argsort(values, dim=1)
        sorted_values = torch.gather(values, 1, order)
        sorted_weights = torch.gather(weights, 1, order)
        cumulative = torch.cumsum(sorted_weights, dim=1)
        total = cumulative[:, -1:].clamp_min(self.eps)
        probabilities = torch.linspace(
            0.0,
            1.0,
            self.num_levels + 2,
            device=values.device,
            dtype=torch.float32,
        )[1:-1]
        targets = total * probabilities.unsqueeze(0)
        positions = torch.searchsorted(cumulative.contiguous(), targets.contiguous())
        positions = positions.clamp_max(values.shape[1] - 1)
        return torch.gather(sorted_values, 1, positions)

    def _normalize_sample_weights(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        weights = weights.to(device=values.device, dtype=torch.float32)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0).expand(values.shape[0], -1)
        if weights.shape != values.shape:
            raise ValueError(
                f"sample_weights returned {tuple(weights.shape)} for values "
                f"with shape {tuple(values.shape)}"
            )

        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights.clamp_min(0.0)
        row_sum = weights.sum(dim=1, keepdim=True)
        fallback = torch.ones_like(weights)
        weights = torch.where(row_sum > self.eps, weights, fallback)
        return weights / weights.mean(dim=1, keepdim=True).clamp_min(self.eps)

    def _fit_weighted_kmeans(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        weights = self._normalize_sample_weights(values, weights)
        centers = self.initial_centers(values, weights)

        for _ in range(self.n_iters):
            distances = (values.unsqueeze(-1) - centers.unsqueeze(1)).abs()
            assignments = distances.argmin(dim=-1)
            del distances

            weighted_sum = torch.zeros_like(centers)
            mass = torch.zeros_like(centers)
            weighted_sum.scatter_add_(1, assignments, values * weights)
            mass.scatter_add_(1, assignments, weights)
            updated = weighted_sum / mass.clamp_min(self.eps)
            centers = torch.where(mass > self.eps, updated, centers)
            centers, _ = torch.sort(centers, dim=1)

        return centers

    @torch.no_grad()
    def quantize(self, W: torch.Tensor, row_chunk: int = 1024) -> QuantResult:
        if W.ndim != 2:
            raise ValueError(f"Codebook quantization expects a matrix, got {tuple(W.shape)}")

        device = W.device
        out_features, in_features = W.shape
        block_size = in_features if self.group_size == -1 else self.group_size
        n_blocks = (in_features + block_size - 1) // block_size
        effective_row_chunk = min(row_chunk, self.fit_row_chunk)

        W_dequant = torch.empty_like(W)
        indices = torch.empty(
            out_features,
            in_features,
            dtype=torch.long,
            device=device,
        )
        block_codebooks = torch.empty(
            out_features,
            n_blocks,
            self.num_levels,
            dtype=torch.float32,
            device=device,
        )
        block_scales = torch.empty(
            out_features,
            n_blocks,
            dtype=torch.float32,
            device=device,
        )

        for row_start in range(0, out_features, effective_row_chunk):
            row_end = min(row_start + effective_row_chunk, out_features)
            rows = W[row_start:row_end].float()

            for block_index in range(n_blocks):
                column_start = block_index * block_size
                column_end = min(column_start + block_size, in_features)
                values = rows[:, column_start:column_end]
                weights = self.sample_weights(
                    values=values,
                    context=self.context,
                    row_start=row_start,
                    row_end=row_end,
                    column_start=column_start,
                    column_end=column_end,
                )
                centers = self._fit_weighted_kmeans(values, weights)

                distances = (values.unsqueeze(-1) - centers.unsqueeze(1)).abs()
                block_indices = distances.argmin(dim=-1)
                dequantized = torch.gather(centers, 1, block_indices)

                W_dequant[row_start:row_end, column_start:column_end] = dequantized.to(W.dtype)
                indices[row_start:row_end, column_start:column_end] = block_indices
                block_codebooks[row_start:row_end, block_index] = centers
                block_scales[row_start:row_end, block_index] = (
                    centers.abs().amax(dim=1).clamp_min(self.eps)
                )

        return QuantResult(
            W_dequant=W_dequant,
            indices=indices,
            q_levels=self.q_levels.to(device),
            block_scales=block_scales,
            block_size=block_size,
            block_codebooks=block_codebooks,
            block_zeros=None,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, bits={self.bits}, "
            f"group_size={self.group_size}, n_iters={self.n_iters}, "
            f"fit_row_chunk={self.fit_row_chunk})"
        )
