"""LeanQuant loss-error-aware codebook reproduced from the upstream flow."""

from __future__ import annotations

import numpy as np
import torch

from .base_codebook import BaseCodebook, CodebookContext


class LeanQuantCodebook(BaseCodebook):
    """Per-channel LeanQuant grids fitted with the upstream inverse-Hessian rule."""

    def __init__(
        self,
        bits: int,
        group_size: int = -1,
        n_iters: int = 100,
        fit_row_chunk: int = 32,
        exponent: float = 4.0,
        kmeans_seed: int = 0,
        percdamp: float = 0.1,
        act_order: bool = True,
        block_size: int = 128,
    ):
        if group_size != -1:
            raise ValueError(
                "Upstream LeanQuant non-uniform codebooks use one grid per output "
                "channel; group_size must be -1"
            )
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
        self.kmeans_seed = kmeans_seed
        self.percdamp = percdamp
        self.act_order = act_order
        self.gptq_block_size = block_size

    def sample_weights(
        self,
        values: torch.Tensor,
        context: CodebookContext,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
    ) -> torch.Tensor:
        del values, row_start, row_end
        if context.hessian is None:
            raise RuntimeError(
                "LeanQuant requires a full activation Hessian or precomputed "
                "upstream codebook centers"
            )
        weights, _ = self._inverse_hessian_weights(
            context.hessian,
            column_start=column_start,
            column_end=column_end,
        )
        return weights

    def fit_centers(
        self,
        values: torch.Tensor,
        context: CodebookContext,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
    ) -> torch.Tensor:
        del row_start, row_end
        if context.hessian is None:
            raise RuntimeError(
                "LeanQuant requires a full activation Hessian or precomputed "
                "upstream codebook centers"
            )
        sample_weight, permutation = self._inverse_hessian_weights(
            context.hessian,
            column_start=column_start,
            column_end=column_end,
        )
        if permutation is not None:
            values = values[:, permutation]
        centers = self._fit_sklearn(values, sample_weight)
        return torch.sort(centers, dim=1).values

    def _fit_sklearn(
        self,
        values: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for upstream-compatible LeanQuant grids"
            ) from exc

        values_np = values.detach().float().cpu().numpy()
        weights_np = sample_weight.detach().float().cpu().numpy()
        results = []
        for row in values_np:
            init = (
                np.linspace(row.min(), row.max(), num=self.num_levels)[:, None]
                if self.num_levels <= 8
                else "k-means++"
            )
            fitted = KMeans(
                n_clusters=self.num_levels,
                init=init,
                n_init="auto",
                random_state=self.kmeans_seed,
                max_iter=100,
                tol=1e-6,
            ).fit(row[:, None], sample_weight=weights_np)
            results.append(fitted.cluster_centers_.reshape(-1))
        return torch.from_numpy(np.stack(results)).to(
            device=values.device,
            dtype=torch.float32,
        )

    def _inverse_hessian_weights(
        self,
        hessian: torch.Tensor,
        column_start: int = 0,
        column_end: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        H = hessian.detach().float().clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1

        permutation = None
        if self.act_order:
            permutation = torch.argsort(torch.diag(H), descending=True)
            H = H[permutation][:, permutation]

        damp = self.percdamp * torch.mean(torch.diag(H))
        diagonal = torch.arange(H.shape[0], device=H.device)
        H[diagonal, diagonal] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)
        weights = torch.diagonal(Hinv).pow(-self.exponent)

        if column_end is not None and (column_start != 0 or column_end != H.shape[0]):
            if self.act_order:
                raise ValueError("Grouped LeanQuant grids are not supported with act-order")
            weights = weights[column_start:column_end]
        return weights, permutation

    @torch.no_grad()
    def fit_upstream_layer(
        self,
        weight: torch.Tensor,
        hessian: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return upstream centers and GPTQ-propagated weights for shadow calibration."""

        W = weight.detach().float().clone()
        H = hessian.detach().float().clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        permutation = None
        inverse_permutation = None
        if self.act_order:
            permutation = torch.argsort(torch.diag(H), descending=True)
            W = W[:, permutation]
            H = H[permutation][:, permutation]
            inverse_permutation = torch.argsort(permutation)

        damp = self.percdamp * torch.mean(torch.diag(H))
        diagonal = torch.arange(H.shape[0], device=H.device)
        H[diagonal, diagonal] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        sample_weight = torch.diagonal(Hinv).pow(-self.exponent)
        raw_centers = self._fit_sklearn(W, sample_weight)
        Q = torch.zeros_like(W)

        for block_start in range(0, W.shape[1], self.gptq_block_size):
            block_end = min(block_start + self.gptq_block_size, W.shape[1])
            count = block_end - block_start
            W1 = W[:, block_start:block_end].clone()
            Q1 = torch.zeros_like(W1)
            errors = torch.zeros_like(W1)
            Hinv1 = Hinv[block_start:block_end, block_start:block_end]

            for offset in range(count):
                column = W1[:, offset]
                divisor = Hinv1[offset, offset]
                codes = torch.argmin(
                    (raw_centers - column[:, None]).abs(),
                    dim=1,
                    keepdim=True,
                )
                quantized = torch.gather(raw_centers, 1, codes).flatten()
                Q1[:, offset] = quantized
                error = (column - quantized) / divisor
                W1[:, offset:] -= error.unsqueeze(1).matmul(
                    Hinv1[offset, offset:].unsqueeze(0)
                )
                errors[:, offset] = error

            Q[:, block_start:block_end] = Q1
            W[:, block_end:] -= errors.matmul(Hinv[block_start:block_end, block_end:])

        if inverse_permutation is not None:
            Q = Q[:, inverse_permutation]
        sorted_centers = torch.sort(raw_centers, dim=1).values
        return sorted_centers, Q.to(weight.dtype)

    def __repr__(self) -> str:
        return (
            f"{super().__repr__()[:-1]}, exponent={self.exponent}, "
            f"percdamp={self.percdamp}, act_order={self.act_order}, "
            f"kmeans_seed={self.kmeans_seed})"
        )
