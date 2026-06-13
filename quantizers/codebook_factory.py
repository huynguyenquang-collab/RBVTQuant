"""Factory for optional learned codebooks without changing the core registry."""

from __future__ import annotations

from .base_codebook import BaseCodebook
from .leanquant_codebook import LeanQuantCodebook
from .squeezellm_codebook import SqueezeLLMCodebook


def get_codebook(
    name: str,
    bits: int,
    group_size: int = -1,
    n_iters: int = 20,
    fit_row_chunk: int = 32,
    leanquant_exponent: float = 4.0,
    leanquant_percdamp: float = 0.1,
    leanquant_act_order: bool = True,
    kmeans_seed: int = 0,
) -> BaseCodebook:
    key = name.lower().replace("-", "").replace("_", "")
    common = {
        "bits": bits,
        "group_size": group_size,
        "fit_row_chunk": fit_row_chunk,
    }
    if key == "leanquant":
        return LeanQuantCodebook(
            **common,
            n_iters=100,
            exponent=leanquant_exponent,
            percdamp=leanquant_percdamp,
            act_order=leanquant_act_order,
            kmeans_seed=kmeans_seed,
        )
    if key == "squeezellm":
        return SqueezeLLMCodebook(**common, n_iters=50)
    raise ValueError("Unknown codebook {!r}. Available: leanquant, squeezellm".format(name))


__all__ = [
    "BaseCodebook",
    "LeanQuantCodebook",
    "SqueezeLLMCodebook",
    "get_codebook",
]
