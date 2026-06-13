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
) -> BaseCodebook:
    key = name.lower().replace("-", "").replace("_", "")
    common = {
        "bits": bits,
        "group_size": group_size,
        "n_iters": n_iters,
        "fit_row_chunk": fit_row_chunk,
    }
    if key == "leanquant":
        return LeanQuantCodebook(
            **common,
            exponent=leanquant_exponent,
        )
    if key == "squeezellm":
        return SqueezeLLMCodebook(**common)
    raise ValueError("Unknown codebook {!r}. Available: leanquant, squeezellm".format(name))


__all__ = [
    "BaseCodebook",
    "LeanQuantCodebook",
    "SqueezeLLMCodebook",
    "get_codebook",
]
