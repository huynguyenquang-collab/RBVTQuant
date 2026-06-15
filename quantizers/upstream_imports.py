"""Load quantization code directly from the nested upstream repositories."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEANQUANT_ROOT = REPO_ROOT / "LeanQuant"
SQUEEZELLM_ROOT = REPO_ROOT / "SqueezeLLM"


@contextmanager
def _python_path(path: Path):
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass


def _verify_source(module, root: Path):
    source = Path(module.__file__).resolve()
    if root.resolve() not in source.parents:
        raise ImportError(f"Loaded {module.__name__} from {source}, expected under {root}")


def load_leanquant_upstream():
    with _python_path(LEANQUANT_ROOT):
        module = importlib.import_module("lean_quantizer")
    _verify_source(module, LEANQUANT_ROOT)
    return module.LeanQuant, module.Quantizer


def load_squeezellm_kmeans():
    with _python_path(SQUEEZELLM_ROOT):
        module = importlib.import_module("quantization.nuq")
    _verify_source(module, SQUEEZELLM_ROOT)
    return module.kmeans_fit


def load_squeezellm_model_parse():
    with _python_path(SQUEEZELLM_ROOT):
        module = importlib.import_module("squeezellm.model_parse")
    _verify_source(module, SQUEEZELLM_ROOT)
    return module


__all__ = [
    "LEANQUANT_ROOT",
    "SQUEEZELLM_ROOT",
    "load_leanquant_upstream",
    "load_squeezellm_kmeans",
    "load_squeezellm_model_parse",
]
