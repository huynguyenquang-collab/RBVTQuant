"""Calibration sampling shared with the LeanQuant and SqueezeLLM upstream code."""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import torch
from datasets import load_dataset


def load_upstream_c4_tokens(
    tokenizer,
    n_samples: int,
    seqlen: int,
    seed: int,
    cache_dir: str | Path,
) -> list[torch.Tensor]:
    """Reproduce the get_c4/get_c4_new training sampler used by both repos."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"upstream_c4_n{n_samples}_len{seqlen}_seed{seed}.pkl"
    )
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    dataset = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        trust_remote_code=True,
    )
    random.seed(seed)
    samples = []
    for _ in range(n_samples):
        while True:
            document_index = random.randint(0, len(dataset) - 1)
            encoded = tokenizer(dataset[document_index]["text"], return_tensors="pt")
            if encoded.input_ids.shape[1] >= seqlen:
                break
        start = random.randint(0, encoded.input_ids.shape[1] - seqlen - 1)
        samples.append(encoded.input_ids[:, start : start + seqlen].cpu())

    with cache_path.open("wb") as handle:
        pickle.dump(samples, handle)
    return samples


__all__ = ["load_upstream_c4_tokens"]
