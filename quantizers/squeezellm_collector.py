"""Fisher collection plus direct invocation of SqueezeLLM's upstream LUT code."""

from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .codebook_store import CodebookStore
from .sensitivity_store import SensitivityStore
from .upstream_imports import load_squeezellm_kmeans, load_squeezellm_model_parse


def _squeezellm_linears(model) -> list[tuple[str, nn.Linear]]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("SqueezeLLM Fisher collection currently requires a Llama model")
    model_parse = load_squeezellm_model_parse()
    names = model_parse.get_sequential("llama")
    result = []
    for layer_index, layer in enumerate(model.model.layers):
        for relative_name in names:
            module = layer.get_submodule(relative_name)
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"Expected Linear at model.layers.{layer_index}.{relative_name}"
                )
            result.append((f"model.layers.{layer_index}.{relative_name}", module))
    return result


def collect_squeezellm_fisher(
    model,
    token_samples: list[torch.Tensor],
    output_dir: str | Path,
    device: str,
) -> Path:
    """Match SqueezeLLM-gradients: square each backward gradient and accumulate."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete", False):
            print(f"Reusing SqueezeLLM Fisher cache: {output_dir}")
            return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    linears = _squeezellm_linears(model)
    handles = [
        module.weight.register_hook(lambda gradient: gradient.pow(2))
        for _, module in linears
    ]
    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False
    model.train()
    model.zero_grad(set_to_none=True)

    try:
        for input_ids in tqdm(token_samples, desc="Collecting SqueezeLLM Fisher"):
            inputs = input_ids.to(device)
            outputs = model(input_ids=inputs, labels=inputs)
            outputs.loss.backward()
    finally:
        for handle in handles:
            handle.remove()
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache

    layers = {}
    for layer_name, module in linears:
        if module.weight.grad is None:
            raise RuntimeError(f"No Fisher gradient collected for {layer_name}")
        filename = layer_name.replace(".", "_") + ".pt"
        torch.save(module.weight.grad.detach().float().cpu(), output_dir / filename)
        layers[f"{layer_name}.weight"] = filename
    model.zero_grad(set_to_none=True)

    manifest_path.write_text(
        json.dumps(
            {
                "complete": True,
                "format": "sum_of_per_sample_gradient_squares",
                "layers": layers,
                "num_examples": len(token_samples),
                "sequence_length": int(token_samples[0].shape[1]),
                "seed": 0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir


def collect_squeezellm_codebooks(
    model,
    sensitivity_store: SensitivityStore,
    store: CodebookStore,
    bits: int,
):
    """Generate dense-only LUTs with SqueezeLLM/quantization/nuq.py::kmeans_fit."""

    metadata = {
        "algorithm": "squeezellm_direct_upstream_dense_fisher_kmeans",
        "source": "SqueezeLLM/quantization/nuq.py",
        "bits": bits,
        "fisher": str(sensitivity_store.path),
        "sparsity": 0.0,
    }
    if store.complete:
        store.validate(metadata)
        print(f"Reusing SqueezeLLM upstream codebook cache: {store.root}")
        return
    store.initialize(metadata)

    kmeans_fit = load_squeezellm_kmeans()
    linears = _squeezellm_linears(model)
    workers = os.cpu_count() or 1
    print(
        "Building SqueezeLLM dense-only LUTs with upstream nuq.py | "
        f"layers={len(linears)} | workers={workers}"
    )
    with Pool(workers) as pool:
        for layer_name, module in tqdm(linears, desc="SqueezeLLM upstream LUTs"):
            weight = module.weight.detach().float().cpu()
            sensitivity = sensitivity_store.get(layer_name, weight.shape)
            weight_np = weight.numpy()
            sensitivity_np = sensitivity.numpy()
            tasks = []
            for row, gradient_row in zip(weight_np, sensitivity_np):
                sample_weight = gradient_row * (row != 0)
                if np.sum(sample_weight) == 0:
                    sample_weight = np.ones_like(sample_weight)
                tasks.append((row.reshape(-1, 1), sample_weight, 2**bits))
            results = list(pool.imap(kmeans_fit, tasks))
            centers = torch.from_numpy(
                np.stack([result[0] for result in results])
            ).float()
            centers = torch.sort(centers, dim=1).values
            store.put(layer_name, centers.unsqueeze(1))

    store.mark_complete()


__all__ = ["collect_squeezellm_codebooks", "collect_squeezellm_fisher"]
