"""Layer-sequential LeanQuant Hessian and codebook collection."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .codebook_store import CodebookStore
from .leanquant_codebook import LeanQuantCodebook


class _StopCapture(Exception):
    pass


class _InputCatcher(nn.Module):
    def __init__(self, module: nn.Module, inputs: list[torch.Tensor], state: dict):
        super().__init__()
        self.module = module
        self.inputs = inputs
        self.state = state

    def forward(self, hidden_states, **kwargs):
        self.inputs.append(hidden_states.detach())
        self.state["kwargs"] = kwargs
        raise _StopCapture


class _HessianAccumulator:
    """Byte-for-byte formula used by LeanQuant.LeanQuant.add_batch."""

    def __init__(self, columns: int, device: torch.device):
        self.H = torch.zeros((columns, columns), device=device)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor):
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        batch_size = inputs.shape[0]
        if inputs.ndim == 3:
            inputs = inputs.reshape(-1, inputs.shape[-1])
        inputs = inputs.t()
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inputs = math.sqrt(2 / self.nsamples) * inputs.float()
        self.H += inputs.matmul(inputs.t())


def _forward_layer(layer: nn.Module, hidden_states: torch.Tensor, kwargs: dict):
    output = layer(hidden_states, **kwargs)
    return output[0] if isinstance(output, (tuple, list)) else output


def _get_subsets(layer: nn.Module) -> list[list[str]]:
    groups = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
    for names in groups:
        for name in names:
            module = layer.get_submodule(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"LeanQuant expected Linear module {name!r}")
    return groups


@torch.no_grad()
def collect_leanquant_codebooks(
    model,
    token_samples: list[torch.Tensor],
    store: CodebookStore,
    codebook: LeanQuantCodebook,
    device: str,
):
    """Run LeanQuant's true-sequential shadow pass and persist its exact grids."""

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("LeanQuant collection currently requires a Llama model")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = {
        "algorithm": "leanquant_upstream_true_sequential",
        "bits": codebook.bits,
        "exponent": codebook.exponent,
        "percdamp": codebook.percdamp,
        "act_order": codebook.act_order,
        "kmeans_seed": codebook.kmeans_seed,
        "block_size": codebook.gptq_block_size,
        "num_examples": len(token_samples),
        "sequence_length": int(token_samples[0].shape[1]),
    }
    if store.complete:
        if store.metadata != metadata:
            raise ValueError(
                f"LeanQuant cache metadata mismatch: {store.metadata} != {metadata}"
            )
        print(f"Reusing LeanQuant codebook cache: {store.root}")
        return
    store.initialize(metadata)

    layers = model.model.layers
    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False
    model.eval()

    captured_inputs: list[torch.Tensor] = []
    capture_state: dict = {}
    first_layer = layers[0]
    layers[0] = _InputCatcher(first_layer, captured_inputs, capture_state)
    try:
        for input_ids in tqdm(token_samples, desc="Capturing LeanQuant inputs"):
            try:
                model(input_ids=input_ids.to(device), use_cache=False)
            except _StopCapture:
                pass
    finally:
        layers[0] = first_layer

    if len(captured_inputs) != len(token_samples):
        raise RuntimeError(
            f"Captured {len(captured_inputs)} LeanQuant inputs for "
            f"{len(token_samples)} samples"
        )
    layer_kwargs = capture_state.get("kwargs", {})
    inps = captured_inputs

    for layer_index, layer in enumerate(layers):
        print(f"LeanQuant shadow layer {layer_index + 1}/{len(layers)}")
        subsets = _get_subsets(layer)
        outs: list[torch.Tensor] = []

        for names in subsets:
            accumulators = {
                name: _HessianAccumulator(
                    layer.get_submodule(name).weight.shape[1],
                    layer.get_submodule(name).weight.device,
                )
                for name in names
            }
            handles = []
            for name in names:
                accumulator = accumulators[name]
                module = layer.get_submodule(name)

                def add_batch(_module, inputs, _output, target=accumulator):
                    target.add_batch(inputs[0].detach())

                handles.append(module.register_forward_hook(add_batch))

            try:
                for sample in inps:
                    _forward_layer(layer, sample, layer_kwargs)
            finally:
                for handle in handles:
                    handle.remove()

            for name in names:
                module = layer.get_submodule(name)
                centers, shadow_weight = codebook.fit_upstream_layer(
                    module.weight.data,
                    accumulators[name].H,
                )
                full_name = f"model.layers.{layer_index}.{name}"
                store.put(full_name, centers.unsqueeze(1))
                module.weight.data = shadow_weight
                del accumulators[name], centers, shadow_weight
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for sample in inps:
            outs.append(_forward_layer(layer, sample, layer_kwargs).detach())
        inps = outs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    store.mark_complete()
    if original_use_cache is not None:
        model.config.use_cache = original_use_cache


__all__ = ["collect_leanquant_codebooks"]
