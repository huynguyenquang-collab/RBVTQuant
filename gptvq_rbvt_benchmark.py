"""Run upstream GPTVQ-1D and GPTVQ-1D+RBVT on a Llama-like HF model.

This file intentionally imports Qualcomm-AI-research/gptvq as an external
checkout from ./GPTVQ. The quantization loop mirrors GPTVQ's llama.py, with the
minimum extra bookkeeping needed to convert 1D VQ centroids/assignments into
RBVT's scalar QuantResult format.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
GPTVQ_ROOT = ROOT / "GPTVQ"
if not GPTVQ_ROOT.exists():
    raise RuntimeError(
        "Missing ./GPTVQ. Clone upstream first: "
        "git clone https://github.com/Qualcomm-AI-research/gptvq.git GPTVQ"
    )
sys.path.insert(0, str(GPTVQ_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transformers  # noqa: E402

if not hasattr(transformers, "Conv1D"):
    from transformers.pytorch_utils import Conv1D  # noqa: E402

    transformers.Conv1D = Conv1D

from gptq import GPTQ  # type: ignore  # noqa: E402
from modelutils import find_layers  # type: ignore  # noqa: E402
from vq_quant import VQQuantizer  # type: ignore  # noqa: E402

from calibration_utils import load_calibration_data  # noqa: E402
from eval_perplexity import RBVTSlidingWindowEvaluator  # noqa: E402
from lm_eval_runner import LMEvalHarnessRunner  # noqa: E402
from quantizers import apply_rbvt  # noqa: E402
from quantizers.base_quantizer import QuantResult  # noqa: E402
from runtime_utils import build_model_slug, load_runtime_env, resolve_hf_token  # noqa: E402


def _hf_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"{device=} requested but CUDA is not available")
    return torch.device(device)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _linear_key(layer_idx: int, name: str) -> str:
    return f"model.layers.{layer_idx}.{name}"


def _layer_call(layer: nn.Module, hidden: torch.Tensor, cache: dict) -> torch.Tensor:
    return layer(hidden, **cache.get("layer_kwargs", {}))[0]


def _make_calibration_batches(tokenizer, texts: Iterable[str], seqlen: int) -> list[tuple[torch.Tensor]]:
    batches = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=seqlen,
        )
        batches.append((encoded["input_ids"],))
    return batches


def _capture_first_layer_inputs(model, batches, device: torch.device, nsamples: int, seqlen: int):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.hidden_size
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=device)
    cache = {"i": 0, "layer_kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"] = dict(kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in batches[:nsamples]:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return inps, torch.zeros_like(inps), cache


def _sequential_groups(full: dict[str, nn.Module], true_sequential: bool) -> list[list[str]]:
    if not true_sequential:
        return [[k for k in list(full.keys()) if "block_sparse_moe.gate" not in k]]
    groups = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
    return [[name for name in group if name in full] for group in groups if any(name in full for name in group)]


def _make_vq_quantizer(args) -> VQQuantizer:
    quantizer = VQQuantizer(
        vq_dim=1,
        columns_per_group=None,
        vq_scaling_blocksize=-1,
        vq_scaling_norm="max",
        vq_scaling_n_bits=4,
        vq_scaling_domain="log",
        kmeans_init_method=args.kmeans_init_method,
        assignment_chunk_size=args.assignment_chunk_size,
        kmeans_iters=args.kmeans_iters,
        codebook_bitwidth=None,
        quantize_per_codebook=True,
        quantize_during_kmeans=False,
        n_subsample=args.kpp_n_subsample,
    )
    quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
    return quantizer


def _gptvq_quant_result(
    *,
    W_dequant: torch.Tensor,
    assignments: list[list[torch.Tensor]],
    centroids: list[torch.Tensor],
    bits: int,
    block_size: int,
) -> QuantResult:
    if len(assignments) != len(centroids):
        raise RuntimeError(
            f"GPTVQ assignments/centroids mismatch: {len(assignments)} vs {len(centroids)}"
        )
    device = W_dequant.device
    rows, cols = W_dequant.shape
    n_blocks = len(centroids)
    K = 2**bits

    all_indices = []
    block_codebooks = torch.empty((rows, n_blocks, K), dtype=torch.float32, device=device)

    for block_idx, (block_assignments, block_centroids) in enumerate(zip(assignments, centroids)):
        centers = block_centroids.to(device=device, dtype=torch.float32).squeeze(-1)
        if centers.shape != (rows, K):
            raise RuntimeError(
                f"GPTVQ 1D centers for block {block_idx} have shape {tuple(centers.shape)}, "
                f"expected {(rows, K)}"
            )

        sorted_centers, old_from_new = torch.sort(centers, dim=1)
        new_from_old = torch.empty_like(old_from_new)
        new_from_old.scatter_(
            dim=1,
            index=old_from_new,
            src=torch.arange(K, device=device).view(1, K).expand(rows, K),
        )

        idx = torch.cat(
            [assignment.to(device=device, dtype=torch.long).reshape(rows, -1) for assignment in block_assignments],
            dim=1,
        )
        idx = torch.gather(new_from_old, dim=1, index=idx)
        all_indices.append(idx)
        block_codebooks[:, block_idx, :] = sorted_centers

    indices = torch.cat(all_indices, dim=1)
    if indices.shape != (rows, cols):
        raise RuntimeError(f"GPTVQ indices have shape {tuple(indices.shape)}, expected {(rows, cols)}")

    return QuantResult(
        W_dequant=W_dequant,
        indices=indices,
        q_levels=torch.linspace(-1.0, 1.0, K, device=device),
        block_scales=block_codebooks.abs().amax(dim=-1).clamp_min(1e-12),
        block_size=block_size,
        block_codebooks=block_codebooks,
        block_zeros=None,
    )


@torch.no_grad()
def quantize_model_gptvq_1d(
    model,
    tokenizer,
    calib_texts: list[str],
    args,
    use_rbvt: bool,
    gptvq_state: dict[str, torch.Tensor] | None = None,
) -> dict:
    device = _hf_device(args.device)
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("GPTVQ runner expects a Llama-like model with model.layers")

    model.seqlen = args.max_length
    batches = _make_calibration_batches(tokenizer, calib_texts, args.max_length)
    actual_n_calib = min(args.n_calib, len(batches))
    if actual_n_calib <= 0:
        raise RuntimeError("No calibration batches were produced.")
    if actual_n_calib != args.n_calib:
        print(f"Using {actual_n_calib} calibration samples; requested {args.n_calib}.")

    inps, outs, cache = _capture_first_layer_inputs(
        model=model,
        batches=batches,
        device=device,
        nsamples=actual_n_calib,
        seqlen=args.max_length,
    )

    layers = model.model.layers
    use_cache = model.config.use_cache
    model.config.use_cache = False

    totals = {
        "flips": 0,
        "candidates": 0,
        "boundary_kept": 0,
        "bias_before": 0.0,
        "bias_after": 0.0,
        "objective_before": 0.0,
        "objective_after": 0.0,
        "variance_increase": 0.0,
    }
    quantized_layers = 0
    tick = time.time()

    for layer_idx in range(len(layers)):
        print(f"\n=== GPTVQ layer {layer_idx + 1}/{len(layers)} ===")
        layer = layers[layer_idx].to(device)
        full = find_layers(layer)

        for names in _sequential_groups(full, args.true_sequential):
            subset = {name: full[name] for name in names}
            gptq = {}
            stat_sum: Dict[str, torch.Tensor] = {}
            stat_sumsq: Dict[str, torch.Tensor] = {}
            stat_count: Dict[str, int] = {}

            for name, module in subset.items():
                gptq[name] = GPTQ(module)
                gptq[name].quantizer = _make_vq_quantizer(args)

            def add_batch(name):
                key = _linear_key(layer_idx, name)

                def hook(_module, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    gptq[name].add_batch(x.data, out.data)
                    if use_rbvt:
                        x_float = x.reshape(-1, x.shape[-1]).detach().float()
                        stat_sum[key] = stat_sum.get(key, torch.zeros(x_float.shape[-1])).to(x_float.device)
                        stat_sum[key] += x_float.sum(dim=0)
                        stat_sumsq[key] = stat_sumsq.get(key, torch.zeros(x_float.shape[-1])).to(x_float.device)
                        stat_sumsq[key] += (x_float * x_float).sum(dim=0)
                        stat_count[key] = stat_count.get(key, 0) + x_float.shape[0]

                return hook

            handles = [module.register_forward_hook(add_batch(name)) for name, module in subset.items()]
            try:
                for sample_idx in range(actual_n_calib):
                    outs[sample_idx] = _layer_call(layer, inps[sample_idx].unsqueeze(0), cache)
            finally:
                for handle in handles:
                    handle.remove()

            for name, module in subset.items():
                key = _linear_key(layer_idx, name)
                W_fp = module.weight.data.detach().clone().float()
                print(f"Quantizing {key} with upstream GPTVQ-1D ...")
                gptq[name].fasterquant(
                    blocksize=args.gptq_blocksize,
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=False,
                    static_groups=False,
                    include_m_step=args.include_m_step,
                    use_vq=True,
                    svd_rank=None,
                    hessian_weighted_lookups=args.hessian_weighted_lookups,
                    only_init_kmeans=False,
                )
                quantized_layers += 1
                if gptvq_state is not None:
                    gptvq_state[key] = module.weight.data.detach().cpu().clone()

                if use_rbvt:
                    if key not in stat_sum:
                        raise RuntimeError(f"Missing RBVT activation stats for {key}")
                    W_gptvq = module.weight.data.detach().float()
                    qres = _gptvq_quant_result(
                        W_dequant=W_gptvq,
                        assignments=gptq[name].assignments,
                        centroids=gptq[name].quantizer.all_centroids,
                        bits=args.wbits,
                        block_size=args.groupsize,
                    )
                    count = max(1, stat_count[key])
                    mu = stat_sum[key].to(device) / count
                    ex2 = stat_sumsq[key].to(device) / count
                    sigma = (ex2 - mu * mu).clamp(min=0.0)
                    W_rbvt, stats = apply_rbvt(
                        W_fp=W_fp.to(device),
                        qres=qres,
                        mu=mu,
                        sigma_ii=sigma if args.rbvt_lambda > 0.0 else None,
                        rbvt_lambda=args.rbvt_lambda,
                        rbvt_topk=args.rbvt_topk if args.rbvt_topk > 0 else None,
                        row_chunk=args.row_chunk,
                        gap_floor=args.gap_floor,
                        strict_descent=args.strict_descent,
                    )
                    module.weight.data = W_rbvt.to(module.weight.data.dtype)
                    for total_key in totals:
                        totals[total_key] += getattr(stats, total_key)
                    del qres, W_rbvt, sigma, mu

                gptq[name].free()
                del W_fp
                torch.cuda.empty_cache()

        for sample_idx in range(actual_n_calib):
            outs[sample_idx] = _layer_call(layer, inps[sample_idx].unsqueeze(0), cache)

        layers[layer_idx] = layer.cpu()
        del layer, gptq
        torch.cuda.empty_cache()
        gc.collect()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    elapsed = time.time() - tick
    stats = {
        "method": "gptvq_rbvt" if use_rbvt else "gptvq",
        "bits": args.wbits,
        "vq_dim": 1,
        "num_linear_layers": quantized_layers,
        "groupsize": args.groupsize,
        "kmeans_iters": args.kmeans_iters,
        "kmeans_init_method": args.kmeans_init_method,
        "include_m_step": args.include_m_step,
        "hessian_weighted_lookups": args.hessian_weighted_lookups,
        "time_sec": elapsed,
    }
    if use_rbvt:
        stats.update(totals)
        stats["rbvt_lambda"] = args.rbvt_lambda
        stats["rbvt_topk"] = args.rbvt_topk
        print(
            "RBVT summary | "
            f"flips={totals['flips']} candidates={totals['candidates']} "
            f"bias={totals['bias_before']:.6e}->{totals['bias_after']:.6e}"
        )
    print(f"GPTVQ variant done in {elapsed:.2f}s")
    return stats


@torch.no_grad()
def _restore_linear_weights(model, state: dict[str, torch.Tensor]):
    modules = dict(model.named_modules())
    missing = []
    for name, weight in state.items():
        module = modules.get(name)
        if module is None or not hasattr(module, "weight"):
            missing.append(name)
            continue
        module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
    if missing:
        raise RuntimeError(f"Missing modules while restoring GPTVQ baseline: {missing[:5]}")


def evaluate_model(model_path: str, label: str, args, hf_token: str | None) -> tuple[dict, dict]:
    evaluator = RBVTSlidingWindowEvaluator(
        device=args.device,
        seed=args.seed,
        stride=args.eval_stride,
        max_length=args.eval_max_length,
        cache_dir=args.eval_cache_dir,
        hf_token=hf_token,
    )
    perplexity = {}
    for dataset_name, texts in {
        "WikiText-2": evaluator.load_wikitext2_test(args.eval_samples),
        "C4": evaluator.load_c4_validation(args.eval_samples),
    }.items():
        result = evaluator.evaluate_model_on_dataset(
            model_path=model_path,
            model_name=label,
            texts=texts,
            dataset_name=dataset_name,
        )
        if result is not None:
            perplexity[dataset_name] = result

    lm_eval = {}
    if args.include_lm_eval:
        runner = LMEvalHarnessRunner(
            tasks=args.lm_eval_tasks,
            device=args.device,
            batch_size=args.lm_eval_batch_size,
            num_fewshot=args.lm_eval_num_fewshot,
            limit=args.lm_eval_limit,
            output_dir=args.lm_eval_output_dir,
            run_name=label.lower(),
            hf_token=hf_token,
        )
        lm_eval = runner.run({label: model_path})
    return perplexity, lm_eval


def _cleanup_model_artifacts(output_dir: Path):
    keep = {"run_summary.json"}
    for child in output_dir.iterdir():
        if child.name in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _write_summary(output_dir: Path, summary: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_variant(variant: str, args, hf_token: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_rbvt = variant == "gptvq_rbvt"
    label = "GPTVQ_RBVT" if use_rbvt else "GPTVQ"
    output_dir = Path(args.output_root) / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}\nRunning {label}\n{'=' * 80}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    model.eval()

    calib_texts = load_calibration_data(
        dataset_name=args.calib_dataset,
        tokenizer=tokenizer,
        n_samples=args.n_calib,
        seqlen=args.max_length,
        seed=args.seed,
        cache_dir=args.calibration_cache_dir,
    )
    quant_stats = quantize_model_gptvq_1d(
        model=model,
        tokenizer=tokenizer,
        calib_texts=calib_texts,
        args=args,
        use_rbvt=use_rbvt,
    )

    print(f"Saving {label} model to {output_dir} ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    perplexity, lm_eval = evaluate_model(str(output_dir), label, args, hf_token=hf_token)
    summary = {
        "model_path": args.model_path,
        "variant": variant,
        "output_dir": str(output_dir),
        "quantization": quant_stats,
        "calibration": {
            "dataset": args.calib_dataset,
            "n_calib": args.n_calib,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "evaluation": {
            "perplexity": perplexity,
            "lm_eval": lm_eval,
            "lm_eval_tasks": args.lm_eval_tasks,
        },
        "args": vars(args),
    }
    _write_summary(output_dir, summary)
    if args.cleanup_model_artifacts:
        _cleanup_model_artifacts(output_dir)
        print(f"Cleaned model artifacts under {output_dir}; kept run_summary.json")
    return summary


def _make_summary(args, variant: str, output_dir: Path, quant_stats: dict, perplexity: dict, lm_eval: dict) -> dict:
    return {
        "model_path": args.model_path,
        "variant": variant,
        "output_dir": str(output_dir),
        "quantization": quant_stats,
        "calibration": {
            "dataset": args.calib_dataset,
            "n_calib": args.n_calib,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "evaluation": {
            "perplexity": perplexity,
            "lm_eval": lm_eval,
            "lm_eval_tasks": args.lm_eval_tasks,
        },
        "args": vars(args),
    }


def run_single_pass_compare(args, hf_token: str | None) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'=' * 80}\nRunning GPTVQ and GPTVQ_RBVT in one GPTVQ pass\n{'=' * 80}")
    output_root = Path(args.output_root)
    gptvq_dir = output_root / "gptvq"
    rbvt_dir = output_root / "gptvq_rbvt"
    gptvq_dir.mkdir(parents=True, exist_ok=True)
    rbvt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    model.eval()

    calib_texts = load_calibration_data(
        dataset_name=args.calib_dataset,
        tokenizer=tokenizer,
        n_samples=args.n_calib,
        seqlen=args.max_length,
        seed=args.seed,
        cache_dir=args.calibration_cache_dir,
    )

    gptvq_state: dict[str, torch.Tensor] = {}
    rbvt_stats = quantize_model_gptvq_1d(
        model=model,
        tokenizer=tokenizer,
        calib_texts=calib_texts,
        args=args,
        use_rbvt=True,
        gptvq_state=gptvq_state,
    )
    gptvq_stats = {
        key: value
        for key, value in rbvt_stats.items()
        if key
        not in {
            "flips",
            "candidates",
            "boundary_kept",
            "bias_before",
            "bias_after",
            "objective_before",
            "objective_after",
            "variance_increase",
            "rbvt_lambda",
            "rbvt_topk",
        }
    }
    gptvq_stats["method"] = "gptvq"
    gptvq_stats["shared_gptvq_pass"] = True
    rbvt_stats["shared_gptvq_pass"] = True

    print(f"Saving GPTVQ_RBVT model to {rbvt_dir} ...")
    model.save_pretrained(rbvt_dir)
    tokenizer.save_pretrained(rbvt_dir)

    print(f"Restoring GPTVQ baseline weights from the shared pass and saving to {gptvq_dir} ...")
    _restore_linear_weights(model, gptvq_state)
    model.save_pretrained(gptvq_dir)
    tokenizer.save_pretrained(gptvq_dir)
    del model, gptvq_state
    torch.cuda.empty_cache()
    gc.collect()

    summaries = []
    for variant, label, output_dir, quant_stats in (
        ("gptvq", "GPTVQ", gptvq_dir, gptvq_stats),
        ("gptvq_rbvt", "GPTVQ_RBVT", rbvt_dir, rbvt_stats),
    ):
        perplexity, lm_eval = evaluate_model(str(output_dir), label, args, hf_token=hf_token)
        summary = _make_summary(
            args=args,
            variant=variant,
            output_dir=output_dir,
            quant_stats=quant_stats,
            perplexity=perplexity,
            lm_eval=lm_eval,
        )
        _write_summary(output_dir, summary)
        summaries.append(summary)
        if args.cleanup_model_artifacts:
            _cleanup_model_artifacts(output_dir)
            print(f"Cleaned model artifacts under {output_dir}; kept run_summary.json")

    return summaries


def print_comparison(summaries: list[dict]):
    print("\n" + "=" * 80)
    print("GPTVQ 1D COMPARISON")
    print("=" * 80)
    for summary in summaries:
        variant = summary["variant"]
        ppl = summary.get("evaluation", {}).get("perplexity", {})
        lm_eval = summary.get("evaluation", {}).get("lm_eval", {})
        print(f"\n[{variant}]")
        for dataset_name in ("WikiText-2", "C4"):
            value = ppl.get(dataset_name, {}).get("perplexity")
            print(f"  ppl/{dataset_name}: {value:.4f}" if isinstance(value, float) else f"  ppl/{dataset_name}: MISSING")
        payload = next(iter(lm_eval.values()), {}) if isinstance(lm_eval, dict) and lm_eval else {}
        task_summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        for task in summary.get("evaluation", {}).get("lm_eval_tasks", []):
            metrics = task_summary.get(task, {})
            metric_value = None
            metric_name = None
            if isinstance(metrics, dict):
                for candidate in ("acc,none", "acc_norm,none", "exact_match,none", "exact_match"):
                    if isinstance(metrics.get(candidate), (int, float)):
                        metric_name = candidate
                        metric_value = float(metrics[candidate])
                        break
            if metric_value is None:
                print(f"  lm_eval/{task}: MISSING")
            else:
                print(f"  lm_eval/{task}/{metric_name}: {metric_value:.4f}")


def build_parser():
    parser = argparse.ArgumentParser(description="Compare upstream GPTVQ-1D with GPTVQ-1D + RBVT")
    parser.add_argument("--model-path", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="./outputs/gptvq_1d_rbvt_colab")
    parser.add_argument("--variants", nargs="+", default=["gptvq", "gptvq_rbvt"], choices=["gptvq", "gptvq_rbvt"])
    parser.add_argument(
        "--single-pass-compare",
        action="store_true",
        help="Run GPTVQ once, snapshot GPTVQ weights, apply RBVT, then save/evaluate both variants.",
    )
    parser.add_argument("--wbits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--gptq-blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--kmeans-init-method", choices=["cdf", "kpp", "mahalanobis"], default="mahalanobis")
    parser.add_argument("--assignment-chunk-size", type=int, default=4096)
    parser.add_argument("--kpp-n-subsample", type=int, default=10000)
    parser.add_argument("--include-m-step", action="store_true", default=True)
    parser.add_argument("--no-include-m-step", dest="include_m_step", action="store_false")
    parser.add_argument("--hessian-weighted-lookups", action="store_true", default=True)
    parser.add_argument("--no-hessian-weighted-lookups", dest="hessian_weighted_lookups", action="store_false")
    parser.add_argument("--true-sequential", action="store_true", default=True)
    parser.add_argument("--no-true-sequential", dest="true_sequential", action="store_false")
    parser.add_argument("--sym", action="store_true", default=False)
    parser.add_argument("--n-calib", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--calib-dataset", choices=["c4", "wikitext2"], default="wikitext2")
    parser.add_argument("--calibration-cache-dir", default="./calibration_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--row-chunk", type=int, default=1024)
    parser.add_argument("--rbvt-lambda", type=float, default=1.0)
    parser.add_argument("--rbvt-topk", type=int, default=0)
    parser.add_argument("--gap-floor", type=float, default=1e-8)
    parser.add_argument("--strict-descent", action="store_true", default=True)
    parser.add_argument("--allow-overshoot", dest="strict_descent", action="store_false")
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--eval-max-length", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--eval-cache-dir", default="./dataset_cache")
    parser.add_argument("--include-lm-eval", action="store_true", default=True)
    parser.add_argument("--no-lm-eval", dest="include_lm_eval", action="store_false")
    parser.add_argument("--lm-eval-tasks", nargs="+", default=["arc_easy", "arc_challenge"])
    parser.add_argument("--lm-eval-num-fewshot", type=int, default=None)
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--lm-eval-limit", type=float, default=100)
    parser.add_argument("--lm-eval-output-dir", default="./outputs/gptvq_1d_rbvt_colab/lm_eval")
    parser.add_argument("--cleanup-model-artifacts", action="store_true", default=True)
    parser.add_argument("--keep-model-artifacts", dest="cleanup_model_artifacts", action="store_false")
    return parser


def main():
    load_runtime_env()
    args = build_parser().parse_args()
    if args.groupsize <= 0:
        raise ValueError("--groupsize must be positive for GPTVQ-1D/RBVT index conversion.")
    if args.rbvt_lambda < 0:
        raise ValueError("--rbvt-lambda must be non-negative.")
    _set_seed(args.seed)
    hf_token = resolve_hf_token()
    print(
        f"Model={args.model_path} | device={args.device} | bits={args.wbits} | "
        f"variants={args.variants} | output={args.output_root}"
    )
    print(f"Model slug: {build_model_slug(args.model_path)}")

    if args.single_pass_compare:
        summaries = run_single_pass_compare(args, hf_token=hf_token)
    else:
        summaries = [run_variant(variant, args, hf_token=hf_token) for variant in args.variants]
    print_comparison(summaries)


if __name__ == "__main__":
    main()
