"""
Unified entrypoint for RBVTQuant research runs.

Features:
- one CLI for quantization + perplexity evaluation;
- supports plain nearest-codeword quantization (RTN) and RBVT refinement;
- keeps the same non-uniform quantizer backbone for both methods.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quantizers.base_quantizer as base_q
from calibration_utils import load_calibration_data
from eval_perplexity import RBVTSlidingWindowEvaluator
from quantizers import apply_rbvt, get_quantizer


def _hf_device_map(device: str):
    return {"": device} if device != "auto" else "auto"


def load_calibration_texts(tokenizer, dataset_name: str, n_samples: int, seqlen: int, seed: int) -> List[str]:
    return load_calibration_data(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        n_samples=n_samples,
        seqlen=seqlen,
        seed=seed,
    )


class ActStatsCollector:
    """Collect per-layer input mean and diagonal covariance statistics."""

    def __init__(self, want_var: bool = True):
        self.sum: Dict[str, torch.Tensor] = {}
        self.sumsq: Dict[str, torch.Tensor] = {}
        self.count: Dict[str, int] = {}
        self.want_var = want_var
        self.hooks = []

    def _hook(self, name: str):
        def hook(_m, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x.reshape(-1, x.shape[-1]).detach().float()
            s = x.sum(dim=0).cpu()
            n = x.shape[0]
            if name not in self.sum:
                self.sum[name] = s
                self.count[name] = n
                if self.want_var:
                    self.sumsq[name] = (x * x).sum(dim=0).cpu()
            else:
                self.sum[name] += s
                self.count[name] += n
                if self.want_var:
                    self.sumsq[name] += (x * x).sum(dim=0).cpu()

        return hook

    def register(self, layers: List[Tuple[str, nn.Module]]):
        for name, module in layers:
            self.hooks.append(module.register_forward_hook(self._hook(name)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def mean(self, name: str) -> torch.Tensor:
        return self.sum[name] / max(1, self.count[name])

    def var(self, name: str) -> torch.Tensor | None:
        if not self.want_var or name not in self.sumsq:
            return None
        m = self.mean(name)
        ex2 = self.sumsq[name] / max(1, self.count[name])
        return (ex2 - m * m).clamp(min=0.0)


def is_lmhead(name: str) -> bool:
    return "lm_head" in name.lower() or name.endswith("lm_head")


@torch.no_grad()
def collect_layer_stats(
    model,
    tokenizer,
    linears: List[Tuple[str, nn.Module]],
    calib_texts: List[str],
    device: str,
    n_calib: int,
    max_length: int,
    want_var: bool,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    collector = ActStatsCollector(want_var=want_var)
    print(f"Collecting activation statistics (want_var={want_var}) ...")
    collector.register(linears)
    for i, text in enumerate(calib_texts[:n_calib]):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model(**inputs, use_cache=False)
        if (i + 1) % 16 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    collector.remove()

    means: Dict[str, torch.Tensor] = {}
    variances: Dict[str, torch.Tensor] = {}
    for n, _ in linears:
        if n in collector.sum:
            means[n] = collector.mean(n)
            v = collector.var(n)
            if v is not None:
                variances[n] = v
    del collector
    gc.collect()
    return means, variances


@torch.no_grad()
def quantize_model(
    model,
    tokenizer,
    quantizer,
    calib_texts: List[str],
    device: str,
    method: str,
    skip_lmhead: bool = True,
    n_calib: int = 128,
    max_length: int = 512,
    row_chunk: int = 1024,
    rbvt_lambda: float = 1.0,
    gap_floor: float = 1e-8,
    strict_descent: bool = True,
):
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if skip_lmhead:
        linears = [(n, m) for (n, m) in linears if not is_lmhead(n)]
    print(
        f"Quantizing {len(linears)} Linear layers "
        f"({'skipping' if skip_lmhead else 'including'} lm_head) | method={method}"
    )

    means: Dict[str, torch.Tensor] = {}
    variances: Dict[str, torch.Tensor] = {}
    if method == "rbvt":
        means, variances = collect_layer_stats(
            model=model,
            tokenizer=tokenizer,
            linears=linears,
            calib_texts=calib_texts,
            device=device,
            n_calib=n_calib,
            max_length=max_length,
            want_var=rbvt_lambda > 0.0,
        )

    total_flips = 0
    total_candidates = 0
    total_boundary_kept = 0
    total_bias_before = 0.0
    total_bias_after = 0.0
    total_objective_before = 0.0
    total_objective_after = 0.0
    total_variance_increase = 0.0

    for n, module in tqdm(linears, desc="Quantizing layers"):
        W = module.weight.data
        qres = quantizer.quantize(W, row_chunk=row_chunk)
        W_out = qres.W_dequant

        if method == "rbvt" and n in means:
            mu = means[n].to(W.device)
            sigma_ii = variances.get(n)
            if sigma_ii is not None:
                sigma_ii = sigma_ii.to(W.device)
            W_out, stats = apply_rbvt(
                W_fp=W,
                qres=qres,
                mu=mu,
                sigma_ii=sigma_ii,
                rbvt_lambda=rbvt_lambda,
                row_chunk=row_chunk,
                gap_floor=gap_floor,
                strict_descent=strict_descent,
            )
            total_flips += stats.flips
            total_candidates += stats.candidates
            total_boundary_kept += stats.boundary_kept
            total_bias_before += stats.bias_before
            total_bias_after += stats.bias_after
            total_objective_before += stats.objective_before
            total_objective_after += stats.objective_after
            total_variance_increase += stats.variance_increase

        module.weight.data = W_out.to(W.dtype)

    if method == "rbvt":
        print(
            "RBVT summary | "
            f"flips={total_flips} | candidates={total_candidates} | "
            f"boundary_kept={total_boundary_kept}"
        )
        print(
            "RBVT objective | "
            f"bias_before={total_bias_before:.6e} -> bias_after={total_bias_after:.6e} | "
            f"objective_before={total_objective_before:.6e} -> objective_after={total_objective_after:.6e} | "
            f"variance_increase={total_variance_increase:.6e}"
        )
    else:
        print("RTN summary | plain nearest-codeword quantization completed.")

    return model


def evaluate_quantized_model(
    model_path: str,
    model_name: str,
    eval_device: str,
    eval_seed: int,
    eval_stride: int,
    eval_max_length: int,
    eval_cache_dir: str,
    eval_samples: int,
):
    evaluator = RBVTSlidingWindowEvaluator(
        device=eval_device,
        seed=eval_seed,
        stride=eval_stride,
        max_length=eval_max_length,
        cache_dir=eval_cache_dir,
    )

    datasets = {
        "WikiText-2": evaluator.load_wikitext2_test(eval_samples),
        "C4": evaluator.load_c4_validation(eval_samples),
    }

    results = {}
    for dataset_name, texts in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Evaluating model on {dataset_name} | name={model_name}")
        print(f"{'=' * 80}")
        dataset_result = evaluator.evaluate_model_on_dataset(
            model_path=model_path,
            model_name=model_name,
            texts=texts,
            dataset_name=dataset_name,
        )
        if dataset_result is None:
            print("  Evaluation failed (no results)")
            continue
        print(
            f"  Perplexity: {dataset_result['perplexity']:.4f} | "
            f"tokens={dataset_result['total_tokens']:,}"
        )
        results[dataset_name] = dataset_result

    print("\n" + "=" * 80)
    print("PERPLEXITY SUMMARY")
    print("=" * 80)
    for dataset_name, data in results.items():
        print(
            f"{dataset_name:<15} "
            f"name={model_name:<12} "
            f"ppl={data['perplexity']:.4f} "
            f"tokens={data['total_tokens']:,}"
        )
    return results


def build_parser():
    p = argparse.ArgumentParser(description="RBVTQuant main entrypoint: quantize + perplexity eval")
    p.add_argument("--model-path", type=str, required=True, help="HF model name or local path")
    p.add_argument("--device", type=str, default="cuda:0", help="Device for model loading/eval, e.g. cuda:0, cuda:1, cpu, or auto")
    p.add_argument("--method", type=str, default="rbvt", choices=["rtn", "rbvt"], help="Quantization method")
    p.add_argument("--quantizer", type=str, default="nf4", choices=["nf3", "nf4", "nvfp4", "codebook3", "codebook4"])
    p.add_argument("--output-dir", type=str, default="./quantized_model")

    p.add_argument("--skip-lmhead", dest="skip_lmhead", action="store_true", default=True)
    p.add_argument("--no-skip-lmhead", dest="skip_lmhead", action="store_false")
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--max-length", type=int, default=512, help="Calibration max token length")
    p.add_argument("--calib-dataset", type=str, default="c4", choices=["c4", "wikitext2"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--asym", dest="asym", action="store_true", default=True)
    p.add_argument("--no-asym", dest="asym", action="store_false")
    p.add_argument("--rbvt-lambda", type=float, default=1.0, help="Lambda in the RBVT surrogate objective")
    p.add_argument("--gap-floor", type=float, default=1e-8, help="Absolute floor on a feasible neighbouring gap")
    p.add_argument("--strict-descent", dest="strict_descent", action="store_true", default=True, help="Enforce sum r_i <= T in projection")
    p.add_argument("--allow-overshoot", dest="strict_descent", action="store_false", help="Use the looser sum r_i <= 2T projection bound")

    p.add_argument("--nf-block-size", type=int, default=64)
    p.add_argument("--nvfp4-block-size", type=int, default=16)
    p.add_argument("--cb-block-size", type=int, default=64)
    p.add_argument("--kmeans-iters", type=int, default=20)
    p.add_argument("--row-chunk", type=int, default=1024)

    p.add_argument("--eval-stride", type=int, default=512)
    p.add_argument("--eval-max-length", type=int, default=2048)
    p.add_argument("--eval-samples", type=int, default=2000, help="Number of documents/samples for stream datasets")
    p.add_argument("--eval-cache-dir", type=str, default="./dataset_cache")
    p.add_argument("--eval-float", dest="eval_float", action="store_true", default=False, help="Also compute perplexity for the original float model path before quantization")
    p.add_argument("--no-eval-float", dest="eval_float", action="store_false")
    return p


def main():
    args = build_parser().parse_args()
    if args.rbvt_lambda < 0.0:
        raise ValueError("--rbvt-lambda must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_q.ASYM = args.asym
    print(f"ASYM mode: {base_q.ASYM}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"--device={device} requested but CUDA is not available")
    print(
        f"Device: {device} | method={args.method} | quantizer={args.quantizer} | "
        f"skip_lmhead={args.skip_lmhead}"
    )

    if args.eval_float:
        evaluate_quantized_model(
            model_path=args.model_path,
            model_name="FLOAT",
            eval_device=device,
            eval_seed=args.seed,
            eval_stride=args.eval_stride,
            eval_max_length=args.eval_max_length,
            eval_cache_dir=args.eval_cache_dir,
            eval_samples=args.eval_samples,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=_hf_device_map(device),
        trust_remote_code=True,
    )
    model.eval()

    quantizer = get_quantizer(
        args.quantizer,
        nf_block_size=args.nf_block_size,
        nvfp4_block_size=args.nvfp4_block_size,
        cb_block_size=args.cb_block_size,
        n_iters=args.kmeans_iters,
        seed=args.seed,
    )
    print(f"Loaded quantizer: {quantizer}")

    calib_texts = load_calibration_texts(
        tokenizer=tokenizer,
        dataset_name=args.calib_dataset,
        n_samples=args.n_calib,
        seqlen=args.max_length,
        seed=args.seed,
    )
    quantize_model(
        model=model,
        tokenizer=tokenizer,
        quantizer=quantizer,
        calib_texts=calib_texts,
        device=device,
        method=args.method,
        skip_lmhead=args.skip_lmhead,
        n_calib=args.n_calib,
        max_length=args.max_length,
        row_chunk=args.row_chunk,
        rbvt_lambda=args.rbvt_lambda,
        gap_floor=args.gap_floor,
        strict_descent=args.strict_descent,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    evaluate_quantized_model(
        model_path=args.output_dir,
        model_name=args.method.upper(),
        eval_device=device,
        eval_seed=args.seed,
        eval_stride=args.eval_stride,
        eval_max_length=args.eval_max_length,
        eval_cache_dir=args.eval_cache_dir,
        eval_samples=args.eval_samples,
    )

    print("Done.")


if __name__ == "__main__":
    main()
