"""
Quick smoke test for lm-evaluation-harness integration.

This bypasses quantization and perplexity so we can quickly verify that lm-eval
imports and runs end-to-end in the current environment.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from lm_eval_runner import LMEvalHarnessRunner


def main():
    parser = argparse.ArgumentParser(description="RBVTQuant lm-eval smoke test")
    parser.add_argument("--model-path", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--task", type=str, default="piqa")
    parser.add_argument("--limit", type=float, default=5)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./outputs/lm_eval_smoke")
    args = parser.parse_args()

    runner = LMEvalHarnessRunner(
        tasks=[args.task],
        device=args.device,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        output_dir=args.output_dir,
        run_name=f"smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        hf_token=None,
    )
    results = runner.run({"SMOKE": args.model_path})
    print("\nLM-EVAL SMOKE TEST PASSED")
    print(results["SMOKE"]["summary"])


if __name__ == "__main__":
    main()
