"""
lm-evaluation-harness integration for RBVTQuant.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class LMEvalHarnessRunner:
    def __init__(
        self,
        tasks: list[str],
        device: str = "cuda",
        batch_size: str = "auto",
        num_fewshot: int | None = None,
        output_dir: str = "./outputs/lm_eval",
        run_name: str | None = None,
        hf_token: str | None = None,
    ):
        self.tasks = tasks
        self.device = device
        self.batch_size = batch_size
        self.num_fewshot = num_fewshot
        self.output_dir = Path(output_dir)
        self.run_name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.hf_token = hf_token
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _model_args(self, model_path: str) -> str:
        dtype = "float16" if self.device.startswith("cuda") else "float32"
        model_args = f"pretrained={model_path},dtype={dtype},trust_remote_code=True"
        if self.hf_token:
            model_args += f",token={self.hf_token}"
        return model_args

    def _make_json_safe(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._make_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._make_json_safe(item) for item in value]

        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return self._make_json_safe(item_method())
            except (TypeError, ValueError):
                pass

        return repr(value)

    def _summarize_results(self, payload: dict) -> dict:
        results = payload.get("results", {})
        summary = {}
        for task_name, metrics in results.items():
            task_summary = {}
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    task_summary[metric_name] = value
            summary[task_name] = task_summary
        return summary

    def _write_raw_results(self, model_name: str, payload: dict):
        output_path = self.output_dir / f"{self.run_name}_{model_name}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self._make_json_safe(payload), handle, indent=2, sort_keys=True)

    def evaluate_model(self, model_name: str, model_path: str) -> dict:
        try:
            from lm_eval import evaluator
        except ImportError as exc:
            raise RuntimeError(
                "lm-eval is not installed. Install the 'lm-eval' package or disable lm-eval with --no-lm-eval."
            ) from exc

        payload = evaluator.simple_evaluate(
            model="hf",
            model_args=self._model_args(model_path),
            tasks=self.tasks,
            device=self.device,
            batch_size=self.batch_size,
            num_fewshot=self.num_fewshot,
            log_samples=False,
        )
        self._write_raw_results(model_name, payload)
        return {
            "tasks": list(self.tasks),
            "summary": self._summarize_results(payload),
            "raw": self._make_json_safe(payload),
        }

    def run(self, model_paths: dict[str, str]) -> dict:
        results = {}
        for model_name, model_path in model_paths.items():
            print(f"\nRunning lm-eval for {model_name} on {', '.join(self.tasks)}...")
            results[model_name] = self.evaluate_model(model_name, model_path)
        return results
