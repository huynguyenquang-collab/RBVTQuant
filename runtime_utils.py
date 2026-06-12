from __future__ import annotations

import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = ROOT / ".env"

DEFAULT_LM_EVAL_TASKS = {
    "smoke": [
        "piqa",
    ],
    "core": [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "piqa",
        "winogrande",
    ],
    "extended": [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "piqa",
        "winogrande",
        "boolq",
        "rte",
        "openbookqa",
        "lambada_openai",
    ],
}


def load_runtime_env(env_path: str | Path | None = None):
    env_file = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    if not env_file.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
        return
    except ImportError:
        pass

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def resolve_hf_token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )


def resolve_wandb_api_key() -> str | None:
    return os.getenv("WANDB_API_KEY")


def build_model_slug(model_ref: str) -> str:
    candidate = str(model_ref).rstrip("/\\").split("/")[-1].split("\\")[-1]

    def replace_numeric_dot(match: re.Match[str]) -> str:
        start = match.start()
        if start > 0 and candidate[start - 1].lower() == "v":
            return match.group(0)
        return match.group(0).replace(".", "p")

    candidate = re.sub(r"\d+\.\d+", replace_numeric_dot, candidate)
    candidate = candidate.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return candidate
