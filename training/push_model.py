"""Publish the most recent training run to the Hugging Face Hub."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
REPO_ID = os.getenv("HF_REPO_ID", "Kiernan1410/auto-ml-model")


def resolve_latest_run() -> Path:
    """Return the run directory written by the most recent train.py invocation."""
    if not MODELS_DIR.is_dir():
        raise SystemExit(
            f"No {MODELS_DIR} directory. Run `python training/train.py` first."
        )

    pointer = MODELS_DIR / "latest.txt"
    if pointer.is_file():
        candidate = MODELS_DIR / pointer.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return candidate
        print(f"{pointer} points at missing {candidate}; falling back to newest run.")

    # Timestamped names sort chronologically; ignore files and the checkpoint cache.
    runs = sorted(p for p in MODELS_DIR.glob("model_*") if p.is_dir())
    if not runs:
        raise SystemExit(
            f"No model_* directories in {MODELS_DIR}. Run `python training/train.py` first."
        )
    return runs[-1]


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Create a write token at "
            "https://huggingface.co/settings/tokens and expose it as the HF_TOKEN "
            "environment variable (or repository secret in CI)."
        )

    model_path = resolve_latest_run()

    metrics_file = model_path / "metrics.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.is_file() else {}
    summary = ", ".join(
        f"{key}={value:.4f}"
        for key, value in sorted(metrics.items())
        if isinstance(value, (int, float))
    )

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    commit_message = f"Automated training run {model_path.name}"
    if summary:
        commit_message += f" ({summary})"

    print(f"Uploading {model_path} to https://huggingface.co/{REPO_ID}")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=commit_message,
        commit_description=f"Pushed at {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC.",
        # Each publish should mirror exactly this run: drop anything already in
        # the repo that this upload does not replace (e.g. stale training
        # checkpoints from an earlier layout). Runs in one atomic commit.
        delete_patterns=["*"],
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
