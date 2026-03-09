from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.append(str(API_DIR))

from _predict_py import predict  # noqa: E402


DATASET_SUMMARY_PATH = API_DIR / "dataset_summary.json"
ARTIFACT_DIR = API_DIR / "artifacts"

_model_metadata: dict[str, Any] | None = None
_dataset_summary: dict[str, Any] | None = None


def ensure_model_ready() -> dict[str, Any]:
    global _model_metadata

    if _model_metadata is not None:
        return _model_metadata

    metadata_path = ARTIFACT_DIR / "notebook_model_metadata.json"
    model_path = ARTIFACT_DIR / "notebook_model.joblib"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Expected files under api/artifacts/."
        )

    _model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return _model_metadata


def analyze_text(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Input text is required.")

    ensure_model_ready()
    return predict(ARTIFACT_DIR, text)


def load_dataset_summary() -> dict[str, Any]:
    global _dataset_summary

    if _dataset_summary is not None:
        return _dataset_summary

    if not DATASET_SUMMARY_PATH.exists():
        raise FileNotFoundError("Dataset summary is missing. Expected file under api/dataset_summary.json.")

    _dataset_summary = json.loads(DATASET_SUMMARY_PATH.read_text(encoding="utf-8"))
    return _dataset_summary


def read_json_body(handler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", 0))
    if content_length <= 0:
        return {}

    raw = handler.rfile.read(content_length)
    if not raw:
        return {}

    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON payload.")


def send_json(handler, status_code: int, payload: dict[str, Any]) -> None:
    import json

    encoded = json.dumps(payload).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)