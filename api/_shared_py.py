from __future__ import annotations

import csv
from pathlib import Path
import sys
from threading import Lock
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.ml.predict import predict  # noqa: E402
from backend.ml.train_model import train_notebook_model  # noqa: E402


DATASET_PATH = ROOT_DIR / "laptops_dataset_final_600.csv"
ARTIFACT_DIR = Path("/tmp/notebook-model-artifacts")

_model_lock = Lock()
_model_metadata: dict[str, Any] | None = None
_dataset_summary: dict[str, Any] | None = None


def ensure_model_ready() -> dict[str, Any]:
    global _model_metadata

    if _model_metadata is not None:
        return _model_metadata

    with _model_lock:
        if _model_metadata is None:
            _model_metadata = train_notebook_model(DATASET_PATH, ARTIFACT_DIR)

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

    class_distribution = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
    }
    sample: list[dict[str, str]] = []
    total_rows = 0
    skipped_rows = 0

    with DATASET_PATH.open("r", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        for row in reader:
            total_rows += 1
            review = (row.get("review") or "").strip()
            if not review:
                skipped_rows += 1
                continue

            rating_raw = row.get("rating", "")
            try:
                rating = float(rating_raw)
            except ValueError:
                rating = 0

            sentiment = "positive" if rating > 3 else "negative"
            class_distribution[sentiment] += 1

            if len(sample) < 3:
                sample.append(
                    {
                        "product": row.get("product_name") or "",
                        "title": row.get("title") or "",
                        "sentiment": sentiment,
                    }
                )

    _dataset_summary = {
        "totalRows": total_rows,
        "skippedRows": skipped_rows,
        "classDistribution": class_distribution,
        "sample": sample,
    }
    return _dataset_summary


def read_json_body(handler) -> dict[str, Any]:
    import json

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