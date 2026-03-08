from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


MODEL_FILE_NAME = "notebook_model.joblib"
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text: object) -> str:
    if not isinstance(text, str):
        return ""

    normalized = text.lower()
    normalized = re.sub(r"<.*?>", " ", normalized)
    normalized = re.sub(r"[^a-zA-Z\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return ""

    words = [word for word in normalized.split() if word not in STOP_WORDS]
    return " ".join(words)


def tfidf_highlights(model, processed_text: str) -> list[dict[str, float | str]]:
    vectorizer = model.named_steps["tfidf"]
    transformed = vectorizer.transform([processed_text])
    if transformed.nnz == 0:
        return []

    feature_names = vectorizer.get_feature_names_out()
    weighted_terms = sorted(
        (
            {"token": str(feature_names[index]), "weight": round(float(value), 4)}
            for index, value in zip(transformed.indices, transformed.data)
        ),
        key=lambda entry: entry["weight"],
        reverse=True,
    )
    return weighted_terms[:6]


def score_prediction(model, classes: list[str], processed_text: str) -> tuple[float, float]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([processed_text])[0]
        by_label = {classes[index]: float(probabilities[index]) for index in range(len(classes))}
        positive = by_label.get("positive", 0.0)
        negative = by_label.get("negative", 0.0)
        score = positive - negative
        confidence = max(by_label.values()) if by_label else 0.0
        return round(float(score), 3), round(float(confidence), 3)

    if hasattr(model, "decision_function"):
        raw_value = float(np.ravel(model.decision_function([processed_text]))[0])
    else:
        classifier = model.named_steps["classifier"]
        vectorizer = model.named_steps["tfidf"]
        raw_value = float(np.ravel(classifier.decision_function(vectorizer.transform([processed_text])))[0])

    score = float(np.tanh(raw_value / 2))
    confidence = float(1 / (1 + np.exp(-abs(raw_value))))
    return round(score, 3), round(confidence, 3)


def infer_sentiment(label: str, score: float) -> str:
    if abs(score) < 0.12:
        return "neutral"
    return label


def predict(artifact_dir: Path, text: str) -> dict[str, object]:
    if not text.strip():
        raise ValueError("Input text is required.")

    artifact_path = artifact_dir / MODEL_FILE_NAME
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    payload = joblib.load(artifact_path)
    model = payload["model"]
    label_encoder = payload["label_encoder"]
    metadata = payload["metadata"]
    processed_text = preprocess_text(text)

    if not processed_text:
        return {
            "text": text.strip(),
            "sentiment": "neutral",
            "score": 0,
            "confidence": 0,
            "tokens": [],
            "explanation": "No meaningful tokens remained after notebook preprocessing.",
            "topContributors": [],
        }

    prediction_index = int(model.predict([processed_text])[0])
    classes = [str(label) for label in label_encoder.classes_]
    predicted_label = str(label_encoder.inverse_transform([prediction_index])[0])
    score, confidence = score_prediction(model, classes, processed_text)
    sentiment = infer_sentiment(predicted_label, score)
    tokens = processed_text.split()

    return {
        "text": text.strip(),
        "sentiment": sentiment,
        "score": score,
        "confidence": confidence,
        "tokens": tokens,
        "explanation": f"{metadata['modelName']} from LaptopML.ipynb predicted {sentiment} using {len(tokens)} processed tokens.",
        "topContributors": tfidf_highlights(model, processed_text),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    result = predict(Path(args.artifact_dir).resolve(), args.text)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)