from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


MODEL_FILE_NAME = "notebook_model.joblib"
METADATA_FILE_NAME = "notebook_model_metadata.json"


def ensure_nltk_resources() -> None:
    resources = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: object) -> str:
    if not isinstance(text, str):
        return ""

    normalized = text.lower()
    normalized = re.sub(r"<.*?>", " ", normalized)
    normalized = re.sub(r"[^a-zA-Z\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return ""

    filtered_words = [
        LEMMATIZER.lemmatize(word)
        for word in normalized.split()
        if word not in STOP_WORDS
    ]
    return " ".join(filtered_words)


def build_top_features(model: Pipeline) -> list[dict[str, float | str]]:
    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    weights: np.ndarray | None = None
    if hasattr(classifier, "feature_importances_"):
        weights = np.asarray(classifier.feature_importances_)
    elif getattr(classifier, "kernel", None) == "linear" and hasattr(classifier, "coef_"):
        coefficients = classifier.coef_
        dense_coefficients = coefficients.toarray() if hasattr(coefficients, "toarray") else np.asarray(coefficients)
        weights = np.abs(dense_coefficients).ravel()

    if weights is None or not len(weights):
        return []

    indices = np.argsort(weights)[::-1][:12]
    return [
        {"token": str(feature_names[index]), "weight": round(float(weights[index]), 4)}
        for index in indices
    ]


def train_notebook_model(dataset_path: Path, artifact_dir: Path) -> dict[str, object]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / MODEL_FILE_NAME
    metadata_path = artifact_dir / METADATA_FILE_NAME

    if model_path.exists() and metadata_path.exists() and metadata_path.stat().st_mtime >= dataset_path.stat().st_mtime:
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    data_frame = pd.read_csv(dataset_path)
    data_frame = data_frame.dropna(subset=["review"]).copy()
    data_frame["title"] = data_frame["title"].fillna("")
    data_frame["sentiment"] = data_frame["rating"].apply(lambda value: "positive" if float(value) > 3 else "negative")
    data_frame["processed_review"] = data_frame["review"].apply(preprocess_text)
    data_frame["processed_title"] = data_frame["title"].apply(preprocess_text)
    data_frame["processed_text"] = (
        data_frame["processed_title"].astype(str).str.strip() + " " + data_frame["processed_review"].astype(str).str.strip()
    ).str.strip()
    data_frame = data_frame[data_frame["processed_text"].str.len() > 0].copy()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data_frame["sentiment"])
    texts = data_frame["processed_text"]

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    svm_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("classifier", SVC(C=1.0, kernel="linear", gamma="scale", random_state=42)),
        ]
    )
    svm_pipeline.fit(x_train, y_train)
    svm_model = svm_pipeline
    svm_accuracy = accuracy_score(y_test, svm_model.predict(x_test))

    rf_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipeline.fit(x_train, y_train)
    rf_model = rf_pipeline
    rf_accuracy = accuracy_score(y_test, rf_model.predict(x_test))

    if svm_accuracy >= rf_accuracy:
        selected_name = "SVM"
        selected_model = svm_model
        selected_accuracy = svm_accuracy
    else:
        selected_name = "Random Forest"
        selected_model = rf_model
        selected_accuracy = rf_accuracy

    priors = data_frame["sentiment"].value_counts(normalize=True)
    metadata = {
        "vocabularySize": int(len(selected_model.named_steps["tfidf"].get_feature_names_out())),
        "trainedRows": int(len(data_frame)),
        "classPrior": {
            "positive": round(float(priors.get("positive", 0.0)), 4),
            "negative": round(float(priors.get("negative", 0.0)), 4),
        },
        "modelName": selected_name,
        "modelSource": "LaptopML.ipynb",
        "validationAccuracy": round(float(selected_accuracy), 4),
        "evaluatedModels": [
            {"name": "SVM", "accuracy": round(float(svm_accuracy), 4)},
            {"name": "Random Forest", "accuracy": round(float(rf_accuracy), 4)},
        ],
        "topFeatures": build_top_features(selected_model),
    }

    payload = {
        "model": selected_model,
        "label_encoder": label_encoder,
        "metadata": metadata,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(payload, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    metadata = train_notebook_model(dataset_path, artifact_dir)
    print(json.dumps(metadata))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)