"""Train the booking-time late-arrival model and write the artefact.

Label: 1 when actual_delay_hours is above 24, else 0. Rows with no arrival,
and cancelled or unknown status, are not training rows. The model does not
predict cancellation.

Holdout is the latest 20% of labeled bookings, cut on a booking-date boundary
so the same timestamp is not in both sides. Metrics are precision, recall, F1,
and AUC on that holdout, written to models/metrics.json next to the joblib file.

Run: python -m app.ml.train
The API never calls this. It only loads the file.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.dq import curate_ports, curate_shipments
from app.ml.features import CATEGORICAL, FEATURE_COLUMNS, NUMERIC, shipment_features
from app.ml.predict import predicted_late
from app.pipeline import _read_csv

MODEL_VERSION = "logreg-v1"


def trainable(row: dict[str, object]) -> bool:
    """True when the row has a real delay and is not CANCELLED or UNKNOWN.

    A missing arrival has a null delay, so it has no label. A cancelled shipment
    that survived quarantine can still be scored later; it is just not used to fit.
    """
    return row.get("actual_delay_hours") is not None and row.get("status") not in {
        "CANCELLED",
        "UNKNOWN",
    }


def split_labeled(
    rows: list[dict[str, object]],
    test_fraction: float = 0.2,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Time-ordered holdout. The test set is the latest slice of labeled bookings.

    Default test_fraction is 0.2, matching the reported metrics. The cut moves
    earlier until the last train booking_date differs from the first test one,
    so one timestamp is not on both sides. Returns (train, test).
    """
    labeled = [row for row in rows if trainable(row)]
    ordered = sorted(labeled, key=lambda row: row["booking_date"])
    if len(ordered) < 2:
        return ordered, []
    test_count = max(1, round(len(ordered) * test_fraction))
    cut = len(ordered) - test_count
    while cut > 1 and ordered[cut - 1]["booking_date"] == ordered[cut]["booking_date"]:
        cut -= 1
    if ordered[cut - 1]["booking_date"] == ordered[cut]["booking_date"]:
        cut = len(ordered) - 1
    return ordered[:cut], ordered[cut:]


def feature_matrix(features: list[dict[str, object]]) -> np.ndarray:
    """Stack feature dicts into a column-aligned array for the sklearn pipeline.

    Column order is FEATURE_COLUMNS. Missing numeric values are NaN so the
    median imputer can fill them. Categorical nulls are not used; cargo is
    already UNKNOWN by the time it gets here.
    """
    table: list[list[object]] = []
    for feature in features:
        row: list[object] = []
        for column in FEATURE_COLUMNS:
            value = feature[column]
            if column in NUMERIC and value is None:
                value = np.nan
            row.append(value)
        table.append(row)
    return np.array(table, dtype=object)


def build_pipeline() -> Pipeline:
    """One-hot categories, median-impute numbers, then balanced logistic regression.

    An honest model, not a high accuracy number. Logistic
    regression with class_weight balanced is the whole model. handle_unknown
    ignore means a new port code at score time becomes zeros instead of an error.
    That unseen code is a reason to retrain, not a reason to crash the request.
    """
    categorical_index = [FEATURE_COLUMNS.index(name) for name in CATEGORICAL]
    numeric_index = [FEATURE_COLUMNS.index(name) for name in NUMERIC]
    prepared = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_index,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric_index),
        ]
    )
    return Pipeline(
        steps=[
            ("prepare", prepared),
            (
                "model",
                LogisticRegression(class_weight="balanced", max_iter=1000),
            ),
        ]
    )


def fit_pipeline(rows: list[dict[str, object]]) -> Pipeline:
    """Fit on every trainable row in the list. Tests use this. main() splits first."""
    labeled = [row for row in rows if trainable(row)]
    pipeline = build_pipeline()
    pipeline.fit(feature_matrix([shipment_features(row) for row in labeled]), _labels(labeled))
    return pipeline


def training_rows(input_dir: Path) -> list[dict[str, object]]:
    """Curated shipments plus both ports' congestion, read from the raw CSVs.

    Uses the same curate functions as the seed, so quarantined rows never become
    labels. Congestion comes from the port file and is known at booking time.
    Quarantined shipments are discarded here; they are not a third return value
    the trainer uses.
    """
    ports = curate_ports(_read_csv(input_dir / "ports.csv"))
    congestion = {str(port["port_code"]): float(port["avg_congestion_score"]) for port in ports}
    shipments, _quarantined = curate_shipments(
        _read_csv(input_dir / "shipments.csv"),
        set(congestion),
    )
    rows: list[dict[str, object]] = []
    for shipment in shipments:
        rows.append(
            {
                **shipment,
                "origin_congestion": congestion[str(shipment["origin_port"])],
                "destination_congestion": congestion[str(shipment["destination_port"])],
            }
        )
    return rows


def evaluate(pipeline: Pipeline, rows: list[dict[str, object]]) -> dict[str, object]:
    """Precision, recall, F1, and AUC on a labeled holdout.

    Also records the positive base rate and the confusion counts, so a weak
    score can be read against how often shipments are actually late. Threshold
    for the class metrics is predicted_late (0.5). AUC uses the raw score.
    """
    features = [shipment_features(row) for row in rows]
    labels = np.array(_labels(rows))
    scores = pipeline.predict_proba(feature_matrix(features))[:, 1]
    predicted = np.array([predicted_late(float(score)) for score in scores])
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "auc": float(roc_auc_score(labels, scores)),
        "base_rate": float(labels.mean()),
        "confusion": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "sklearn_version": sklearn_version,
        "model_version": MODEL_VERSION,
    }


def main() -> int:
    """Train on data/raw, save models/delay_model.joblib and models/metrics.json.

    The joblib payload is the pipeline, model_version, and the sklearn version.
    predict.py refuses the file when that sklearn version does not match.
    """
    root = Path(__file__).resolve().parents[2]
    rows = training_rows(root / "data" / "raw")
    train, test = split_labeled(rows)
    pipeline = build_pipeline()
    pipeline.fit(feature_matrix([shipment_features(row) for row in train]), _labels(train))
    metrics = evaluate(pipeline, test)
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "model_version": MODEL_VERSION, "sklearn_version": sklearn_version},
        model_dir / "delay_model.joblib",
    )
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


def _labels(rows: list[dict[str, object]]) -> list[int]:
    """1 when actual delay is more than 24 hours, else 0."""
    return [1 if float(row["actual_delay_hours"]) > 24 else 0 for row in rows]


if __name__ == "__main__":
    raise SystemExit(main())
