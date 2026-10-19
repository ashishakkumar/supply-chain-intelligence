from datetime import datetime, timedelta

from app.ml.features import FEATURE_COLUMNS, shipment_features
from app.ml.predict import predicted_late
from app.ml.train import feature_matrix, fit_pipeline, split_labeled


def _row(
    shipment_id: str,
    when: datetime,
    delay: float | None,
    origin: str = "CNSHA",
    containers: int = 10,
    weight: float | None = 100.0,
) -> dict[str, object]:
    return {
        "shipment_id": shipment_id,
        "origin_port": origin,
        "destination_port": "NLRTM",
        "cargo_type": "Electronics",
        "container_count": containers,
        "weight_tons": weight,
        "transit_days_planned": 20.0,
        "origin_congestion": 0.8,
        "destination_congestion": 0.7,
        "booking_date": when,
        "actual_delay_hours": delay,
        "actual_departure": when,
        "actual_arrival": when,
        "status": "DELIVERED",
    }


def test_feature_list_excludes_leakage_columns() -> None:
    forbidden = {"actual_departure", "actual_arrival", "status", "customer_id", "vessel_id"}
    assert forbidden.isdisjoint(FEATURE_COLUMNS)


def test_container_clip_and_null_weight() -> None:
    when = datetime(2024, 6, 1)
    assert shipment_features(_row("a", when, 0, containers=40))["container_count"] == 40
    assert shipment_features(_row("b", when, 0, containers=41))["container_count"] == 40
    assert shipment_features(_row("c", when, 0, containers=996))["container_count"] == 40
    assert shipment_features(_row("d", when, 0, weight=None))["weight_tons"] is None


def test_time_split_keeps_later_bookings_in_the_test_set() -> None:
    start = datetime(2024, 1, 1)
    rows = []
    for index in range(10):
        delay = None if index == 3 else (30.0 if index % 2 else 1.0)
        rows.append(_row(f"S{index}", start + timedelta(days=index), delay))
    train, test = split_labeled(rows)
    assert all(row["shipment_id"] != "S3" for row in train + test)
    assert train
    assert test
    assert max(row["booking_date"] for row in train) < min(row["booking_date"] for row in test)


def test_threshold_treats_one_half_as_late() -> None:
    assert predicted_late(0.5) is True
    assert predicted_late(0.49) is False


def test_cancelled_and_unknown_with_arrival_are_not_trained(monkeypatch) -> None:
    start = datetime(2024, 1, 1)
    rows = [
        _row(f"S{index}", start + timedelta(days=index), 30.0 if index % 2 else 1.0)
        for index in range(8)
    ]
    cancelled = _row("C", start + timedelta(days=2), 48.0)
    cancelled["status"] = "CANCELLED"
    unknown = _row("U", start + timedelta(days=4), 48.0)
    unknown["status"] = "UNKNOWN"
    rows.extend([cancelled, unknown])
    train, test = split_labeled(rows)
    assert {row["shipment_id"] for row in train + test}.isdisjoint({"C", "U"})
    seen: list[str] = []

    def spy(row: dict[str, object]) -> dict[str, object]:
        seen.append(str(row["shipment_id"]))
        return shipment_features(row)

    monkeypatch.setattr("app.ml.train.shipment_features", spy)
    fit_pipeline(rows)
    assert "C" not in seen
    assert "U" not in seen


def test_unseen_port_returns_a_score() -> None:
    start = datetime(2024, 1, 1)
    rows = []
    for index in range(12):
        origin = "CNSHA" if index < 6 else "NLRTM"
        delay = 30.0 if index % 2 else 1.0
        rows.append(_row(f"T{index}", start + timedelta(days=index), delay, origin=origin))
    pipeline = fit_pipeline(rows)
    probe = shipment_features(_row("new", start, 1.0, origin="ZZNEW"))
    score = float(pipeline.predict_proba(feature_matrix([probe]))[0, 1])
    assert 0 <= score <= 1
