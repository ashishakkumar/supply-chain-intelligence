from collections.abc import Mapping
from datetime import datetime

FEATURE_COLUMNS = [
    "origin_port",
    "destination_port",
    "cargo_type",
    "container_count",
    "weight_tons",
    "planned_transit_days",
    "origin_congestion",
    "destination_congestion",
    "booking_month",
]
CATEGORICAL = ["origin_port", "destination_port", "cargo_type"]
NUMERIC = [
    "container_count",
    "weight_tons",
    "planned_transit_days",
    "origin_congestion",
    "destination_congestion",
    "booking_month",
]
CONTAINER_CLIP = 40


def shipment_features(row: Mapping[str, object]) -> dict[str, object]:
    booking = row["booking_date"]
    if isinstance(booking, datetime):
        month = booking.month
    else:
        month = int(row["booking_month"])
    cargo = row["cargo_type"] or "UNKNOWN"
    weight = row["weight_tons"]
    return {
        "origin_port": row["origin_port"],
        "destination_port": row["destination_port"],
        "cargo_type": cargo,
        "container_count": min(int(row["container_count"]), CONTAINER_CLIP),
        "weight_tons": None if weight is None else float(weight),
        "planned_transit_days": float(row["transit_days_planned"]),
        "origin_congestion": float(row["origin_congestion"]),
        "destination_congestion": float(row["destination_congestion"]),
        "booking_month": month,
    }
