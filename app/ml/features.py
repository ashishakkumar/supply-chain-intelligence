"""
this file is used to create the features for the machine learning model
featues and why theyre important and input and output:




- origin_port : the origin port of the shipment
- destination_port : the destination port of the shipment
- cargo_type : the type of the cargo
- container_count : the number of containers in the shipment
- weight_tons : the weight of the cargo in tons
- planned_transit_days : the planned transit days of the shipment
- origin_congestion : the congestion of the origin port
- destination_congestion : the congestion of the destination port
- booking_month : the month of the booking date
"""


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
""" this is the maximum number of containers that can be in a shipment """

def shipment_features(row: Mapping[str, object]) -> dict[str, object]:
    """Booking-time feature row for one shipment.

    Allowed, because each is known at booking: origin, destination, cargo,
    container count clipped at 40, weight, planned transit days, both ports'
    congestion scores, and booking month.

    Left out on purpose:
    actual_departure and actual_arrival would leak the outcome.
    status matches the 24-hour label on the canonical rows, so it would leak too.
    port events are not known at booking.

    A null cargo becomes the category UNKNOWN. A null weight stays null and the
    training pipeline imputes the median. The clip is only on the feature; the
    curated container_count column keeps the raw number.
    """
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
