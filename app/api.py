"""HTTP routes for API service.

GET /health lives in app/main.py. The other five required endpoints are here:
one shipment, a filtered list, route stats, delay prediction, and the DQ report.
Errors are structured JSON with a code and a message, not a bare 500.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi import Path as ApiPath
from pydantic import BaseModel

from app.db import Port, QuarantineShipment, Shipment
from app.dq import build_report
from app.ml.features import shipment_features
from app.ml.predict import get_artifact, predicted_late
from app.ml.train import feature_matrix
from app.pipeline import _read_csv
from app.queries import (
    fetch_shipment,
    list_shipments,
    open_session,
    port_by_code,
    route_delay_stats,
)

router = APIRouter()
StatusName = Literal["DELIVERED", "DELAYED", "CANCELLED", "COMPLETED", "UNKNOWN"]


class PredictRequest(BaseModel):
    """Body of POST /predict-delay. The only field is the shipment id."""

    shipment_id: str


def shipment_not_available(shipment_id: str, kind: str, reason: str | None) -> HTTPException:
    """Turn a missing or quarantined id into a 404 with a structured JSON body.

    A quarantined row is not a curated
    shipment, so the client gets shipment_quarantined and the reason from the
    quarantine table. An id that is in neither table gets shipment_not_found.
    Both stay 404: the shipment is not available to read or to score.
    """
    if kind == "quarantined":
        return HTTPException(
            status_code=404,
            detail={
                "code": "shipment_quarantined",
                "message": f"Shipment {shipment_id} was quarantined: {reason}",
            },
        )
    return HTTPException(
        status_code=404,
        detail={
            "code": "shipment_not_found",
            "message": f"No shipment {shipment_id}",
        },
    )


@router.get("/shipments/{shipment_id}")
def get_shipment(
    shipment_id: Annotated[
        str,
        ApiPath(
            openapi_examples={"curated": {"summary": "A curated shipment", "value": "SHP-00623"}}
        ),
    ],
) -> dict[str, object]:
    """GET /shipments/{id} — full details for one curated shipment .

    Steps:
    1. Open a read session and look up the id in shipments, then quarantine.
    2. If the row is quarantined or missing, raise the 404 from shipment_not_available.
    3. If it is curated, return the shipment payload, including the derived fields
       (delay, on-time flag, route key, planned and actual transit days).
    """
    with open_session() as session:
        kind, row = fetch_shipment(session, shipment_id)
        if kind != "found":
            reason = row.reason if isinstance(row, QuarantineShipment) else None
            raise shipment_not_available(shipment_id, kind, reason)
        assert isinstance(row, Shipment)
        return _shipment_payload(row)


@router.get("/shipments")
def get_shipments(
    origin: Annotated[
        str | None,
        Query(openapi_examples={"code": {"summary": "Origin port code", "value": "NLRTM"}}),
    ] = None,
    destination: Annotated[
        str | None,
        Query(openapi_examples={"code": {"summary": "Destination port code", "value": "PKKAR"}}),
    ] = None,
    status: Annotated[
        StatusName | None,
        Query(
            openapi_examples={"delivered": {"summary": "Delivered shipments", "value": "DELIVERED"}}
        ),
    ] = None,
    booking_from: Annotated[
        datetime | None,
        Query(
            alias="from",
            openapi_examples={
                "booking": {
                    "summary": "Booking timestamp, no timezone",
                    "value": "2024-08-13 09:00:00",
                }
            },
        ),
    ] = None,
    booking_to: Annotated[
        datetime | None,
        Query(
            alias="to",
            openapi_examples={
                "booking": {
                    "summary": "Booking timestamp, no timezone",
                    "value": "2024-08-13 09:00:00",
                }
            },
        ),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, object]:
    """GET /shipments — list curated shipments.

    Filters required by the brief: origin, destination, status, and a booking
    date range (query names from and to). Pagination is limit and offset.
    A booking timestamp that carries a timezone is rejected with 422, because
    the files and the columns are timestamp without time zone.
    """
    try:
        with open_session() as session:
            rows, total = list_shipments(
                session,
                origin,
                destination,
                status,
                booking_from,
                booking_to,
                limit,
                offset,
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "validation_error", "message": str(exc)},
        ) from exc
    return {
        "items": [_shipment_payload(row) for row in rows],
        "limit": limit,
        "offset": offset,
        "total": total,
    }


@router.get("/routes/{origin}/{destination}/stats")
def get_route_stats(
    origin: Annotated[
        str,
        ApiPath(openapi_examples={"name": {"summary": "Port name or code", "value": "Shanghai"}}),
    ],
    destination: Annotated[
        str,
        ApiPath(openapi_examples={"name": {"summary": "Port name or code", "value": "Rotterdam"}}),
    ],
) -> dict[str, object]:
    """GET /routes/{origin}/{destination}/stats.

    Returns average delay hours, on-time rate, and shipment count for one route.
    Origin and destination may be a port code or a port name (Shanghai works).
    An unknown port is 404 unknown_port, not an empty stats object.
    The assistant tool get_route_stats reads the same query.
    """
    with open_session() as session:
        stats = route_delay_stats(session, origin, destination)
    if stats is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "unknown_port",
                "message": f"Unknown port in route {origin} to {destination}",
            },
        )
    return stats


@router.post("/predict-delay")
def predict_delay(
    body: Annotated[
        PredictRequest,
        Body(
            openapi_examples={
                "curated": {
                    "summary": "Score a curated shipment",
                    "value": {"shipment_id": "SHP-00623"},
                }
            }
        ),
    ],
) -> dict[str, object]:
    """POST /predict-delay — score one shipment with the saved model.

    The artefact is loaded at process startup, not trained on this request.
    A missing or sklearn-mismatched file is 503 model_unavailable; the process
    still serves the other routes. Features are booking-time only: no actual
    departure, no actual arrival, and no status (status matches the 24-hour label).
    late_score is the model's positive-class score. predicted_late is that score
    at the 0.5 threshold. The assistant calls this HTTP route; it does not score
    inside the prompt.
    """
    artifact = get_artifact()
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "Delay model artefact is not loaded.",
            },
        )
    with open_session() as session:
        kind, row = fetch_shipment(session, body.shipment_id)
        if kind != "found":
            reason = row.reason if isinstance(row, QuarantineShipment) else None
            raise shipment_not_available(body.shipment_id, kind, reason)
        assert isinstance(row, Shipment)
        origin = port_by_code(session, row.origin_port)
        destination = port_by_code(session, row.destination_port)
        assert isinstance(origin, Port)
        assert isinstance(destination, Port)
        features = shipment_features(
            {
                "origin_port": row.origin_port,
                "destination_port": row.destination_port,
                "cargo_type": row.cargo_type,
                "container_count": row.container_count,
                "weight_tons": None if row.weight_tons is None else float(row.weight_tons),
                "transit_days_planned": row.transit_days_planned,
                "origin_congestion": float(origin.avg_congestion_score),
                "destination_congestion": float(destination.avg_congestion_score),
                "booking_date": row.booking_date,
            }
        )
    pipeline = artifact["pipeline"]
    score = float(pipeline.predict_proba(feature_matrix([features]))[0, 1])
    return {
        "shipment_id": body.shipment_id,
        "late_score": score,
        "predicted_late": predicted_late(score),
        "model_version": artifact["model_version"],
    }


@router.get("/data-quality/report")
def data_quality_report() -> dict[str, object]:
    """GET /data-quality/report — the standalone DQ result as JSON.

    Reads the raw CSVs and runs build_report. It does not query Postgres and it
    does not load or change any table. The same function writes data/quality/report.json
    when you run python -m app.dq --input data/raw.
    """
    directory = Path(os.environ.get("DATA_DIR", "data/raw"))
    return {
        "checks": build_report(
            _read_csv(directory / "shipments.csv"),
            _read_csv(directory / "ports.csv"),
            _read_csv(directory / "port_events.csv"),
        )
    }


def _shipment_payload(row: Shipment) -> dict[str, object]:
    """Shape one curated shipment as the JSON body for the shipment endpoints.

    Includes the raw booking fields and the derived curated columns:
    actual_delay_hours, on_time_flag, route_key, transit_days_planned,
    transit_days_actual. weight_tons is a float so JSON does not emit a Decimal.
    """
    return {
        "shipment_id": row.shipment_id,
        "customer_id": row.customer_id,
        "origin_port": row.origin_port,
        "destination_port": row.destination_port,
        "vessel_id": row.vessel_id,
        "booking_date": _stamp(row.booking_date),
        "planned_departure": _stamp(row.planned_departure),
        "actual_departure": _stamp(row.actual_departure),
        "planned_arrival": _stamp(row.planned_arrival),
        "actual_arrival": _stamp(row.actual_arrival),
        "container_count": row.container_count,
        "cargo_type": row.cargo_type,
        "weight_tons": None if row.weight_tons is None else float(row.weight_tons),
        "status": row.status,
        "actual_delay_hours": row.actual_delay_hours,
        "on_time_flag": row.on_time_flag,
        "route_key": row.route_key,
        "transit_days_planned": row.transit_days_planned,
        "transit_days_actual": row.transit_days_actual,
    }


def _stamp(value: object) -> str | None:
    """Format a naive timestamp as 'YYYY-MM-DD HH:MM:SS', or null when it is missing.

    The CSVs and the Postgres columns have no offset, so the API does not add one.
    """
    if value is None:
        return None
    return value.isoformat(sep=" ")
