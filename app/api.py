import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
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
    shipment_id: str


def shipment_not_available(shipment_id: str, kind: str, reason: str | None) -> HTTPException:
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
def get_shipment(shipment_id: str) -> dict[str, object]:
    with open_session() as session:
        kind, row = fetch_shipment(session, shipment_id)
        if kind != "found":
            reason = row.reason if isinstance(row, QuarantineShipment) else None
            raise shipment_not_available(shipment_id, kind, reason)
        assert isinstance(row, Shipment)
        return _shipment_payload(row)


@router.get("/shipments")
def get_shipments(
    origin: str | None = None,
    destination: str | None = None,
    status: StatusName | None = None,
    booking_from: Annotated[datetime | None, Query(alias="from")] = None,
    booking_to: Annotated[datetime | None, Query(alias="to")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, object]:
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
def get_route_stats(origin: str, destination: str) -> dict[str, object]:
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
def predict_delay(body: PredictRequest) -> dict[str, object]:
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
    directory = Path(os.environ.get("DATA_DIR", "data/raw"))
    return {
        "checks": build_report(
            _read_csv(directory / "shipments.csv"),
            _read_csv(directory / "ports.csv"),
            _read_csv(directory / "port_events.csv"),
        )
    }


def _shipment_payload(row: Shipment) -> dict[str, object]:
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
    if value is None:
        return None
    return value.isoformat(sep=" ")
