import os
from datetime import datetime

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.db import Port, QuarantineShipment, SessionLocal, Shipment
from app.domain import rank_routes, resolve_port, route_stats

_SESSION: sessionmaker[Session] | None = None


def open_session() -> Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = SessionLocal(os.environ["DATABASE_URL"])
    session = _SESSION()
    # ponytail: one 5s cap for every read session; the loader uses its own engine
    session.execute(text("SET LOCAL statement_timeout = '5s'"))
    return session


def fetch_shipment(session: Session, shipment_id: str) -> tuple[str, Shipment | QuarantineShipment | None]:
    row = session.get(Shipment, shipment_id)
    if row is not None:
        return "found", row
    quarantined = session.scalars(
        select(QuarantineShipment).where(QuarantineShipment.shipment_id == shipment_id)
    ).first()
    if quarantined is not None:
        return "quarantined", quarantined
    return "missing", None


def list_shipments(
    session: Session,
    origin: str | None,
    destination: str | None,
    status: str | None,
    booking_from: datetime | None,
    booking_to: datetime | None,
    limit: int,
    offset: int,
) -> tuple[list[Shipment], int]:
    if booking_from is not None and booking_from.tzinfo is not None:
        raise ValueError("from: booking dates have no timezone")
    if booking_to is not None and booking_to.tzinfo is not None:
        raise ValueError("to: booking dates have no timezone")
    statement = select(Shipment)
    count_statement = select(func.count()).select_from(Shipment)
    filters = []
    if origin is not None:
        filters.append(Shipment.origin_port == origin)
    if destination is not None:
        filters.append(Shipment.destination_port == destination)
    if status is not None:
        filters.append(Shipment.status == status)
    if booking_from is not None:
        filters.append(Shipment.booking_date >= booking_from)
    if booking_to is not None:
        filters.append(Shipment.booking_date <= booking_to)
    if filters:
        statement = statement.where(*filters)
        count_statement = count_statement.where(*filters)
    total = int(session.scalar(count_statement) or 0)
    rows = session.scalars(
        statement.order_by(Shipment.shipment_id).limit(limit).offset(offset)
    ).all()
    return list(rows), total


def _ports(session: Session) -> list[dict[str, str]]:
    return [
        {"port_code": port.port_code, "port_name": port.port_name}
        for port in session.scalars(select(Port)).all()
    ]


def route_delay_stats(session: Session, origin: str, destination: str) -> dict[str, object] | None:
    ports = _ports(session)
    origin_code = resolve_port(origin, ports)
    destination_code = resolve_port(destination, ports)
    if origin_code is None or destination_code is None:
        return None
    delays = list(
        session.scalars(
            select(Shipment.actual_delay_hours).where(
                Shipment.origin_port == origin_code,
                Shipment.destination_port == destination_code,
            )
        ).all()
    )
    stats = route_stats(delays)
    return {
        "origin": origin_code,
        "destination": destination_code,
        "average_delay_hours": stats["average_delay_hours"],
        "on_time_rate": stats["on_time_rate"],
        "shipment_count": stats["shipment_count"],
    }


def top_routes(
    session: Session,
    start: datetime,
    end: datetime,
    origin: str | None = None,
    destination: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[dict[str, object]]:
    ports = _ports(session)
    filters = [Shipment.booking_date >= start, Shipment.booking_date < end]
    for value, column in (
        (origin, Shipment.origin_port),
        (destination, Shipment.destination_port),
    ):
        if value is None:
            continue
        code = resolve_port(value, ports)
        if code is None:
            raise ValueError(f"Unknown port {value}")
        filters.append(column == code)
    if status is not None:
        filters.append(Shipment.status == status)
    rows = session.scalars(select(Shipment).where(*filters)).all()
    payload = [
        {
            "origin": row.origin_port,
            "destination": row.destination_port,
            "booking_date": row.booking_date,
            "delay": row.actual_delay_hours,
        }
        for row in rows
    ]
    return rank_routes(payload, start, end, limit)


def latest_booking(session: Session) -> datetime | None:
    return session.scalar(select(func.max(Shipment.booking_date)))


def port_by_code(session: Session, port_code: str) -> Port | None:
    return session.get(Port, port_code)
