"""Read queries over the curated tables.

The API and the assistant tools share these functions, so a shipment list and
a route average mean the same thing in both places. Writes happen in the seed,
not here. Each session sets a 5 second statement timeout.
"""

import os
from datetime import datetime

from sqlalchemy import func, select, text ##sqlalchemy is a library for the sql database, it is used to interact with the database
from sqlalchemy.orm import Session, sessionmaker ## orm is object relational mapping, it is used to map the database tables to python classes
## Session is a database session, it is used to interact with the database
## sessionmaker is a factory for creating Session objects

from app.db import Port, QuarantineShipment, SessionLocal, Shipment
# Port is a class that represents a port
# QuarantineShipment is a class that represents a quarantine shipment
# SessionLocal is a class that represents a session local
# Shipment is a class that represents a shipment
# why did we use these classes ? because we want to use the database tables as python classes
# so we can use the database tables as python classes
# and we can use the python classes to interact with the database
# and we can use the python classes to interact with the database
## why we created only these classes ? because we want to use the database tables as python classes
from app.domain import rank_routes, resolve_port, route_stats
## rank_routes is a function that ranks the routes based on the delay
## resolve_port is a function that resolves the port code to the port name
## route_stats is a function that calculates the statistics of the routes
# are these used as tools in the assistant ? no, they are used to interact with the database
# why do they exist ?who is calling them ? the assistant is calling them to get the data from the database

_SESSION: sessionmaker[Session] | None = None


def open_session() -> Session:
    """
    This function opens a session to the database.
    It is used to interact with the database.
    It is used to get the data from the database.
    It is used to insert the data into the database.
    It is used to update the data in the database.
    It is used to delete the data from the database.
    """
    global _SESSION
    if _SESSION is None:
        _SESSION = SessionLocal(os.environ["DATABASE_URL"])
    session = _SESSION()
    # ponytail: one 5s cap for every read session; the loader uses its own engine
    session.execute(text("SET LOCAL statement_timeout = '5s'"))
    return session


def fetch_shipment(session: Session, shipment_id: str) -> tuple[str, Shipment | QuarantineShipment | None]:
    """
    This function fetches a shipment from the database.
    It is used to get the data from the database.
    
    QuarantineShipment is a class that represents a quarantine shipment
    Shipment is a class that represents a shipment
    quarantine means that the shipment is not allowed to be shipped to the destination port
    why class for quarantine shipment ? because we want to separate the quarantine shipments from the regular shipments
    
    fetching steps :
    1. fetch the shipment from the database
    2. if the shipment is found, return the shipment
    3. if the shipment is not found, check if the shipment is quarantined
    4. if the shipment is quarantined, return the quarantine shipment
    5. if the shipment is not quarantined, return the missing shipment

    """
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

    """
    This function lists the shipments from the database.
    It is used to get the data from the database.

    fetching steps :
    1. check if the booking from date has a timezone
    2. check if the booking to date has a timezone
    3. create a select statement to select the shipments from the database
    4. create a count statement to count the shipments from the database
    5. create a filters list to filter the shipments from the database
    6. if the origin is not None, add the origin to the filters
    7. if the destination is not None, add the destination to the filters
    8. if the status is not None, add the status to the filters

    # why we check if the booking from and booking to date have a timezone ? because we want to ensure that the booking dates are in the correct timezone
    # why we raise an error if the booking from and booking to date have no timezone ? because we want to ensure that the booking dates are in the correct timezone
    # why we create a select statement to select the shipments from the database ? because we want to select the shipments from the database
    # why we create a count statement to count the shipments from the database ? because we want to count the shipments from the database
    # why we create a filters list to filter the shipments from the database ? because we want to filter the shipments from the database
    # why we add the origin to the filters ? because we want to filter the shipments by the origin
    # why we add the destination to the filters ? because we want to filter the shipments by the destination
    # why we add the status to the filters ? because we want to filter the shipments by the status"""
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
    """ internal function to get the ports from the database used by the assistant to resolve the port code to the port name"""
    return [
        {"port_code": port.port_code, "port_name": port.port_name}
        for port in session.scalars(select(Port)).all()
    ]


def route_delay_stats(session: Session, origin: str, destination: str) -> dict[str, object] | None:
    """ internal function to get the route delay statistics from the database used by the assistant 
    to get the route delay statistics for a given origin and destination"""
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
    """ internal function to get the top routes from the database used by the assistant 
    to get the top routes for a given start and end date, origin and destination and status"""
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
    """ internal function to get the latest booking date from the database used by the assistant 
    to get the latest booking date"""
    return session.scalar(select(func.max(Shipment.booking_date)))


def port_by_code(session: Session, port_code: str) -> Port | None:
    """ internal function to get the port by code from the database used by the assistant 
    to get the port by code"""
    return session.get(Port, port_code)
