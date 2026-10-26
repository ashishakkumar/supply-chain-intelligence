"""
why are we using sqlalchemy ? because we want to use the sql database to store the data
is this an orm ? yes, it is an orm because it is used to map the database tables to python classes
how do we tackle the drift between the database and the python classes ? we use the sqlalchemy to map the database tables to python classes
why aren't we using the .sql files to create the database tables ? because we want to use the sqlalchemy to map the database tables to python classes
how will migration be handled ? here ? 
like this :
1. create a new column in the database table 
2. create a new column in the python class
3. create a new migration file
4. run the migration file
5. repeat the process for all the changes

These classes map the tables created by db/migrations.
They do not create the tables. The data model is the table names,
key fields, and how the tables relate. That map is the classes below.

ports is the reference table. shipments.origin_port and shipments.destination_port
and port_events.port_code are foreign keys to ports.port_code.
quarantine_shipments and quarantine_events have no foreign key: an unknown port
is why many of those rows were kept out of the curated tables.
"""

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

STATUSES = ("DELIVERED", "DELAYED", "CANCELLED", "COMPLETED", "UNKNOWN")


class Base(DeclarativeBase):
    """Shared SQLAlchemy base. Tables are created from SQL files, not from this class."""


class Port(Base):
    """ports — reference data from ports.csv.

    Key: port_code. Congestion is constrained to [0, 1]. BEANR may have a null
    country; the check flags that and does not invent one.
    """
    __tablename__ = "ports"
    __table_args__ = (
        CheckConstraint(
            "avg_congestion_score >= 0 AND avg_congestion_score <= 1",
            name="congestion_between_0_and_1",
        ),
    )

    port_code: Mapped[str] = mapped_column(Text, primary_key=True)
    port_name: Mapped[str] = mapped_column(Text)
    country: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str] = mapped_column(Text)
    timezone: Mapped[str] = mapped_column(Text)
    avg_congestion_score: Mapped[float] = mapped_column(Numeric(3, 2))


class Shipment(Base):
    """shipments — curated container shipments.

    Key: shipment_id. origin_port and destination_port reference ports.
    Derived columns filled by the pipeline, not stored in the raw CSV:
    actual_delay_hours, on_time_flag, route_key, transit_days_planned,
    transit_days_actual. Timestamps have no time zone. weight_tons and
    container_count cannot be negative. status is one of the canonical labels.
    """
    __tablename__ = "shipments"
    __table_args__ = (
        CheckConstraint("weight_tons >= 0", name="weight_non_negative"),
        CheckConstraint("container_count >= 0", name="container_non_negative"),
        CheckConstraint(
            "status IN (" + ", ".join(f"'{status}'" for status in STATUSES) + ")",
            name="status_canonical",
        ),
        Index("ix_shipments_route", "origin_port", "destination_port"),
        Index("ix_shipments_booking", "booking_date"),
    )

    shipment_id: Mapped[str] = mapped_column(Text, primary_key=True)
    customer_id: Mapped[str] = mapped_column(Text)
    origin_port: Mapped[str] = mapped_column(Text, ForeignKey("ports.port_code"))
    destination_port: Mapped[str] = mapped_column(Text, ForeignKey("ports.port_code"))
    vessel_id: Mapped[str] = mapped_column(Text)
    booking_date: Mapped[object] = mapped_column(DateTime(timezone=False))
    planned_departure: Mapped[object] = mapped_column(DateTime(timezone=False))
    actual_departure: Mapped[object | None] = mapped_column(DateTime(timezone=False))
    planned_arrival: Mapped[object] = mapped_column(DateTime(timezone=False))
    actual_arrival: Mapped[object | None] = mapped_column(DateTime(timezone=False))
    container_count: Mapped[int] = mapped_column(Integer)
    cargo_type: Mapped[str | None] = mapped_column(Text)
    weight_tons: Mapped[float | None] = mapped_column(Numeric(10, 2))
    status: Mapped[str] = mapped_column(Text)
    actual_delay_hours: Mapped[float | None] = mapped_column(Float)
    on_time_flag: Mapped[bool | None] = mapped_column(Boolean)
    route_key: Mapped[str] = mapped_column(Text)
    transit_days_planned: Mapped[float | None] = mapped_column(Float)
    transit_days_actual: Mapped[float | None] = mapped_column(Float)


class Event(Base):
    """port_events — curated port event feed, treated as a stream.

    event_id is not unique: conflicting duplicates are quarantined, and identical
    duplicates would still share an id, so the primary key is a surrogate id.
    port_code references ports. delay_minutes cannot be negative.
    Blank event type and blank notes are stored as null.
    """
    __tablename__ = "port_events"
    __table_args__ = (
        CheckConstraint("delay_minutes >= 0", name="delay_minutes_non_negative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(Text)
    event_timestamp: Mapped[object] = mapped_column(DateTime(timezone=False))
    port_code: Mapped[str] = mapped_column(Text, ForeignKey("ports.port_code"))
    vessel_id: Mapped[str] = mapped_column(Text)
    event_type: Mapped[str | None] = mapped_column(Text)
    delay_minutes: Mapped[int] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)


class QuarantineShipment(Base):
    """quarantine_shipments — raw shipment rows the pipeline refused.

    Critical issues land here: conflicting duplicate shipment id, unknown port.
    raw is the original CSV row. No foreign key, because the port may not exist.
    GET /shipments/{id} returns 404 shipment_quarantined when the id is only here.
    """
    __tablename__ = "quarantine_shipments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    shipment_id: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    raw: Mapped[dict[str, object]] = mapped_column(JSONB)


class QuarantineEvent(Base):
    """quarantine_events — raw event rows the pipeline refused.

    Conflicting duplicate event ids are all stored, with no winner picked.
    raw is the original CSV row.
    """
    __tablename__ = "quarantine_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    raw: Mapped[dict[str, object]] = mapped_column(JSONB)


def SessionLocal(database_url: str) -> sessionmaker[object]:
    """Session factory for one DATABASE_URL.

    The API and the assistant tools share this Postgres. Each open_session()
    call checks out a short read session with a statement timeout.
    """
    engine = create_engine(database_url)
    return sessionmaker(bind=engine)


def engine_for(database_url: str):
    """Engine for the migrate and seed commands, which manage their own transactions."""
    return create_engine(database_url)
