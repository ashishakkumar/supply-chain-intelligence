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
    pass


class Port(Base):
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
    __tablename__ = "quarantine_shipments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    shipment_id: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    raw: Mapped[dict[str, object]] = mapped_column(JSONB)


class QuarantineEvent(Base):
    __tablename__ = "quarantine_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    raw: Mapped[dict[str, object]] = mapped_column(JSONB)


def SessionLocal(database_url: str) -> sessionmaker[object]:
    engine = create_engine(database_url)
    return sessionmaker(bind=engine)


def engine_for(database_url: str):
    return create_engine(database_url)
