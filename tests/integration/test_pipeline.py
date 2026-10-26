"""Integration tests for the seed: idempotent load, derived columns, rollback."""

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select, text

from app.db import (
    Event,
    Port,
    QuarantineEvent,
    QuarantineShipment,
    SessionLocal,
    Shipment,
)
from app.pipeline import load, main

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

_TRUNCATE = """
TRUNCATE TABLE
    port_events,
    shipments,
    quarantine_events,
    quarantine_shipments,
    ports
RESTART IDENTITY CASCADE
"""


@pytest.fixture(scope="module")
def database_url() -> str:
    """Postgres URL from the environment. The suite fails if it is unset, on purpose."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is unset")
    return url


def _clear(database_url: str) -> None:
    """Empty the serving tables so a test can prove a first seed and a failed seed."""
    engine = create_engine(database_url)
    try:
        with engine.begin() as conn:
            if conn.execute(text("SELECT to_regclass('public.ports')")).scalar() is None:
                return
            conn.execute(text(_TRUNCATE))
    finally:
        engine.dispose()


def test_load_is_idempotent_and_keeps_derived_columns(database_url: str) -> None:
    """A second seed does not rewrite rows. Derived delay, on-time, and route key are stored. Critical duplicates are quarantined."""
    load(database_url, RAW)
    with SessionLocal(database_url)() as session:
        port = session.get(Port, "BEANR")
        assert port is not None
        port.port_name = "mutated-for-idempotency"
        session.commit()

    load(database_url, RAW)

    with SessionLocal(database_url)() as session:
        assert session.scalar(select(func.count()).select_from(Shipment)) == 4886
        assert session.scalar(select(func.count()).select_from(QuarantineShipment)) == 134
        assert session.scalar(select(func.count()).select_from(Event)) == 24794
        assert session.scalar(select(func.count()).select_from(QuarantineEvent)) == 206
        assert session.scalar(select(func.count()).select_from(Port)) == 25

        row = session.get(Shipment, "SHP-00623")
        assert row is not None
        assert row.route_key == "NLRTM → PKKAR"
        assert row.actual_delay_hours == 12
        assert row.on_time_flag is True
        assert row.transit_days_planned is not None

        assert session.get(Shipment, "SHP-02149") is None
        quarantined = session.scalars(
            select(QuarantineShipment).where(QuarantineShipment.shipment_id == "SHP-02149")
        ).all()
        assert quarantined
        assert {item.reason for item in quarantined} == {"conflicting_shipment_id"}

        antwerp = session.get(Port, "BEANR")
        assert antwerp is not None
        assert antwerp.country is None
        assert antwerp.port_name == "mutated-for-idempotency"


def test_failed_first_load_leaves_tables_empty(database_url: str) -> None:
    """A crash before commit rolls the insert back. A failed first seed leaves the tables empty."""

    _clear(database_url)

    def fail(_session: object) -> None:
        """Force the seed to fail after the inserts and before the commit."""
        raise RuntimeError("loader failed")

    with pytest.raises(RuntimeError, match="loader failed"):
        load(database_url, RAW, before_commit=fail)

    with SessionLocal(database_url)() as session:
        assert session.scalar(select(func.count()).select_from(Port)) == 0
        assert session.scalar(select(func.count()).select_from(Shipment)) == 0
        assert session.scalar(select(func.count()).select_from(Event)) == 0
        session.execute(text("SELECT 1"))


def test_main_prints_seeded_then_already_seeded(
    database_url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`uv run db-seed` prints seeded once, then already seeded, and does not rewrite a port name."""
    _clear(database_url)
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_DIR", str(RAW))

    assert main() == 0
    assert capsys.readouterr().out.strip() == "seeded"

    assert main() == 0
    assert capsys.readouterr().out.strip() == "already seeded"

    with SessionLocal(database_url)() as session:
        assert session.scalar(select(func.count()).select_from(Shipment)) == 4886
        antwerp = session.get(Port, "BEANR")
        assert antwerp is not None
        assert antwerp.port_name != "mutated-for-idempotency"
