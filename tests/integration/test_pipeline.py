import os
from pathlib import Path

import pytest
from sqlalchemy import func, select, text

from app.db import (
    Event,
    Port,
    QuarantineEvent,
    QuarantineShipment,
    SessionLocal,
    Shipment,
)
from app.pipeline import load

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is unset")
    return url


def test_load_is_idempotent_and_keeps_derived_columns(database_url: str) -> None:
    load(database_url, RAW)
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


def test_failed_load_leaves_the_previous_rows(database_url: str) -> None:
    load(database_url, RAW)
    with SessionLocal(database_url)() as session:
        before = session.scalar(select(func.count()).select_from(Shipment))

    def fail(_session: object) -> None:
        raise RuntimeError("loader failed")

    with pytest.raises(RuntimeError, match="loader failed"):
        load(database_url, RAW, before_commit=fail)

    with SessionLocal(database_url)() as session:
        assert session.scalar(select(func.count()).select_from(Shipment)) == before
        session.execute(text("SELECT 1"))
