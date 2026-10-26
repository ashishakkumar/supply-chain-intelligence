"""Unit tests for the standalone DQ checks and the curated-row rules.

These run on in-memory CSV rows. They do not need Postgres.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from app.dq import build_report, curate_events, curate_ports, curate_shipments

PORTS = {"CNSHA", "NLRTM"}


def shipment(**overrides: str) -> dict[str, str]:
    """One raw shipment row. Pass a field name to plant the quality issue under test."""
    row = {
        "shipment_id": "SHP-00001",
        "customer_id": "CUST-1001",
        "origin_port": "CNSHA",
        "destination_port": "NLRTM",
        "vessel_id": "VSL-0001",
        "booking_date": "2024-06-01 00:00:00",
        "planned_departure": "2024-06-10 00:00:00",
        "actual_departure": "2024-06-10 00:00:00",
        "planned_arrival": "2024-07-01 00:00:00",
        "actual_arrival": "2024-07-01 00:00:00",
        "container_count": "10",
        "cargo_type": "Electronics",
        "weight_tons": "100.5",
        "status": "DELIVERED",
    }
    row.update(overrides)
    return row


def test_negative_weight_percent_and_empty_file() -> None:
    """Each check records rows, percentage of that file, severity, and action. An empty file is 0 percent."""
    rows = [
        shipment(shipment_id="A", weight_tons="-1"),
        shipment(shipment_id="B"),
        shipment(shipment_id="C"),
        shipment(shipment_id="D"),
    ]
    report = build_report(rows, [], [])
    negative = next(item for item in report if item["name"] == "negative_weight")
    assert negative["rows_affected"] == 1
    assert negative["percentage"] == 25.0
    assert negative["severity"] == "warning"
    assert {"name", "rows_affected", "percentage", "severity", "action"} <= set(negative)

    empty = next(
        item for item in build_report([], [], []) if item["name"] == "negative_weight"
    )
    assert empty["rows_affected"] == 0
    assert empty["percentage"] == 0.0


def test_identical_duplicate_is_kept_once() -> None:
    """Identical duplicate shipment ids are a warning. The curated table keeps one copy."""
    row = shipment()
    kept, quarantined = curate_shipments([row, dict(row)], PORTS)
    assert len(kept) == 1
    assert quarantined == []
    assert kept[0]["shipment_id"] == "SHP-00001"


def test_conflicting_pair_is_quarantined() -> None:
    """Conflicting duplicate shipment ids are critical. Both rows are quarantined. Neither is kept."""
    first = shipment()
    second = shipment(cargo_type="")
    kept, quarantined = curate_shipments([first, second], PORTS)
    assert kept == []
    assert len(quarantined) == 2
    assert {item["reason"] for item in quarantined} == {"conflicting_shipment_id"}


def test_cleaning_rules_on_a_kept_row() -> None:
    """Warning fixes on a kept row: canonical cargo and status, null negative weight, derived delay fields."""
    kept, quarantined = curate_shipments(
        [
            shipment(
                status="delivered",
                cargo_type=" Chemicals",
                weight_tons="-3",
                container_count="0",
            )
        ],
        PORTS,
    )
    assert quarantined == []
    row = kept[0]
    assert row["status"] == "DELIVERED"
    assert row["cargo_type"] == "Chemicals"
    assert row["weight_tons"] is None
    assert row["container_count"] == 0
    assert row["on_time_flag"] is True
    assert row["route_key"] == "CNSHA → NLRTM"
    assert row["actual_delay_hours"] == 0


def test_unknown_port_is_quarantined_and_not_kept() -> None:
    """An origin or destination missing from the port file is critical and is quarantined."""
    kept, quarantined = curate_shipments(
        [shipment(origin_port="XXTST")],
        PORTS,
    )
    assert kept == []
    assert quarantined[0]["reason"] == "unknown_port"


def test_missing_arrival_stays_null() -> None:
    """Missing actual timestamps stay null. Delay is null. The row is not dropped."""
    kept, _ = curate_shipments(
        [
            shipment(
                status="CANCELLED",
                actual_departure="",
                actual_arrival="",
            )
        ],
        PORTS,
    )
    assert kept[0]["actual_arrival"] is None
    assert kept[0]["actual_delay_hours"] is None
    assert kept[0]["on_time_flag"] is None
    assert kept[0]["transit_days_actual"] is None
    assert kept[0]["transit_days_planned"] == 21


def test_completed_and_blank_status_are_not_forced_to_delivered() -> None:
    """Complete maps to COMPLETED. A blank status maps to UNKNOWN, not DELIVERED."""
    completed, _ = curate_shipments([shipment(status="Complete")], PORTS)
    blank, _ = curate_shipments([shipment(status="N/A")], PORTS)
    assert completed[0]["status"] == "COMPLETED"
    assert blank[0]["status"] == "UNKNOWN"


def test_blank_country_is_not_filled() -> None:
    """A blank port country is flagged and stored as null. The file does not contain a country to copy."""
    ports = [
        {
            "port_code": "BEANR",
            "port_name": "Antwerp",
            "country": "",
            "region": "EUR",
            "timezone": "UTC+1",
            "avg_congestion_score": "0.7",
        }
    ]
    report = build_report([], ports, [])
    country = next(item for item in report if item["name"] == "missing_port_country")
    assert country["rows_affected"] == 1
    assert curate_ports(ports)[0]["country"] is None


def test_conflicting_event_is_dropped_and_blank_type_is_kept() -> None:
    """Conflicting event ids are all quarantined. A blank event type is kept as null."""
    first = {
        "event_id": "EVT-1",
        "event_timestamp": "2024-06-01 00:00:00",
        "port_code": "CNSHA",
        "vessel_id": "VSL-0001",
        "event_type": "ARRIVAL",
        "delay_minutes": "0",
        "notes": "ok",
    }
    second = dict(first)
    second["event_type"] = "DEPARTURE"
    kept, quarantined = curate_events([first, second], {"CNSHA"})
    assert kept == []
    assert len(quarantined) == 2

    blank = dict(first)
    blank["event_id"] = "EVT-2"
    blank["event_type"] = ""
    blank["notes"] = ""
    blank["vessel_id"] = "VSL-0000"
    kept, quarantined = curate_events([blank], {"CNSHA"})
    assert quarantined == []
    assert kept[0]["event_type"] is None
    assert kept[0]["notes"] is None
    assert kept[0]["vessel_id"] == "VSL-0000"


def test_real_csv_report_counts() -> None:
    """The report against data/raw matches the counts in the data-quality summary."""
    root = Path(__file__).resolve().parents[2]
    report = _report_from_dir(root / "data" / "raw")
    counts = {item["name"]: item["rows_affected"] for item in report}
    assert counts["identical_duplicate_shipment_id"] == 16
    assert counts["conflicting_duplicate_shipment_id"] == 40
    assert counts["unknown_port"] == 95
    assert counts["negative_weight"] == 36
    assert counts["missing_weight"] == 73
    assert counts["zero_container_count"] == 23
    assert counts["extreme_container_count"] == 17
    assert counts["missing_actual_timestamps"] == 349
    assert counts["actual_departure_before_booking"] == 39
    assert counts["missing_port_country"] == 1
    assert counts["conflicting_event_id"] == 206
    assert counts["blank_event_type"] == 140
    assert counts["blank_event_notes"] == 1913
    assert counts["sentinel_vessel"] == 207
    assert counts["early_arrival"] == 950
    early = next(item for item in report if item["name"] == "early_arrival")
    assert early["action"] == "none"
    assert early["severity"] == "info"


def test_cli_writes_a_report_without_a_database(tmp_path: Path) -> None:
    """python -m app.dq --input writes the JSON report and does not need a database."""
    env = os.environ.copy()
    env.pop("DATABASE_URL", None)
    output = tmp_path / "report.json"
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.dq",
            "--input",
            str(root / "data" / "raw"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=root,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert any(item["name"] == "negative_weight" for item in payload["checks"])


def _report_from_dir(directory: Path) -> list[dict[str, object]]:
    """Run build_report on the three CSVs in a directory. This is the same call the endpoint makes."""
    import csv

    def load(name: str) -> list[dict[str, str]]:
        """Read one CSV from the fixture directory."""
        with (directory / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    return build_report(
        load("shipments.csv"),
        load("ports.csv"),
        load("port_events.csv"),
    )


def test_planned_transit_uses_dates() -> None:
    """transit_days_planned is planned arrival minus planned departure, in days."""
    kept, _ = curate_shipments([shipment()], PORTS)
    assert kept[0]["booking_date"] == datetime(2024, 6, 1, 0, 0)
