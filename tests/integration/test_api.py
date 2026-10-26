"""HTTP tests against a real uvicorn process and Postgres.

Covers a successful shipment lookup, a 404, the DQ report, and predict-delay,
plus the filters, the JSON access log, and a missing model.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from app.db import SessionLocal, Shipment
from app.pipeline import load

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def _free_port() -> int:
    """An unused local port so two test runs do not fight over 8000."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _drain(pipe, bucket: list[str]) -> None:
    """Copy the server's stdout into a list so a test can read the access log."""
    for line in pipe:
        bucket.append(line)  # noqa: PERF402 — this streams a pipe, it does not copy a list


@pytest.fixture(scope="module")
def database_url() -> str:
    """Seed Postgres once for this module. Fails if DATABASE_URL is unset."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is unset")
    load(url, RAW)
    return url


def _start_server(database_url: str, model_path: str) -> tuple[str, subprocess.Popen[str], list[str]]:
    """Start uvicorn the way a reviewer would, and wait until GET /health answers."""
    port = _free_port()
    env = os.environ.copy()
    env.update(
        {
            "DATABASE_URL": database_url,
            "MODEL_PATH": model_path,
            "DATA_DIR": str(RAW),
            "APP_VERSION": "0.1.0",
            "PYTHONPATH": str(ROOT),
            "PYTHONUNBUFFERED": "1",
        }
    )
    stdout: list[str] = []
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    threading.Thread(target=_drain, args=(proc.stdout, stdout), daemon=True).start()
    threading.Thread(target=_drain, args=(proc.stderr, []), daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}")
        try:
            httpx.get(f"{url}/health", timeout=0.5)
            return url, proc, stdout
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server did not start")


@pytest.fixture(scope="module")
def server(database_url: str):
    """A running API with the committed delay model loaded at startup."""
    url, proc, stdout = _start_server(
        database_url,
        str(ROOT / "models" / "delay_model.joblib"),
    )
    yield url, stdout
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="module")
def server_without_model(database_url: str):
    """A running API whose MODEL_PATH does not exist, so predict must return 503."""
    url, proc, _stdout = _start_server(database_url, str(ROOT / "models" / "missing.joblib"))
    yield url
    proc.terminate()
    proc.wait(timeout=10)


def test_shipment_lookup_unknown_and_quarantine(server, database_url: str) -> None:
    """A real shipment returns 200. An unknown id and a quarantined id return 404 JSON."""
    url, _stdout = server
    found = httpx.get(f"{url}/shipments/SHP-00623", timeout=10)
    assert found.status_code == 200
    assert found.json()["route_key"] == "NLRTM → PKKAR"
    assert found.json()["on_time_flag"] is True

    missing = httpx.get(f"{url}/shipments/SHP-99999", timeout=10)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "shipment_not_found"

    quarantined = httpx.get(f"{url}/shipments/SHP-02149", timeout=10)
    assert quarantined.status_code == 404
    assert quarantined.json()["error"]["code"] == "shipment_quarantined"
    assert quarantined.json()["error"]["message"] != missing.json()["error"]["message"]


def test_route_stats_and_unknown_port(server) -> None:
    """Route stats return average delay, on-time rate, and count. An unknown port is 404."""
    url, _stdout = server
    stats = httpx.get(f"{url}/routes/Shanghai/Rotterdam/stats", timeout=10)
    assert stats.status_code == 200
    body = stats.json()
    assert body["origin"] == "CNSHA"
    assert body["destination"] == "NLRTM"
    assert body["shipment_count"] > 0
    assert 0 <= body["on_time_rate"] <= 1

    unknown = httpx.get(f"{url}/routes/ZZZZZ/NLRTM/stats", timeout=10)
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "unknown_port"


def test_quality_report(server) -> None:
    """GET /data-quality/report returns the DQ checks as JSON."""
    url, _stdout = server
    response = httpx.get(f"{url}/data-quality/report", timeout=30)
    assert response.status_code == 200
    names = {item["name"] for item in response.json()["checks"]}
    assert "negative_weight" in names
    assert "early_arrival" in names


def test_validation_errors_share_one_shape(server) -> None:
    """A bad query and a bad predict body are both 422 with code and message."""
    url, _stdout = server
    for params in (
        {"limit": 201},
        {"from": "not-a-date"},
        {"status": "delivered"},
    ):
        response = httpx.get(f"{url}/shipments", params=params, timeout=10)
        assert response.status_code == 422
        body = response.json()
        assert set(body) == {"error"}
        assert set(body["error"]) == {"code", "message"}


def test_booking_filter_is_inclusive_and_offset_past_the_end_keeps_total(server) -> None:
    """GET /shipments filters by origin, destination, status, and an inclusive booking range, and paginates."""
    url, _stdout = server
    booking = "2024-08-13 09:00:00"
    matched = httpx.get(
        f"{url}/shipments",
        params={
            "origin": "NLRTM",
            "destination": "PKKAR",
            "from": booking,
            "to": booking,
        },
        timeout=10,
    )
    assert matched.status_code == 200
    body = matched.json()
    assert any(item["shipment_id"] == "SHP-00623" for item in body["items"])

    later = httpx.get(
        f"{url}/shipments",
        params={
            "origin": "NLRTM",
            "destination": "PKKAR",
            "from": "2024-08-13 09:00:01",
            "to": "2024-08-13 09:00:01",
        },
        timeout=10,
    )
    assert later.json()["total"] == 0

    page = httpx.get(f"{url}/shipments", params={"limit": 1}, timeout=10)
    total = page.json()["total"]
    past = httpx.get(f"{url}/shipments", params={"offset": total, "limit": 1}, timeout=10)
    assert past.status_code == 200
    assert past.json()["items"] == []
    assert past.json()["total"] == total


def test_request_log_is_json(server) -> None:
    """The access line is JSON with method, path, status code, and latency."""
    url, stdout = server
    httpx.get(f"{url}/health", timeout=10)
    deadline = time.time() + 5
    while time.time() < deadline and not any('"method"' in line for line in stdout):
        time.sleep(0.1)
    line = next(item for item in stdout if '"method"' in item)
    payload = json.loads(line)
    assert {"method", "path", "status_code", "latency_ms"} <= set(payload)


def test_top_routes_honors_origin(database_url: str) -> None:
    """The assistant ranking query keeps the origin filter and uses the latest booking quarter."""
    from datetime import datetime

    from sqlalchemy import func

    from app.queries import top_routes

    start = datetime(2025, 10, 1)
    end = datetime(2026, 1, 1)
    with SessionLocal(database_url)() as session:
        known = session.scalar(
            select(func.count())
            .select_from(Shipment)
            .where(
                Shipment.origin_port == "CNSHA",
                Shipment.booking_date >= start,
                Shipment.booking_date < end,
                Shipment.actual_delay_hours.is_not(None),
            )
        )
        assert known and known > 0
        routes = top_routes(session, start, end, origin="Shanghai")
        assert routes
        assert {row["origin"] for row in routes} == {"CNSHA"}
        with pytest.raises(ValueError, match="Unknown port"):
            top_routes(session, start, end, origin="Not A Port")


def test_timezone_offset_is_rejected(server) -> None:
    """A booking filter with a timezone offset is 422. The files have no offset."""
    url, _stdout = server
    response = httpx.get(
        f"{url}/shipments",
        params={"from": "2024-08-13T09:00:00+05:30"},
        timeout=10,
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_missing_model_is_503(server_without_model) -> None:
    """predict-delay with no artefact is 503. The process is still up."""
    response = httpx.post(
        f"{server_without_model}/predict-delay",
        json={"shipment_id": "SHP-00623"},
        timeout=10,
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_cancelled_shipment_still_receives_a_score(server, database_url: str) -> None:
    """A cancelled shipment that survived quarantine can still be scored. The model was not trained on cancellations."""
    url, _stdout = server
    with SessionLocal(database_url)() as session:
        cancelled = session.scalars(
            select(Shipment).where(Shipment.status == "CANCELLED")
        ).first()
    assert cancelled is not None
    response = httpx.post(
        f"{url}/predict-delay",
        json={"shipment_id": cancelled.shipment_id},
        timeout=10,
    )
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["late_score"] <= 1
    assert isinstance(body["predicted_late"], bool)
    assert body["model_version"]
