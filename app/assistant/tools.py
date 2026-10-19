import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime

import httpx
from langchain_core.tools import tool

from app.domain import latest_quarter
from app.queries import (
    latest_booking,
    list_shipments,
    open_session,
    route_delay_stats,
    top_routes,
)

TOOL_TIMEOUT_SECONDS = 5


def run_with_timeout(function: Callable[[], str], timeout: float = TOOL_TIMEOUT_SECONDS) -> str:
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(function)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        return "Tool timed out."
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def make_predict_tool(post: Callable[..., object], base_url: str, timeout: float = TOOL_TIMEOUT_SECONDS):
    @tool
    def predict_delay(shipment_id: str) -> str:
        """Call the delay-model HTTP endpoint for one shipment id."""

        def send() -> str:
            response = post(
                f"{base_url.rstrip('/')}/predict-delay",
                json={"shipment_id": shipment_id},
                timeout=timeout,
            )
            return str(response.text)

        return run_with_timeout(send, timeout)

    return predict_delay


def build_data_tools() -> list[object]:
    @tool
    def query_shipments(
        origin: str | None = None,
        destination: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        group_by: str = "none",
        quarter: str | None = None,
    ) -> str:
        """Read curated shipments. Use group_by='route' and quarter='latest' to rank routes by average delay."""

        def send() -> str:
            try:
                start = _parse(date_from)
                end = _parse(date_to, end=True)
            except ValueError as exc:
                return json.dumps({"error": str(exc)})
            supplied = bool(date_from and date_from.strip()) or bool(date_to and date_to.strip())
            with open_session() as session:
                window = None
                if quarter == "latest" or (group_by == "route" and not supplied):
                    latest = latest_booking(session)
                    if latest is None:
                        return json.dumps({"routes": [], "window": None})
                    window = latest_quarter([latest], now=datetime.now())
                    start = window["start"]
                    end = window["end"]
                if group_by == "route":
                    if not isinstance(start, datetime) or not isinstance(end, datetime):
                        return json.dumps(
                            {"error": "A route ranking needs both date bounds, or quarter='latest'."}
                        )
                    try:
                        routes = top_routes(session, start, end, origin, destination, status)
                    except ValueError as exc:
                        return json.dumps({"error": str(exc)})
                    return json.dumps({"window": _window_payload(window), "routes": routes}, default=str)
                try:
                    rows, total = list_shipments(session, origin, destination, status, start, end, 50, 0)
                except ValueError as exc:
                    return json.dumps({"error": str(exc)})
                return json.dumps(
                    {
                        "total": total,
                        "items": [
                            {
                                "shipment_id": row.shipment_id,
                                "origin_port": row.origin_port,
                                "destination_port": row.destination_port,
                                "status": row.status,
                                "booking_date": row.booking_date.isoformat(sep=" "),
                                "actual_delay_hours": row.actual_delay_hours,
                                "on_time_flag": row.on_time_flag,
                            }
                            for row in rows
                        ],
                    },
                    default=str,
                )

        return run_with_timeout(send)

    @tool
    def get_route_stats(origin: str, destination: str) -> str:
        """Return average delay hours, on-time rate, and shipment count. Origin and destination can be a code or a port name."""

        def send() -> str:
            with open_session() as session:
                stats = route_delay_stats(session, origin, destination)
            if stats is None:
                return json.dumps(
                    {"error": f"Unknown port in route {origin} to {destination}"}
                )
            return json.dumps(stats, default=str)

        return run_with_timeout(send)

    return [query_shipments, get_route_stats]


def build_tools() -> list[object]:
    base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    return [*build_data_tools(), make_predict_tool(httpx.post, base_url)]


def _parse(value: str | None, *, end: bool = False) -> datetime | None:
    if value is None or not value.strip():
        return None
    text = value.strip()
    clock = text[10:]
    if text.endswith(("Z", "z")) or "+" in clock or "-" in clock:
        raise ValueError("booking dates have no timezone")
    text = text.replace("T", " ")
    parsed = None
    date_only = False
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        date_only = fmt == "%Y-%m-%d"
        break
    if parsed is None:
        raise ValueError("invalid booking date")
    if end and date_only:
        return parsed.replace(hour=23, minute=59, second=59)
    return parsed


def _window_payload(window: dict[str, object] | None) -> dict[str, object] | None:
    if window is None:
        return None
    return {
        "year": window["year"],
        "quarter": window["quarter"],
        "start": window["start"].isoformat(sep=" ") if isinstance(window["start"], datetime) else window["start"],
        "end": window["end"].isoformat(sep=" ") if isinstance(window["end"], datetime) else window["end"],
    }
