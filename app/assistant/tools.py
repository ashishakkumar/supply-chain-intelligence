"""The three tools the assistant is allowed to call.

query_shipments(filters) reads the curated shipment table.
get_route_stats(origin, destination) returns average delay, on-time rate, and count.
predict_delay(shipment_id) calls POST /predict-delay. It is the only tool that
uses HTTP. The other two use the database directly, same as the API.

Each tool runs on a worker thread with a 5 second cap. A timeout is a tool
result ("Tool timed out."), not a crashed turn.
"""

import json
import os
from collections.abc import Callable # this is for the type hinting, it is used to hint the type of the function
from concurrent.futures import ThreadPoolExecutor # this is for the threading, it is used to run the function in a separate thread
from concurrent.futures import TimeoutError as FuturesTimeoutError # this is for the timeout error, it is used to handle the timeout error
from datetime import datetime

import httpx #async http client : being used to make the http requests for the tools
from langchain_core.tools import tool # this is for the tool decorator, it is used to decorate the function as a tool

from app.domain import latest_quarter # this is for the latest quarter, it is used to get the latest quarter
from app.queries import (
    latest_booking, # this is for the latest booking, it is used to get the latest booking date
    list_shipments, # this is for the list shipments, it is used to list the shipments from the database
    open_session, # this is for the open session, it is used to open a session to the database
    route_delay_stats, # this is for the route delay stats, it is used to get the route delay stats for a given origin and destination
    top_routes, # this is for the top routes, it is used to get the top routes for a given start and end date, origin and destination and status
)

TOOL_TIMEOUT_SECONDS = 5


def run_with_timeout(function: Callable[[], str], timeout: float = TOOL_TIMEOUT_SECONDS) -> str:
    """Run a tool body on one worker thread and return its text, or a timeout message.

    The pool is shut down without waiting, so a slow database call does not
    hold the assistant past the cap. The default cap is 5 seconds.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(function)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        return "Tool timed out."
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def make_predict_tool(post: Callable[..., object], base_url: str, timeout: float = TOOL_TIMEOUT_SECONDS):
    """Build the predict_delay tool against one API base URL.

    post is injected so tests can fake the HTTP call. The real CLI passes
    httpx.post and API_BASE_URL. The tool does not load the joblib file itself.
    """

    @tool
    def predict_delay(shipment_id: str) -> str:
        """Call the delay-model HTTP endpoint for one shipment id.

        Use this for "what is the delay risk for shipment SHP-...". The number
        comes from POST /predict-delay, not from the prompt.
        """

        def send() -> str:
            """POST the shipment id and return the response body as text."""
            response = post(
                f"{base_url.rstrip('/')}/predict-delay",
                json={"shipment_id": shipment_id},
                timeout=timeout,
            )
            return str(response.text)

        return run_with_timeout(send, timeout)

    return predict_delay


def build_data_tools() -> list[object]:
    """query_shipments and get_route_stats, both reading curated Postgres."""

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
        """Read curated shipments. Use group_by='route' and quarter='latest' to rank routes by average delay.

        Filters are origin, destination, status, and a booking date range.
        quarter='latest' is the latest booking quarter in the table, not today's
        calendar. Say the window this tool returns when you rank routes.
        """

        def send() -> str:
            """Query the curated tables and return JSON text."""
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
        """Return average delay hours, on-time rate, and shipment count. Origin and destination can be a code or a port name.

        Use this for one route, for example Shanghai to Rotterdam. Names are
        resolved to port codes. An unknown port is an error object, not a zero.
        """

        def send() -> str:
            """Look up one route and return the stats JSON."""
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
    """The three tools for a CLI turn: the two data tools plus predict_delay."""
    base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    return [*build_data_tools(), make_predict_tool(httpx.post, base_url)]


def _parse(value: str | None, *, end: bool = False) -> datetime | None:
    """Parse a booking bound from a tool argument into a naive datetime.

    Empty is None. A Z, a plus, or an offset in the time part is rejected,
    because booking dates have no timezone. A date with no clock is the start
    of that day, or 23:59:59 when it is the end bound.
    """
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
    """JSON-ready year, quarter, start, and end for the ranking window.

    The sample answer has to name this window. It is 2025 Q4 in the supplied file.
    """
    if window is None:
        return None
    return {
        "year": window["year"],
        "quarter": window["quarter"],
        "start": window["start"].isoformat(sep=" ") if isinstance(window["start"], datetime) else window["start"],
        "end": window["end"].isoformat(sep=" ") if isinstance(window["end"], datetime) else window["end"],
    }
