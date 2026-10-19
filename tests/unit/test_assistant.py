import json
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app.assistant.graph import answer, build_graph
from app.assistant.tools import build_data_tools, make_predict_tool, run_with_timeout
from app.domain import rank_routes


class AlwaysTools:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[{"name": "ping", "args": {}, "id": f"call-{self.calls}"}],
        )


class Scripted:
    def __init__(self, content: str) -> None:
        self.content = content

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content=self.content)


def test_sixth_call_in_one_message_is_not_executed(tmp_path: Path) -> None:
    calls = {"n": 0}

    @tool
    def ping() -> str:
        """A test tool."""
        calls["n"] += 1
        return "ok"

    class SixAtOnce:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "ping", "args": {}, "id": f"c{index}"} for index in range(6)],
                )
            return AIMessage(content="done")

    path = tmp_path / "log.jsonl"
    answer("rank the routes", SixAtOnce(), [ping], path)
    assert calls["n"] == 5
    assert "Not executed: tool cap." in path.read_text(encoding="utf-8")


def test_graph_stops_after_five_tool_calls(tmp_path: Path) -> None:
    calls = {"n": 0}

    @tool
    def ping() -> str:
        """A test tool."""
        calls["n"] += 1
        return "ok"

    answer("rank the routes", AlwaysTools(), [ping], tmp_path / "log.jsonl")
    assert calls["n"] == 5


def test_factual_answer_without_a_tool_is_replaced(tmp_path: Path) -> None:
    text = answer(
        "What is the on-time rate?",
        Scripted("The on-time rate is 99%"),
        [],
        tmp_path / "log.jsonl",
    )
    assert text == "I can only answer that from the shipment tools."


def test_decorated_refusal_or_id_question_is_replaced(tmp_path: Path) -> None:
    dressed = answer(
        "rate",
        Scripted("I can't help with that. The on-time rate is 99%."),
        [],
        tmp_path / "a.jsonl",
    )
    mixed = answer(
        "rate",
        Scripted("The on-time rate is 99% for shipment id SHP-1?"),
        [],
        tmp_path / "b.jsonl",
    )
    assert dressed == "I can only answer that from the shipment tools."
    assert mixed == "I can only answer that from the shipment tools."


def test_refusal_and_missing_id_are_kept(tmp_path: Path) -> None:
    refusal = answer("Write a poem", Scripted("I can't help with that."), [], tmp_path / "a.jsonl")
    question = answer(
        "score this",
        Scripted("Which shipment id should I use?"),
        [],
        tmp_path / "b.jsonl",
    )
    assert refusal == "I can't help with that."
    assert question == "Which shipment id should I use?"


def test_predict_tool_returns_the_endpoint_error() -> None:
    class Response:
        text = '{"error":{"code":"shipment_not_found","message":"No shipment SHP-1"}}'

    def post(url: str, json: dict[str, str], timeout: float) -> Response:
        assert url.endswith("/predict-delay")
        assert json == {"shipment_id": "SHP-1"}
        assert timeout == 5
        return Response()

    tool = make_predict_tool(post, "http://localhost:8000")
    assert "shipment_not_found" in tool.invoke({"shipment_id": "SHP-1"})


def test_log_records_the_exchange_and_not_the_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-key")
    path = tmp_path / "log.jsonl"
    answer("hello", Scripted("I can't help with that."), [], path)
    line = path.read_text(encoding="utf-8")
    payload = json.loads(line)
    assert payload["prompt"] == "hello"
    assert payload["final"] == "I can't help with that."
    assert "tools" in payload
    assert "super-secret-key" not in line


def test_tool_timeout() -> None:
    def slow() -> str:
        time.sleep(1)
        return "done"

    started = time.perf_counter()
    assert run_with_timeout(slow, timeout=0.05) == "Tool timed out."
    assert time.perf_counter() - started < 0.3


def test_rank_routes_uses_the_window_and_known_delays() -> None:
    rows = [
        {"origin": "A", "destination": "B", "booking_date": datetime(2025, 10, 2), "delay": 10.0},
        {"origin": "A", "destination": "B", "booking_date": datetime(2025, 10, 3), "delay": None},
        {"origin": "C", "destination": "D", "booking_date": datetime(2025, 10, 4), "delay": 40.0},
        {"origin": "C", "destination": "D", "booking_date": datetime(2025, 9, 1), "delay": 100.0},
    ]
    ranked = rank_routes(rows, datetime(2025, 10, 1), datetime(2026, 1, 1))
    assert ranked[0]["origin"] == "C"
    assert ranked[0]["average_delay_hours"] == 40
    assert ranked[1]["shipment_count"] == 2
    assert ranked[1]["average_delay_hours"] == 10


def test_graph_compiles_with_no_tools() -> None:
    assert build_graph(Scripted("I can't help with that."), []) is not None


class _Session:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def test_bad_date_does_not_query(monkeypatch) -> None:
    def fail():
        raise AssertionError("opened a session")

    monkeypatch.setattr("app.assistant.tools.open_session", fail)
    tool = build_data_tools()[0]
    result = json.loads(tool.invoke({"date_from": "2024-08-13T09:00:00+05:30"}))
    assert "error" in result


def test_date_only_end_includes_that_day(monkeypatch) -> None:
    captured: dict[str, datetime | None] = {}

    def fake_list(session, origin, destination, status, start, end, limit, offset):
        captured["end"] = end
        captured["start"] = start
        return [], 0

    monkeypatch.setattr("app.assistant.tools.open_session", _Session)
    monkeypatch.setattr("app.assistant.tools.list_shipments", fake_list)
    tool = build_data_tools()[0]
    tool.invoke({"date_from": "2024-08-13 09:00", "date_to": "2025-10-14"})
    assert captured["start"] == datetime(2024, 8, 13, 9, 0)
    assert captured["end"] == datetime(2025, 10, 14, 23, 59, 59)


def test_route_ranking_keeps_filters(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    def fake_top(session, start, end, origin=None, destination=None, status=None, limit=20):
        captured["origin"] = origin
        captured["destination"] = destination
        captured["status"] = status
        return []

    monkeypatch.setattr("app.assistant.tools.open_session", _Session)
    monkeypatch.setattr("app.assistant.tools.latest_booking", lambda session: datetime(2025, 10, 14))
    monkeypatch.setattr("app.assistant.tools.top_routes", fake_top)
    tool = build_data_tools()[0]
    tool.invoke({"origin": "CNSHA", "status": "DELIVERED", "group_by": "route", "quarter": "latest"})
    assert captured == {"origin": "CNSHA", "destination": None, "status": "DELIVERED"}
