"""Unit tests for the assistant guardrails, tools, and LLM log.

The model is a fake. These tests do not call an external LLM.
"""

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from app.assistant.graph import answer, build_graph
from app.assistant.tools import build_data_tools, make_predict_tool, run_with_timeout
from app.domain import rank_routes


class AlwaysTools:
    """A fake model that requests another tool call on every turn, so the cap can be tested."""
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        """Accept the tool list and stay this fake model."""
        return self

    def invoke(self, messages):
        """Ask for another ping call, so the graph keeps going until the cap."""
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[{"name": "ping", "args": {}, "id": f"call-{self.calls}"}],
        )


class Scripted:
    """A fake model that returns one fixed sentence and never calls a tool."""
    def __init__(self, content: str) -> None:
        self.content = content

    def bind_tools(self, tools):
        """Accept the tool list and stay this fake model."""
        return self

    def invoke(self, messages):
        """Return the scripted sentence and do not call a tool."""
        return AIMessage(content=self.content)


def test_sixth_call_in_one_message_is_not_executed(tmp_path: Path) -> None:
    """Five tool calls run. The sixth in the same message is not executed."""
    calls = {"n": 0}

    @tool
    def ping() -> str:
        """A test tool."""
        calls["n"] += 1
        return "ok"

    class SixAtOnce:
        """Asks for six tool calls in a single model message."""
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, tools):
            """Accept the tool list and stay this fake model."""
            return self

        def invoke(self, messages):
            """On the first turn, request six tool calls at once."""
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
    """After five tool rounds the graph stops instead of looping."""
    calls = {"n": 0}

    @tool
    def ping() -> str:
        """A test tool."""
        calls["n"] += 1
        return "ok"

    answer("rank the routes", AlwaysTools(), [ping], tmp_path / "log.jsonl")
    assert calls["n"] == 5


def test_factual_answer_without_a_tool_is_replaced(tmp_path: Path) -> None:
    """A number or ranking with no tool result is replaced. The model must cite a tool."""
    text = answer(
        "What is the on-time rate?",
        Scripted("The on-time rate is 99%"),
        [],
        tmp_path / "log.jsonl",
    )
    assert text == "I can only answer that from the shipment tools."


def test_decorated_refusal_or_id_question_is_replaced(tmp_path: Path) -> None:
    """A refusal or an id question with extra words is not the allowed exception, so it is replaced."""
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
    """The exact off-domain refusal and the missing-id question are allowed without a tool call."""
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
    """predict_delay returns the HTTP body. It does not score the shipment itself."""
    class Response:
        """Minimal HTTP response. The predict tool reads .text."""
        text = '{"error":{"code":"shipment_not_found","message":"No shipment SHP-1"}}'

    def post(url: str, json: dict[str, str], timeout: float) -> Response:
        """Stand-in for httpx.post. Checks the predict-delay URL, body, and timeout."""
        assert url.endswith("/predict-delay")
        assert json == {"shipment_id": "SHP-1"}
        assert timeout == 5
        return Response()

    tool = make_predict_tool(post, "http://localhost:8000")
    assert "shipment_not_found" in tool.invoke({"shipment_id": "SHP-1"})


def test_log_records_the_exchange_and_not_the_api_key(tmp_path: Path, monkeypatch) -> None:
    """The log has the prompt, tools, tool results, and final text, and not the API key."""
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
    """A tool that exceeds the cap returns "Tool timed out." instead of hanging the turn."""
    def slow() -> str:
        """Sleep past the tool cap so the timeout path returns instead of this text."""
        time.sleep(1)
        return "done"

    started = time.perf_counter()
    assert run_with_timeout(slow, timeout=0.05) == "Tool timed out."
    assert time.perf_counter() - started < 0.3


def test_rank_routes_uses_the_window_and_known_delays() -> None:
    """Highest-delay ranking stays inside the quarter window and skips routes with no known delay."""
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


class _MemorySaver(MemorySaver):
    """In-memory checkpoint with the setup() method the Postgres saver has."""
    def setup(self) -> None:
        """Match PostgresSaver.setup. MemorySaver has nothing to create."""
        return None


@contextmanager
def _memory_checkpointer(_url: str):
    """Stand-in for PostgresSaver.from_conn_string so the thread tests need no database."""
    yield _MemorySaver()


def test_plain_invocation_does_not_carry_context(monkeypatch, tmp_path: Path) -> None:
    """A CLI call without --thread does not open a checkpoint. The next question starts clean."""
    seen: list[list[str]] = []

    class Remember:
        """Records each turn's user texts so the test can see that context was not kept."""
        def bind_tools(self, tools):
            """Accept the tool list and stay this fake model."""
            return self

        def invoke(self, messages):
            """Record the user texts and refuse, which needs no tool."""
            seen.append(
                [message.content for message in messages if isinstance(message, HumanMessage)]
            )
            return AIMessage(content="I can't help with that.")

    def fail_open(*_args, **_kwargs):
        """Fail the test if a plain invocation tries to open a checkpoint."""
        raise AssertionError("opened a checkpointer")

    monkeypatch.setattr("app.assistant.__main__.build_chat_model", lambda: Remember())
    monkeypatch.setattr("app.assistant.__main__.build_tools", list)
    monkeypatch.setenv("LLM_LOG_PATH", str(tmp_path / "log.jsonl"))
    monkeypatch.setattr("app.assistant.graph.PostgresSaver.from_conn_string", fail_open)
    from app.assistant.__main__ import main

    assert main(["hi"]) == 0
    assert main(["what did I just ask?"]) == 0
    assert seen == [["hi"], ["what did I just ask?"]]


def test_debug_flag_prints_prompt_tool_call_and_response(monkeypatch, capsys, tmp_path: Path) -> None:
    """--debug prints the system prompt, the tool call, and the tool response."""
    @tool
    def ping(q: str) -> str:
        """A test tool."""
        return "pong"

    class Once:
        """Calls one tool, then answers with the off-domain refusal."""
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, tools):
            """Accept the tool list and stay this fake model."""
            return self

        def invoke(self, messages):
            """Call ping once, then stop with the off-domain refusal."""
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "ping", "args": {"q": "x"}, "id": "c1"}],
                )
            return AIMessage(content="I can't help with that.")

    monkeypatch.setattr("app.assistant.__main__.build_chat_model", lambda: Once())
    monkeypatch.setattr("app.assistant.__main__.build_tools", lambda: [ping])
    monkeypatch.setenv("LLM_LOG_PATH", str(tmp_path / "log.jsonl"))
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(
        "app.assistant.graph.PostgresSaver.from_conn_string", _memory_checkpointer
    )
    from app.assistant.__main__ import main

    assert main(["--debug", "--thread", "t1", "What is the capital of France?"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "I can't help with that."
    assert "You answer questions about shipments" in captured.err
    assert "What is the capital of France?" in captured.err
    assert '"q": "x"' in captured.err
    assert "pong" in captured.err


def test_graph_compiles_with_no_tools() -> None:
    """The graph still compiles when the model has nothing to call."""
    assert build_graph(Scripted("I can't help with that."), []) is not None


class _Session:
    """Stand-in for a database session. The bad-date test must not open a real one."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def test_bad_date_does_not_query(monkeypatch) -> None:
    """A timezone in a tool date is an error object. The database is not queried."""
    def fail():
        """Fail the test if a bad date still opens a database session."""
        raise AssertionError("opened a session")

    monkeypatch.setattr("app.assistant.tools.open_session", fail)
    tool = build_data_tools()[0]
    result = json.loads(tool.invoke({"date_from": "2024-08-13T09:00:00+05:30"}))
    assert "error" in result


def test_date_only_end_includes_that_day(monkeypatch) -> None:
    """A date-only end bound includes that calendar day, through 23:59:59."""
    captured: dict[str, datetime | None] = {}

    def fake_list(session, origin, destination, status, start, end, limit, offset):
        """Capture the parsed date bounds and return an empty page."""
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
    """query_shipments with group_by route passes origin, destination, and status into the ranking."""
    captured: dict[str, str | None] = {}

    def fake_top(session, start, end, origin=None, destination=None, status=None, limit=20):
        """Capture the ranking filters and return no routes."""
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
