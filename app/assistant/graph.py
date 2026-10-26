"""LangGraph loop for the assistant.

The model decides which tool to call. This module enforces the guardrails:
refuse off-domain questions (the system prompt), stop after 5 tool rounds,
and replace a factual answer that did not come from a tool. Every turn is
appended to the LLM log: prompt, tools, tool results, final text, token usage.
"""

import json
import os
import sys
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

MAX_TOOL_ROUNDS = 5
REFUSAL = "I can't help with that."
REPLACEMENT = "I can only answer that from the shipment tools."
SYSTEM_PROMPT = (
    "You answer questions about shipments, ports, routes, and delay risk. "
    "Use query_shipments with group_by='route' and quarter='latest' when the user asks which routes "
    "have the highest average delay this quarter, and say the window the tool returns. "
    "Use get_route_stats for one origin and destination. Names such as Shanghai are accepted. "
    "Use predict_delay to call the model endpoint for a shipment id. "
    f"If the question is not about shipments, ports, routes, or delay, reply exactly: {REFUSAL} "
    "If a shipment id is required and missing, ask: Which shipment id should I use? "
    "Do not state a number or a route ranking unless a tool returned it."
)


class AssistantState(TypedDict):
    """Graph state: the chat messages, and how many tool calls this question has used.

    tool_rounds enforces the cap of five tool calls per question.
    """

    messages: Annotated[list, add_messages]
    tool_rounds: int


def build_graph(model: object, tools: list[object], checkpointer: object | None = None):
    """Compile the agent/tools graph. An optional checkpointer keeps one thread.

    Guardrail: at most five tool rounds per question. The sixth attempt
    returns a stop message and does not call another tool. Without a checkpointer
    the next CLI invocation starts from an empty history.
    """

    def call_model(state: AssistantState) -> dict[str, object]:
        """Ask the model for the next message, or stop when the tool cap is hit."""
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            return {"messages": [AIMessage(content="Stopped after 5 tool calls.")]}
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": [response]}

    def route(state: AssistantState) -> str:
        """Go to the tools node when the model requested a call, otherwise end."""
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            return END
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    def run_tools(state: AssistantState) -> dict[str, object]:
        """Run the model's tool calls, up to the calls still left under the cap.

        A tool exception becomes the tool message text. The turn still finishes.
        Calls past the cap get "Not executed: tool cap." so the model sees that
        they did not run.
        """
        last = state["messages"][-1]
        remaining = MAX_TOOL_ROUNDS - state.get("tool_rounds", 0)
        calls = list(getattr(last, "tool_calls", None) or [])
        allowed, skipped = calls[:remaining], calls[remaining:]
        by_name = {getattr(item, "name", ""): item for item in tools}
        messages: list[ToolMessage] = []
        for call in allowed:
            name, args, call_id = _call_fields(call)
            tool = by_name.get(name)
            try:
                content = "Unknown tool." if tool is None else tool.invoke(args)
            except Exception as exc:  # noqa: BLE001 — a tool error is a result, not a failed turn
                content = str(exc)
            messages.append(ToolMessage(content=str(content), tool_call_id=call_id, name=name))
        for call in skipped:
            name, _args, call_id = _call_fields(call)
            messages.append(
                ToolMessage(content="Not executed: tool cap.", tool_call_id=call_id, name=name)
            )
        return {"messages": messages, "tool_rounds": state.get("tool_rounds", 0) + len(allowed)}

    graph = StateGraph(AssistantState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def answer(
    question: str,
    model: object,
    tools: list[object],
    log_path: Path,
    debug: bool = False,
    thread_id: str | None = None,
) -> str:
    """Answer one question and append the exchange to the LLM log.

    With no thread_id the turn is stateless. With a thread_id, PostgresSaver
    keeps that thread only, so a later question with the same id can see the
    earlier messages. The system prompt is sent on the first turn of a thread.
    debug prints the prompt, tool calls, and tool results to stderr.
    """
    if thread_id is None:
        state = build_graph(model, tools).invoke(_turn(question, include_system=True))
    else:
        url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
        with PostgresSaver.from_conn_string(url) as checkpointer:
            checkpointer.setup()
            graph = build_graph(model, tools, checkpointer)
            config = {"configurable": {"thread_id": thread_id}}
            include_system = not graph.get_state(config).values.get("messages")
            state = graph.invoke(_turn(question, include_system=include_system), config)
    final = finalize(state["messages"])
    _log_exchange(log_path, question, state["messages"], final)
    if debug:
        _debug_trace(state["messages"])
    return final


def _turn(question: str, *, include_system: bool) -> dict[str, object]:
    """The messages for one user turn. tool_rounds starts at 0.

    include_system is true for a new thread. A continued thread already has
    the system prompt in the checkpoint, so it is not sent again.
    """
    messages: list[object] = []
    if include_system:
        messages.append(SystemMessage(content=SYSTEM_PROMPT))
    messages.append(HumanMessage(content=question))
    return {"messages": messages, "tool_rounds": 0}


def finalize(messages: list[object]) -> str:
    """Last assistant text, after the cite-the-tool guardrail.

    If any tool ran, the model's final text is kept. A refusal and the
    "which shipment id" question are also kept, because those are not facts
    about a shipment. Any other answer with no tool result is replaced, so
    the model cannot state a delay or a ranking it did not look up.
    """
    used_tool = any(isinstance(message, ToolMessage) for message in messages)
    final = ""
    for message in messages:
        if isinstance(message, AIMessage) and not message.tool_calls and message.content:
            final = message.content if isinstance(message.content, str) else str(message.content)
    if used_tool or _allowed_without_tool(final):
        return final
    return REPLACEMENT


def _usage(messages: list[object]) -> dict[str, object] | None:
    """Token usage from the last model message, for the cost note in the log."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            usage = getattr(message, "usage_metadata", None)
            if usage:
                return dict(usage)
    return None


def _allowed_without_tool(text: str) -> bool:
    """True for the two replies that are allowed to contain no tool result.

    The off-domain refusal, and the question asking which shipment id to use.
    """
    stripped = text.strip()
    return stripped in {REFUSAL, "Which shipment id should I use?"}


def _call_fields(call: object) -> tuple[str, dict[str, object], str]:
    """Name, arguments, and id of one tool call, whether it is a dict or an object."""
    if isinstance(call, dict):
        args = call.get("args") or {}
        return str(call["name"]), dict(args), str(call["id"])
    args = getattr(call, "args", None) or {}
    return str(getattr(call, "name", "")), dict(args), str(getattr(call, "id", ""))


def _debug_trace(messages: list[object]) -> None:
    """Print the turn to stderr when the CLI is started with --debug."""
    for message in messages:
        if isinstance(message, SystemMessage):
            print("--- system ---", file=sys.stderr)
            print(message.content, file=sys.stderr)
        elif isinstance(message, HumanMessage):
            print("--- user ---", file=sys.stderr)
            print(message.content, file=sys.stderr)
        elif isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                name, args, _call_id = _call_fields(call)
                print(f"--- tool call {name} ---", file=sys.stderr)
                print(json.dumps(args), file=sys.stderr)
            if message.content and not message.tool_calls:
                print("--- assistant ---", file=sys.stderr)
                text = message.content if isinstance(message.content, str) else str(message.content)
                print(text, file=sys.stderr)
        elif isinstance(message, ToolMessage):
            print(f"--- tool response {message.name} ---", file=sys.stderr)
            print(message.content, file=sys.stderr)


def _log_exchange(path: Path, prompt: str, messages: list[object], final: str) -> None:
    """Append one JSON line: prompt, tools called, tool responses, final output, usage.

    Observability. Default path is logs/llm_calls.jsonl. The line is
    appended, so earlier calls stay. This is the record used to price a query.
    """
    tools: list[str] = []
    results: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            tools.extend(str(call["name"]) for call in message.tool_calls)
        if isinstance(message, ToolMessage):
            results.append({"name": str(message.name), "content": str(message.content)})
    record = {
        "prompt": prompt,
        "tools": tools,
        "tool_results": results,
        "final": final,
        "usage": _usage(messages),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
