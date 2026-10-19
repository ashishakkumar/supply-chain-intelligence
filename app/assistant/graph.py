import json
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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
    messages: Annotated[list, add_messages]
    tool_rounds: int


def build_graph(model: object, tools: list[object]):
    def call_model(state: AssistantState) -> dict[str, object]:
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            return {"messages": [AIMessage(content="Stopped after 5 tool calls.")]}
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": [response]}

    def route(state: AssistantState) -> str:
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            return END
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    def run_tools(state: AssistantState) -> dict[str, object]:
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
    return graph.compile()


def answer(question: str, model: object, tools: list[object], log_path: Path) -> str:
    graph = build_graph(model, tools)
    state = graph.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
            "tool_rounds": 0,
        }
    )
    final = finalize(state["messages"])
    _log_exchange(log_path, question, state["messages"], final)
    return final


def finalize(messages: list[object]) -> str:
    used_tool = any(isinstance(message, ToolMessage) for message in messages)
    final = ""
    for message in messages:
        if isinstance(message, AIMessage) and not message.tool_calls and message.content:
            final = message.content if isinstance(message.content, str) else str(message.content)
    if used_tool or _allowed_without_tool(final):
        return final
    return REPLACEMENT


def _usage(messages: list[object]) -> dict[str, object] | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            usage = getattr(message, "usage_metadata", None)
            if usage:
                return dict(usage)
    return None


def _allowed_without_tool(text: str) -> bool:
    stripped = text.strip()
    return stripped in {REFUSAL, "Which shipment id should I use?"}


def _call_fields(call: object) -> tuple[str, dict[str, object], str]:
    if isinstance(call, dict):
        args = call.get("args") or {}
        return str(call["name"]), dict(args), str(call["id"])
    args = getattr(call, "args", None) or {}
    return str(getattr(call, "name", "")), dict(args), str(getattr(call, "id", ""))


def _log_exchange(path: Path, prompt: str, messages: list[object], final: str) -> None:
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
