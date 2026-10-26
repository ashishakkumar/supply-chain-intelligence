"""Thread memory for the assistant, against the real Postgres checkpointer."""

import os
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.assistant.graph import answer


class Remember:
    """A fake model that records the user texts it was shown and then refuses."""
    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def bind_tools(self, tools):
        """Accept the tool list and stay this fake model."""
        return self

    def invoke(self, messages):
        """Record the user texts from this turn, including earlier turns on the same thread."""
        self.seen.append(
            [message.content for message in messages if isinstance(message, HumanMessage)]
        )
        return AIMessage(content="I can't help with that.")


def test_same_thread_remembers_the_previous_question(tmp_path: Path) -> None:
    """The same --thread id keeps the earlier question. A second call can see it."""
    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is unset")
    model = Remember()
    thread_id = f"test-{uuid.uuid4()}"
    answer("first question", model, [], tmp_path / "a.jsonl", thread_id=thread_id)
    answer("second question", model, [], tmp_path / "b.jsonl", thread_id=thread_id)
    assert model.seen[1] == ["first question", "second question"]


def test_other_thread_does_not_see_the_question(tmp_path: Path) -> None:
    """A different thread id does not see the other thread's question."""
    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is unset")
    model = Remember()
    answer("first question", model, [], tmp_path / "a.jsonl", thread_id=f"test-{uuid.uuid4()}")
    answer("second question", model, [], tmp_path / "b.jsonl", thread_id=f"test-{uuid.uuid4()}")
    assert model.seen[1] == ["second question"]
