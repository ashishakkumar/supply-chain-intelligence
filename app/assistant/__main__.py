"""CLI for the assistant. A UI is not required.

uv run assistant "Which routes have had the highest average delay this quarter?"
Pass the same --thread value to continue that thread only. --debug prints the
tool trace. A plain invocation does not keep chat context.
"""

import os
import sys
from pathlib import Path

from app.assistant.graph import answer
from app.assistant.llm import build_chat_model
from app.assistant.tools import build_tools


def _take_thread(args: list[str]) -> str | None:
    """Pull --thread <id> out of the argument list. The id is the checkpoint key.

    Missing, empty, or a following flag means there is no thread. The caller
    then tells the user to pass --thread and a question.
    """
    if "--thread" not in args:
        return None
    index = args.index("--thread")
    if index + 1 >= len(args):
        return None
    thread_id = args.pop(index + 1)
    args.pop(index)
    if not thread_id or thread_id.startswith("--"):
        return None
    return thread_id


def main(argv: list[str] | None = None) -> int:
    """Run one assistant question and print the answer.

    Steps:
    1. Read the question from the command arguments. --debug is a flag.
    2. --thread <id> continues that Postgres checkpoint only.
    3. Build the chat model from OPENAI_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL.
    4. Build the three tools and call answer, which logs the exchange.
    Exit 2 when the question or the thread id is missing.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    # ponytail: `--debug` is always the flag; a question that is only that token cannot be asked
    debug = "--debug" in args
    args = [arg for arg in args if arg != "--debug"]
    asked_for_thread = "--thread" in args
    thread_id = _take_thread(args)
    question = " ".join(args).strip()
    if asked_for_thread and thread_id is None:
        print("Pass --thread <id> and a question.", file=sys.stderr)
        return 2
    if not question:
        print("Pass a question as the command argument.", file=sys.stderr)
        return 2
    log_path = Path(os.environ.get("LLM_LOG_PATH", "logs/llm_calls.jsonl"))
    print(
        answer(
            question,
            build_chat_model(),
            build_tools(),
            log_path,
            debug=debug,
            thread_id=thread_id,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

    # to use this function, you can run the following command:
    # python -m app.assistant "What is the capital of France?"
