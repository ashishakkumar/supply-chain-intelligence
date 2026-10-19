import os
import sys
from pathlib import Path

from app.assistant.graph import answer
from app.assistant.llm import build_chat_model
from app.assistant.tools import build_tools


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    question = " ".join(args).strip()
    if not question:
        print("Pass a question as the command argument.", file=sys.stderr)
        return 2
    log_path = Path(os.environ.get("LLM_LOG_PATH", "logs/llm_calls.jsonl"))
    print(answer(question, build_chat_model(), build_tools(), log_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
