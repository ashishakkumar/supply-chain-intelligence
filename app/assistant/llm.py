"""Chat model for the assistant.

Any OpenAI-compatible endpoint. The README records the choice as ChatOpenAI
pointed at OPENAI_BASE_URL. Cost per query is that endpoint's price for the
tokens written to logs/llm_calls.jsonl. There is no fixed vendor price here.
"""

import os

from langchain_openai import ChatOpenAI


def build_chat_model() -> ChatOpenAI:
    """The only place a chat model is constructed. Point OPENAI_BASE_URL at your endpoint.

    Reads OPENAI_MODEL, OPENAI_API_KEY, and OPENAI_BASE_URL. The API process
    does not call this. A timeout here leaves GET /health up, which is one of
    the failure modes in the production support runbook.
    """
    return ChatOpenAI(
        model=os.environ["OPENAI_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        
    )
