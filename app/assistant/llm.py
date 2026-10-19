import os

from langchain_openai import ChatOpenAI


def build_chat_model() -> ChatOpenAI:
    """The only place a chat model is constructed. Point OPENAI_BASE_URL at your endpoint."""
    return ChatOpenAI(
        model=os.environ["OPENAI_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
