"""LangChain Ollama answer-provider adapter."""

import os

from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import Prompt


class LangChainOllamaProvider:
    """Use LangChain's ChatOllama while matching the local provider interface."""

    def __init__(self, host: str | None = None, model=None, chat_model=None):
        self.host = (
            host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "granite4.2:3b")
        if chat_model is None:
            from langchain_ollama import ChatOllama

            chat_model = ChatOllama(model=self.model, base_url=self.host)
        self.chat_model = chat_model

    def __call__(self, prompt: Prompt) -> str:
        response = self.chat_model.invoke(
            [
                SystemMessage(content=prompt.system),
                HumanMessage(content=prompt.user),
            ]
        )
        answer = getattr(response, "content", None)
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("LangChain Ollama returned an empty answer")
        return answer