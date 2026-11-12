import pytest

from screenplay_rag.langchain_ollama import LangChainOllamaProvider
from screenplay_rag.prompts import Prompt


class FakeResponse:
    content = "A LangChain grounded answer."


class FakeChatModel:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FakeResponse()


def test_langchain_provider_preserves_system_and_user_messages():
    chat_model = FakeChatModel()
    provider = LangChainOllamaProvider(
        host="http://ollama.test",
        model="granite4.2:3b",
        chat_model=chat_model,
    )

    answer = provider(Prompt(system="System rules", user="User question"))

    assert answer == "A LangChain grounded answer."
    assert [message.type for message in chat_model.messages] == ["system", "human"]
    assert [message.content for message in chat_model.messages] == [
        "System rules",
        "User question",
    ]
    assert provider.host == "http://ollama.test"
    assert provider.model == "granite4.2:3b"


def test_langchain_provider_rejects_empty_response():
    class EmptyChatModel:
        def invoke(self, messages):
            return type("Response", (), {"content": ""})()

    provider = LangChainOllamaProvider(chat_model=EmptyChatModel())

    with pytest.raises(RuntimeError, match="empty answer"):
        provider(Prompt(system="", user=""))