from types import SimpleNamespace

from core.config import Settings
from pipelines.rag import RagPipeline


class FakeRetriever:
    def invoke(self, question: str):
        return [SimpleNamespace(page_content=f"source for {question}", metadata={"modality": "text"})]


class FakeGenerator:
    def generate(self, question: str, context: str, chat_history=None) -> str:
        assert "source for" in context
        return f"answered: {question}"


def test_rag_pipeline_with_fake_dependencies(tmp_path) -> None:
    settings = Settings()
    settings.conversation.enabled = False
    settings.rag.system_prompt = "Use context only."
    pipeline = RagPipeline(settings, FakeRetriever(), FakeGenerator())

    response = pipeline.ask("What is tested?")

    assert response.answer == "answered: What is tested?"
    assert len(response.sources) == 1
