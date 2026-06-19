from types import SimpleNamespace

from modules.document_processing import format_context


def test_format_context_for_modalities() -> None:
    docs = [
        SimpleNamespace(page_content="text summary", metadata={"modality": "text", "original": "full text"}),
        SimpleNamespace(page_content="table summary", metadata={"modality": "table"}),
        SimpleNamespace(page_content="image summary", metadata={"modality": "image"}),
    ]

    context = format_context(docs)

    assert "TEXT" in context
    assert "TABLE" in context
    assert "IMAGE" in context
