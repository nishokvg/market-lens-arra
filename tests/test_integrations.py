from market_lens.integrations import (
    FactExtractionService,
    MemoryService,
    SourceKnowledgeBase,
    _chat_model_settings,
    _embedding_settings,
)


def test_memory_is_optional_without_an_api_key(monkeypatch):
    monkeypatch.delenv("MEM0_API_KEY", raising=False)

    result = MemoryService().recall("researcher-1", "apparel sourcing preferences")

    assert result.status == "disabled"
    assert result.items == []


def test_knowledge_base_is_optional_without_credentials(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = SourceKnowledgeBase().retrieve("custom apparel suppliers")

    assert result.status == "disabled"
    assert result.items == []


def test_memory_provider_initialization_failure_is_non_blocking(monkeypatch):
    monkeypatch.setattr(MemoryService, "_client", lambda _self: (_ for _ in ()).throw(RuntimeError("provider unavailable")))

    result = MemoryService().recall("researcher-1", "apparel sourcing preferences")

    assert result.status == "unavailable"


def test_llm_extraction_is_opt_in(monkeypatch):
    monkeypatch.delenv("USE_LLM_EXTRACTION", raising=False)

    result = FactExtractionService().extract("Supplier A", ["MOQ"], [])

    assert result.status == "disabled"


def test_embedding_settings_support_an_openai_compatible_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "openai/text-embedding-3-small")

    assert _embedding_settings() == {
        "model": "text-embedding-3-small",
        "api_key": "test-key",
        "model_name": "openai/text-embedding-3-small",
        "api_base": "https://openrouter.ai/api/v1",
    }


def test_chat_model_settings_support_an_openai_compatible_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-4.1-mini")

    assert _chat_model_settings() == {
        "model": "openai/gpt-4.1-mini",
        "temperature": 0,
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
    }
