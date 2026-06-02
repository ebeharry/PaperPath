from unittest.mock import MagicMock, patch
import pytest

from src.data_classes import Paper
from src.literature_review.embedder import EmbedderProtocol, embed_papers, make_embedder


def _paper(paper_id: str, title: str = "Title", abstract: str = "Abstract") -> Paper:
    return Paper(paper_id=paper_id, title=title, abstract=abstract, authors=[], year=2023, url=None, source="arxiv")


class _MockEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i)] for i in range(len(texts))]


# ---------------------------------------------------------------------------
# embed_papers
# ---------------------------------------------------------------------------

def test_embed_papers_constructs_correct_texts():
    captured = []

    class CapturingEmbedder:
        def embed(self, texts):
            captured.extend(texts)
            return [[0.0]] * len(texts)

    papers = [_paper("p1", title="T1", abstract="A1"), _paper("p2", title="T2", abstract="")]
    embed_papers(papers, CapturingEmbedder())
    assert captured[0] == "T1. A1"
    assert captured[1] == "T2."


def test_embed_papers_returns_paper_id_to_vector_map():
    papers = [_paper("id-a"), _paper("id-b")]
    result = embed_papers(papers, _MockEmbedder())
    assert set(result.keys()) == {"id-a", "id-b"}
    assert isinstance(result["id-a"], list)


def test_embed_papers_empty_list():
    result = embed_papers([], _MockEmbedder())
    assert result == {}


def test_embed_papers_strips_trailing_period_for_empty_abstract():
    captured = []

    class CapturingEmbedder:
        def embed(self, texts):
            captured.extend(texts)
            return [[0.0]] * len(texts)

    papers = [_paper("p1", title="T1", abstract="")]
    embed_papers(papers, CapturingEmbedder())
    assert captured[0] == "T1."


# ---------------------------------------------------------------------------
# make_embedder
# ---------------------------------------------------------------------------

def test_make_embedder_unknown_raises():
    with pytest.raises(ValueError, match="Unknown embed backend"):
        make_embedder("unknown")


def test_make_embedder_openai_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_embedder("openai")


def test_make_embedder_openai_raises_when_package_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import sys
    original = sys.modules.get("openai")
    sys.modules["openai"] = None  # simulate missing
    try:
        with pytest.raises((ImportError, TypeError)):
            make_embedder("openai")
    finally:
        if original is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = original


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------

def test_openai_embedder_calls_api_with_correct_args(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_openai = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1, 0.2]
    mock_openai.OpenAI.return_value.embeddings.create.return_value.data = [mock_embedding]

    with patch.dict("sys.modules", {"openai": mock_openai}):
        from src.literature_review import embedder as emb_mod
        import importlib
        importlib.reload(emb_mod)
        embedder = emb_mod.OpenAIEmbedder(api_key="test-key")
        result = embedder.embed(["hello world"])

    assert result == [[0.1, 0.2]]


# ---------------------------------------------------------------------------
# LocalEmbedder
# ---------------------------------------------------------------------------

def test_local_embedder_calls_encode(monkeypatch):
    import numpy as np
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_st.SentenceTransformer.return_value = mock_model

    with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
        from src.literature_review import embedder as emb_mod
        import importlib
        importlib.reload(emb_mod)
        embedder = emb_mod.LocalEmbedder()
        result = embedder.embed(["text one", "text two"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_local_embedder_raises_without_package():
    import sys
    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = None
    try:
        with pytest.raises((ImportError, TypeError)):
            from src.literature_review import embedder as emb_mod
            import importlib
            importlib.reload(emb_mod)
            emb_mod.LocalEmbedder()
    finally:
        if original is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = original


# ---------------------------------------------------------------------------
# EmbedderProtocol structural check
# ---------------------------------------------------------------------------

def test_mock_embedder_satisfies_protocol():
    assert isinstance(_MockEmbedder(), EmbedderProtocol)
