from __future__ import annotations
import contextlib
import io
import os
from typing import Protocol, runtime_checkable

from src.data_classes import Paper


@runtime_checkable
class EmbedderProtocol(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    MODEL = "text-embedding-3-small"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set; pass api_key= or set the environment variable")
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("openai package is required for --embed-backend openai; run: pip install openai")
        self._client = _openai.OpenAI(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.MODEL, input=texts)
        return [item.embedding for item in response.data]


class LocalEmbedder:
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package is required for --embed-backend local; "
                "run: pip install sentence-transformers"
            )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


def make_embedder(backend: str, **kwargs) -> EmbedderProtocol:
    if backend == "openai":
        return OpenAIEmbedder(**kwargs)
    if backend == "local":
        return LocalEmbedder(**kwargs)
    raise ValueError(f"Unknown embed backend: {backend!r}; choices are 'openai', 'local'")


def embed_papers(papers: list[Paper], embedder: EmbedderProtocol) -> dict[str, list[float]]:
    texts = [f"{p.title}. {p.abstract}".strip() for p in papers]
    vectors = embedder.embed(texts)
    return {p.paper_id: v for p, v in zip(papers, vectors)}
