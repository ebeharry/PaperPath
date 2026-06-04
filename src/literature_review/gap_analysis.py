from __future__ import annotations
import logging
import os
from typing import Protocol, runtime_checkable

from src.data_classes import ClusterAnalysis, GapAnalysisReport, Paper
from src.utils import ABSTRACT_LIMIT, parse_json_from_response

logger = logging.getLogger(__name__)

_ALL_PAPERS_CLUSTER_ID = -1


@runtime_checkable
class LLMClientProtocol(Protocol):
    def complete(self, prompt: str) -> str: ...


def _require_api_key(env_var: str, api_key: str | None) -> str:
    key = api_key or os.environ.get(env_var)
    if not key:
        raise ValueError(f"{env_var} is not set; pass api_key= or set the environment variable")
    return key


def _require_import(package: str, backend: str):
    try:
        import importlib
        return importlib.import_module(package)
    except ImportError:
        raise ImportError(
            f"{package} package is required for --llm-backend {backend}; run: pip install {package}"
        )


class OpenAILLMClient:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        key = _require_api_key("OPENAI_API_KEY", api_key)
        _openai = _require_import("openai", "openai")
        self._client = _openai.OpenAI(api_key=key)
        self._model = model

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


class AnthropicLLMClient:
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        key = _require_api_key("ANTHROPIC_API_KEY", api_key)
        _anthropic = _require_import("anthropic", "anthropic")
        self._client = _anthropic.Anthropic(api_key=key)
        self._model = model

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenRouterLLMClient:
    def __init__(self, model: str = "openai/gpt-oss-120b:free", api_key: str | None = None):
        key = _require_api_key("OPENROUTER_API_KEY", api_key)
        _openai = _require_import("openai", "openrouter")
        self._client = _openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        self._model = model

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


_VALID_BACKENDS: dict[str, type] = {
    "openai": OpenAILLMClient,
    "anthropic": AnthropicLLMClient,
    "openrouter": OpenRouterLLMClient,
}

_VALID_KWARGS = {"model", "api_key"}


def make_llm_client(backend: str, **kwargs) -> LLMClientProtocol:
    cls = _VALID_BACKENDS.get(backend)
    if cls is None:
        raise ValueError(f"Unknown LLM backend: {backend!r}; choices are {list(_VALID_BACKENDS)}")
    unknown = set(kwargs) - _VALID_KWARGS
    if unknown:
        raise TypeError(f"make_llm_client() got unexpected keyword arguments: {unknown}")
    return cls(**kwargs)


def _build_cluster_prompt(papers: list[Paper], query: str) -> str:
    papers_text = "\n".join(f"- {p.title}: {(p.abstract or '')[:ABSTRACT_LIMIT]}" for p in papers)
    return (
        f'You are a research analyst. Given a research query and a set of related papers, '
        f'identify the research landscape.\n\n'
        f'Query: "{query}"\n\n'
        f'Papers:\n{papers_text}\n\n'
        f'Respond with ONLY a JSON object (no markdown, no code fences) with exactly these three keys:\n'
        f'{{"what_exists": "...", "what_is_contested": "...", "what_is_missing": "..."}}\n\n'
        f'Each value should be 2-4 sentences.'
    )


def _parse_llm_response(response: str) -> tuple[str, str, str]:
    data = parse_json_from_response(response)
    if data:
        return (
            data.get("what_exists", ""),
            data.get("what_is_contested", ""),
            data.get("what_is_missing", ""),
        )
    return response, "", ""


def analyse_cluster(
    cluster_id: int,
    papers: list[Paper],
    query: str,
    llm: LLMClientProtocol,
) -> ClusterAnalysis:
    prompt = _build_cluster_prompt(papers, query)
    raw = llm.complete(prompt)
    what_exists, what_is_contested, what_is_missing = _parse_llm_response(raw)
    return ClusterAnalysis(
        cluster_id=cluster_id,
        paper_ids=[p.paper_id for p in papers],
        what_exists=what_exists,
        what_is_contested=what_is_contested,
        what_is_missing=what_is_missing,
    )


def build_gap_report(
    query: str,
    clusters: dict[int, list[Paper]],
    llm: LLMClientProtocol,
) -> GapAnalysisReport:
    cluster_analyses = [
        analyse_cluster(cid, papers, query, llm)
        for cid, papers in sorted(clusters.items())
    ]
    all_papers = [p for papers in clusters.values() for p in papers]
    overall = analyse_cluster(_ALL_PAPERS_CLUSTER_ID, all_papers, query, llm)
    return GapAnalysisReport(
        input=query,
        clusters=cluster_analyses,
        overall_what_exists=overall.what_exists,
        overall_what_is_contested=overall.what_is_contested,
        overall_what_is_missing=overall.what_is_missing,
    )
