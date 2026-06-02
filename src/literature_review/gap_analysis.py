from __future__ import annotations
import json
import logging
import os
import re
from typing import Protocol, runtime_checkable

from src.data_classes import ClusterAnalysis, GapAnalysisReport, Paper

logger = logging.getLogger(__name__)

_ABSTRACT_LIMIT = 300


@runtime_checkable
class LLMClientProtocol(Protocol):
    def complete(self, prompt: str) -> str: ...


class OpenAILLMClient:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set; pass api_key= or set the environment variable")
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("openai package is required for --llm-backend openai; run: pip install openai")
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
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set; pass api_key= or set the environment variable")
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("anthropic package is required for --llm-backend anthropic; run: pip install anthropic")
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
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not set; pass api_key= or set the environment variable")
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("openai package is required for --llm-backend openrouter; run: pip install openai")
        self._client = _openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        self._model = model

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


def make_llm_client(backend: str, **kwargs) -> LLMClientProtocol:
    if backend == "openai":
        return OpenAILLMClient(**kwargs)
    if backend == "anthropic":
        return AnthropicLLMClient(**kwargs)
    if backend == "openrouter":
        return OpenRouterLLMClient(**kwargs)
    raise ValueError(f"Unknown LLM backend: {backend!r}; choices are 'openai', 'anthropic', 'openrouter'")


def _build_cluster_prompt(papers: list[Paper], query: str) -> str:
    paper_lines = []
    for p in papers:
        abstract_snippet = p.abstract[:_ABSTRACT_LIMIT]
        paper_lines.append(f"- {p.title}: {abstract_snippet}")
    papers_text = "\n".join(paper_lines)
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
    # try to extract JSON from code fences or bare object
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", response, re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(1))
            return (
                data.get("what_exists", ""),
                data.get("what_is_contested", ""),
                data.get("what_is_missing", ""),
            )
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse LLM JSON response; using raw text fallback")

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
    overall = analyse_cluster(-1, all_papers, query, llm)
    return GapAnalysisReport(
        input=query,
        clusters=cluster_analyses,
        overall_what_exists=overall.what_exists,
        overall_what_is_contested=overall.what_is_contested,
        overall_what_is_missing=overall.what_is_missing,
    )
