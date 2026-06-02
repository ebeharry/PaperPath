import json
import pytest
from unittest.mock import MagicMock, patch

from src.literature_review.data_classes import ClusterAnalysis, GapAnalysisReport, Paper
from src.literature_review.gap_analysis import (
    LLMClientProtocol,
    OpenRouterLLMClient,
    analyse_cluster,
    build_gap_report,
    make_llm_client,
    _build_cluster_prompt,
    _parse_llm_response,
)


def _paper(paper_id: str, title: str = "Title", abstract: str = "Abstract.") -> Paper:
    return Paper(paper_id=paper_id, title=title, abstract=abstract, authors=[], year=2023, url=None, source="arxiv")


class _MockLLMClient:
    def complete(self, prompt: str) -> str:
        return json.dumps({
            "what_exists": "Existing work summary.",
            "what_is_contested": "Contested area.",
            "what_is_missing": "Missing research.",
        })


# ---------------------------------------------------------------------------
# _build_cluster_prompt
# ---------------------------------------------------------------------------

def test_build_cluster_prompt_contains_query():
    papers = [_paper("p1", title="T1")]
    prompt = _build_cluster_prompt(papers, "graph neural networks")
    assert "graph neural networks" in prompt


def test_build_cluster_prompt_contains_paper_titles():
    papers = [_paper("p1", title="My Paper Title")]
    prompt = _build_cluster_prompt(papers, "query")
    assert "My Paper Title" in prompt


def test_build_cluster_prompt_truncates_abstract():
    long_abstract = "x" * 500
    papers = [_paper("p1", title="T", abstract=long_abstract)]
    prompt = _build_cluster_prompt(papers, "query")
    assert long_abstract not in prompt
    assert "x" * 300 in prompt


def test_build_cluster_prompt_requests_json_keys():
    papers = [_paper("p1")]
    prompt = _build_cluster_prompt(papers, "query")
    assert "what_exists" in prompt
    assert "what_is_contested" in prompt
    assert "what_is_missing" in prompt


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------

def test_parse_llm_response_valid_json():
    raw = json.dumps({"what_exists": "A", "what_is_contested": "B", "what_is_missing": "C"})
    exists, contested, missing = _parse_llm_response(raw)
    assert exists == "A"
    assert contested == "B"
    assert missing == "C"


def test_parse_llm_response_json_in_code_fence():
    raw = '```json\n{"what_exists": "A", "what_is_contested": "B", "what_is_missing": "C"}\n```'
    exists, contested, missing = _parse_llm_response(raw)
    assert exists == "A"
    assert contested == "B"
    assert missing == "C"


def test_parse_llm_response_malformed_falls_back_gracefully():
    raw = "This is not JSON at all."
    exists, contested, missing = _parse_llm_response(raw)
    assert exists == raw
    assert contested == ""
    assert missing == ""


def test_parse_llm_response_partial_json_falls_back():
    raw = '{"what_exists": "only this"}'
    exists, contested, missing = _parse_llm_response(raw)
    assert exists == "only this"
    assert contested == ""
    assert missing == ""


# ---------------------------------------------------------------------------
# analyse_cluster
# ---------------------------------------------------------------------------

def test_analyse_cluster_returns_cluster_analysis():
    papers = [_paper("p1"), _paper("p2")]
    result = analyse_cluster(0, papers, "query", _MockLLMClient())
    assert isinstance(result, ClusterAnalysis)
    assert result.cluster_id == 0
    assert result.paper_ids == ["p1", "p2"]
    assert result.what_exists == "Existing work summary."
    assert result.what_is_contested == "Contested area."
    assert result.what_is_missing == "Missing research."


def test_analyse_cluster_id_is_preserved():
    papers = [_paper("p1")]
    result = analyse_cluster(42, papers, "query", _MockLLMClient())
    assert result.cluster_id == 42


# ---------------------------------------------------------------------------
# build_gap_report
# ---------------------------------------------------------------------------

def test_build_gap_report_returns_gap_analysis_report():
    clusters = {0: [_paper("p1"), _paper("p2")], 1: [_paper("p3")]}
    result = build_gap_report("my query", clusters, _MockLLMClient())
    assert isinstance(result, GapAnalysisReport)
    assert result.query == "my query"


def test_build_gap_report_cluster_count_matches_input():
    clusters = {0: [_paper("p1")], 1: [_paper("p2")], 2: [_paper("p3")]}
    result = build_gap_report("query", clusters, _MockLLMClient())
    assert len(result.clusters) == 3


def test_build_gap_report_overall_fields_are_populated():
    clusters = {0: [_paper("p1")]}
    result = build_gap_report("query", clusters, _MockLLMClient())
    assert result.overall_what_exists != ""
    assert isinstance(result.overall_what_is_contested, str)
    assert isinstance(result.overall_what_is_missing, str)


def test_build_gap_report_clusters_sorted_by_id():
    clusters = {2: [_paper("p3")], 0: [_paper("p1")], 1: [_paper("p2")]}
    result = build_gap_report("query", clusters, _MockLLMClient())
    assert [c.cluster_id for c in result.clusters] == [0, 1, 2]


def test_build_gap_report_single_cluster():
    clusters = {0: [_paper("p1"), _paper("p2")]}
    result = build_gap_report("query", clusters, _MockLLMClient())
    assert len(result.clusters) == 1
    assert result.clusters[0].cluster_id == 0


# ---------------------------------------------------------------------------
# make_llm_client
# ---------------------------------------------------------------------------

def test_make_llm_client_unknown_raises():
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        make_llm_client("unknown")


def test_make_llm_client_openai_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_llm_client("openai")


def test_make_llm_client_anthropic_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_llm_client("anthropic")


def test_make_llm_client_openrouter_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        make_llm_client("openrouter")


def test_make_llm_client_openrouter_constructs_with_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with patch("openai.OpenAI"):
        client = make_llm_client("openrouter")
    assert isinstance(client, OpenRouterLLMClient)


# ---------------------------------------------------------------------------
# LLMClientProtocol structural check
# ---------------------------------------------------------------------------

def test_mock_llm_satisfies_protocol():
    assert isinstance(_MockLLMClient(), LLMClientProtocol)
