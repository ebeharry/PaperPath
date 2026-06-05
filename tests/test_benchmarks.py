from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data_classes import (
    AbstractDraft,
    DraftReport,
    Paper,
    RelatedWorkDraft,
    RelatedWorkSubsection,
)
from benchmarks.metrics.abstract_quality import abstract_embedding_similarity
from benchmarks.metrics.latex_check import compile_success
from benchmarks.metrics.retrieval import hallucinated_citation_rate, retrieval_relevance_at_k
from benchmarks.runner import BenchmarkResult, run_benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paper(pid: str = "p1") -> Paper:
    return Paper(
        paper_id=pid,
        title="Title",
        abstract="Abstract.",
        authors=["Alice Smith"],
        year=2023,
        url=None,
        source="semantic_scholar",
    )


def _subsection(cited_ids: list[str]) -> RelatedWorkSubsection:
    return RelatedWorkSubsection(
        cluster_id=0,
        theme="Theme",
        paragraph="Para.",
        cited_paper_ids=cited_ids,
    )


def _draft_report(cited_ids: list[str], abstract_text: str = "Generated abstract.") -> DraftReport:
    return DraftReport(
        input="query",
        related_work=RelatedWorkDraft(
            subsections=[_subsection(cited_ids)],
            full_text="Full text.",
        ),
        abstract=AbstractDraft(
            background="Background.",
            prior_work_summary="Prior work.",
            gap="Gap.",
            proposed_approach="Approach.",
            expected_contribution="Contribution.",
            full_text=abstract_text,
        ),
    )


class _MockEmbedder:
    def __init__(self, vectors: list[list[float]] | None = None):
        self._vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._vectors is not None:
            return self._vectors[: len(texts)]
        return [[1.0, 0.0] for _ in texts]


# ---------------------------------------------------------------------------
# hallucinated_citation_rate
# ---------------------------------------------------------------------------

def test_hallucinated_citation_rate_zero():
    subs = [_subsection(["p1", "p2"])]
    assert hallucinated_citation_rate(subs, {"p1", "p2"}) == 0.0


def test_hallucinated_citation_rate_all():
    subs = [_subsection(["x", "y"])]
    assert hallucinated_citation_rate(subs, {"p1", "p2"}) == 1.0


def test_hallucinated_citation_rate_partial():
    subs = [_subsection(["p1", "x"])]
    result = hallucinated_citation_rate(subs, {"p1"})
    assert result == pytest.approx(0.5)


def test_hallucinated_citation_rate_no_subsections():
    assert hallucinated_citation_rate([], {"p1"}) == 0.0


def test_hallucinated_citation_rate_empty_cited_ids():
    subs = [_subsection([])]
    assert hallucinated_citation_rate(subs, {"p1"}) == 0.0


def test_hallucinated_citation_rate_across_multiple_subsections():
    subs = [_subsection(["p1"]), _subsection(["x"])]
    result = hallucinated_citation_rate(subs, {"p1"})
    assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# abstract_embedding_similarity
# ---------------------------------------------------------------------------

def test_abstract_embedding_similarity_identical_vectors():
    embedder = _MockEmbedder([[1.0, 0.0], [1.0, 0.0]])
    result = abstract_embedding_similarity("text", "text", embedder)
    assert result == pytest.approx(1.0)


def test_abstract_embedding_similarity_orthogonal_vectors():
    embedder = _MockEmbedder([[1.0, 0.0], [0.0, 1.0]])
    result = abstract_embedding_similarity("text a", "text b", embedder)
    assert result == pytest.approx(0.0)


def test_abstract_embedding_similarity_opposite_vectors():
    embedder = _MockEmbedder([[1.0, 0.0], [-1.0, 0.0]])
    result = abstract_embedding_similarity("a", "b", embedder)
    assert result == pytest.approx(-1.0)


def test_abstract_embedding_similarity_unnormalized_inputs():
    # should normalize before dot product — result should still be 1.0
    embedder = _MockEmbedder([[3.0, 0.0], [5.0, 0.0]])
    result = abstract_embedding_similarity("a", "b", embedder)
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compile_success
# ---------------------------------------------------------------------------

def test_compile_success_exit_zero():
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result):
        assert compile_success("/tmp/paper.tex") is True


def test_compile_success_exit_nonzero():
    mock_result = MagicMock()
    mock_result.returncode = 1
    with patch("subprocess.run", return_value=mock_result):
        assert compile_success("/tmp/paper.tex") is False


def test_compile_success_pdflatex_missing_raises():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="pdflatex not found"):
            compile_success("/tmp/paper.tex")


def test_compile_success_timeout_returns_false():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pdflatex", timeout=60)):
        assert compile_success("/tmp/paper.tex") is False


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

_GOLD = {
    "paper_id": "gold123",
    "title": "Gold Paper",
    "query": "some query",
    "gold_abstract": "This is the gold abstract.",
    "gold_references": ["ref1", "ref2"],
}


def test_run_benchmark_returns_dataclass():
    papers = [_paper("ref1")]
    result = run_benchmark(_GOLD, papers)
    assert isinstance(result, BenchmarkResult)


def test_run_benchmark_hcr_none_without_draft():
    papers = [_paper("p1")]
    result = run_benchmark(_GOLD, papers)
    assert result.hallucinated_citation_rate is None


def test_run_benchmark_hcr_populated_with_draft():
    papers = [_paper("p1")]
    draft = _draft_report(["p1"])
    result = run_benchmark(_GOLD, papers, draft_report=draft)
    assert result.hallucinated_citation_rate == pytest.approx(0.0)


def test_run_benchmark_abs_sim_none_without_embedder():
    papers = [_paper("p1")]
    draft = _draft_report(["p1"])
    result = run_benchmark(_GOLD, papers, draft_report=draft, embedder=None)
    assert result.abstract_embedding_similarity is None


def test_run_benchmark_abs_sim_populated_with_embedder():
    papers = [_paper("p1")]
    draft = _draft_report(["p1"], abstract_text="Generated abstract.")
    embedder = _MockEmbedder([[1.0, 0.0], [1.0, 0.0]])
    result = run_benchmark(_GOLD, papers, draft_report=draft, embedder=embedder)
    assert result.abstract_embedding_similarity is not None


def test_run_benchmark_compile_none_without_tex_path():
    result = run_benchmark(_GOLD, [_paper("p1")])
    assert result.compile_success is None


def test_run_benchmark_gold_ids_in_result():
    result = run_benchmark(_GOLD, [])
    assert result.gold_paper_id == "gold123"
    assert result.gold_title == "Gold Paper"


# ---------------------------------------------------------------------------
# retrieval_relevance_at_k
# ---------------------------------------------------------------------------

def test_retrieval_relevance_at_k_identical():
    embedder = _MockEmbedder([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    papers = [_paper("p1"), _paper("p2")]
    result = retrieval_relevance_at_k(papers, "gold abstract", embedder, k=2)
    assert result == pytest.approx(1.0)


def test_retrieval_relevance_at_k_orthogonal():
    embedder = _MockEmbedder([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    papers = [_paper("p1"), _paper("p2")]
    result = retrieval_relevance_at_k(papers, "gold abstract", embedder, k=2)
    assert result == pytest.approx(0.0)


def test_retrieval_relevance_at_k_respects_k():
    # k=1: only p1 is embedded; second vector is gold abstract
    embedder = _MockEmbedder([[1.0, 0.0], [1.0, 0.0]])
    papers = [_paper("p1"), _paper("p2")]
    result = retrieval_relevance_at_k(papers, "gold abstract", embedder, k=1)
    assert result == pytest.approx(1.0)


def test_retrieval_relevance_at_k_empty_papers():
    result = retrieval_relevance_at_k([], "gold abstract", _MockEmbedder(), k=10)
    assert result == 0.0


def test_retrieval_relevance_at_k_empty_gold_abstract():
    result = retrieval_relevance_at_k([_paper("p1")], "", _MockEmbedder(), k=10)
    assert result == 0.0


def test_retrieval_relevance_at_k_unnormalized_vectors():
    # vectors with different magnitudes but same direction → sim = 1.0
    embedder = _MockEmbedder([[3.0, 0.0], [5.0, 0.0]])
    result = retrieval_relevance_at_k([_paper("p1")], "gold abstract", embedder, k=1)
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# run_benchmark — retrieval_relevance_at_k field
# ---------------------------------------------------------------------------

def test_run_benchmark_retrieval_relevance_populated_with_embedder():
    papers = [_paper("p1")]
    embedder = _MockEmbedder([[1.0, 0.0], [1.0, 0.0]])
    result = run_benchmark(_GOLD, papers, embedder=embedder)
    assert result.retrieval_relevance_at_k is not None


def test_run_benchmark_retrieval_relevance_none_without_embedder():
    papers = [_paper("p1")]
    result = run_benchmark(_GOLD, papers, embedder=None)
    assert result.retrieval_relevance_at_k is None
