from __future__ import annotations

import pytest

from src.data_classes import Paper
from src.literature_review.ranker import rank_papers


def _paper(
    paper_id: str = "p1",
    title: str = "A paper title",
    abstract: str = "An abstract.",
    year: int | None = 2020,
    citation_count: int | None = None,
    source: str = "arxiv",
) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=[],
        year=year,
        url=None,
        citation_count=citation_count,
        source=source,
    )


class _ConstantEmbedder:
    """Returns identical unit vectors for every text — isolates non-similarity components."""
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0]] * len(texts)


class _MockEmbedder:
    """Returns a pre-configured list of vectors in order."""
    def __init__(self, vectors: list[list[float]]):
        self._vectors = vectors
        self.call_count = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return self._vectors[: len(texts)]


# --- empty / single ---

def test_rank_papers_empty_returns_empty():
    assert rank_papers("query", [], _ConstantEmbedder()) == []


def test_rank_papers_empty_does_not_call_embedder():
    class _TrackingEmbedder:
        called = False
        def embed(self, texts):
            _TrackingEmbedder.called = True
            return []
    rank_papers("query", [], _TrackingEmbedder())
    assert not _TrackingEmbedder.called


def test_rank_papers_single_paper():
    result = rank_papers("query", [_paper("p1")], _ConstantEmbedder())
    assert len(result) == 1
    assert result[0].paper_id == "p1"
    assert isinstance(result[0].final_score, float)


def test_rank_papers_sets_all_score_components():
    result = rank_papers("q", [_paper("p1", year=2026)], _ConstantEmbedder(), current_year=2026)
    p = result[0]
    assert p.rank == 1
    assert p.title_score is not None
    assert p.abstract_score is not None
    assert p.recency_score is not None
    assert p.citation_score is not None
    assert p.final_score is not None


# --- similarity ordering ---

def test_rank_papers_orders_by_score():
    # query [1,0], p0 title similar [1,0], p1 title dissimilar [0,1]
    vectors = [
        [1.0, 0.0],  # query
        [1.0, 0.0],  # p0 title — similar
        [0.5, 0.5],  # p0 abstract
        [0.0, 1.0],  # p1 title — dissimilar
        [0.0, 1.0],  # p1 abstract
    ]
    papers = [_paper("p0", title="similar"), _paper("p1", title="different")]
    result = rank_papers("similar", papers, _MockEmbedder(vectors))
    assert result[0].paper_id == "p0"
    assert result[1].paper_id == "p1"
    assert result[0].final_score > result[1].final_score


# --- recency ---

def test_recency_score_current_year():
    # All embeddings identical [1,0]: title_sim=1, abstract_sim=1, recency=1, citation_pct=0
    # score = 0.35 + 0.35 + 0.20 + 0.0 = 0.90
    result = rank_papers("q", [_paper("p1", year=2026)], _ConstantEmbedder(), current_year=2026)
    assert result[0].final_score == pytest.approx(0.90, abs=1e-3)


def test_recency_score_five_years_ago():
    # recency = max(0, 1 - 5/10) = 0.5
    # score = 0.35 + 0.35 + 0.10 + 0.0 = 0.80
    result = rank_papers("q", [_paper("p1", year=2021)], _ConstantEmbedder(), current_year=2026)
    assert result[0].final_score == pytest.approx(0.80, abs=1e-3)


def test_recency_score_old_paper_clamps_to_zero():
    # year=2006, current=2026: 1 - 20/10 = -1 → clamped to 0
    # score = 0.35 + 0.35 + 0.0 + 0.0 = 0.70
    result = rank_papers("q", [_paper("p1", year=2006)], _ConstantEmbedder(), current_year=2026)
    assert result[0].final_score == pytest.approx(0.70, abs=1e-3)


def test_recency_score_missing_year():
    # year=None → recency=0.0; score = 0.70
    result = rank_papers("q", [_paper("p1", year=None)], _ConstantEmbedder(), current_year=2026)
    assert result[0].final_score == pytest.approx(0.70, abs=1e-3)


# --- citation percentile ---

def test_citation_percentile_ordering():
    papers = [
        _paper("low", citation_count=10),
        _paper("high", citation_count=1000),
    ]
    result = rank_papers("q", papers, _ConstantEmbedder(), current_year=2026)
    assert result[0].paper_id == "high"


def test_citation_percentile_values():
    # 2 papers: low rank 0 → 0.0, high rank 1 → 1.0
    # high: 0.35+0.35+0.0+0.10 = 0.80  |  low: 0.35+0.35+0.0+0.0 = 0.70
    papers = [
        _paper("low", year=None, citation_count=1),
        _paper("high", year=None, citation_count=100),
    ]
    result = rank_papers("q", papers, _ConstantEmbedder(), current_year=2026)
    scores = {p.paper_id: p.final_score for p in result}
    assert scores["high"] == pytest.approx(0.80, abs=1e-3)
    assert scores["low"] == pytest.approx(0.70, abs=1e-3)


def test_citation_percentile_all_none():
    papers = [_paper("p1", citation_count=None), _paper("p2", citation_count=None)]
    result = rank_papers("q", papers, _ConstantEmbedder(), current_year=2026)
    assert len(result) == 2
    assert result[0].final_score == pytest.approx(result[1].final_score, abs=1e-6)


def test_citation_percentile_single_paper_with_count():
    # Only one paper with a count → stays 0.0; score = 0.70
    result = rank_papers("q", [_paper("p1", year=None, citation_count=9999)], _ConstantEmbedder(), current_year=2026)
    assert result[0].final_score == pytest.approx(0.70, abs=1e-3)


# --- score bounds ---

def test_scores_bounded_by_one():
    ss_paper = Paper(
        paper_id="ss", title="T", abstract="A", authors=[],
        year=2026, url=None, citation_count=999, source="semantic_scholar"
    )
    low_paper = Paper(
        paper_id="low", title="T", abstract="A", authors=[],
        year=2026, url=None, citation_count=1, source="semantic_scholar"
    )
    result = rank_papers("q", [ss_paper, low_paper], _ConstantEmbedder(), current_year=2026)
    for p in result:
        assert p.final_score <= 1.0 + 1e-9


# --- rank field ---

def test_rank_field_assigned_sequentially():
    papers = [_paper("p0"), _paper("p1"), _paper("p2")]
    result = rank_papers("q", papers, _ConstantEmbedder(), current_year=2026)
    ranks = [p.rank for p in result]
    assert ranks == [1, 2, 3]
