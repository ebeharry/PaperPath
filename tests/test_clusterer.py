import numpy as np
import pytest

from src.data_classes import Paper
from src.literature_review.clusterer import cluster_papers, _select_k


def _paper(paper_id: str) -> Paper:
    return Paper(paper_id=paper_id, title=f"Paper {paper_id}", abstract="Abstract.", authors=[], year=2023, url=None, source="arxiv")


def _embeddings(papers: list[Paper], vectors: list[list[float]]) -> dict[str, list[float]]:
    return {p.paper_id: v for p, v in zip(papers, vectors)}


# ---------------------------------------------------------------------------
# _select_k
# ---------------------------------------------------------------------------

def test_select_k_returns_1_for_small_n():
    tiny = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    assert _select_k(tiny) == 1


def test_select_k_detects_two_clear_clusters():
    cluster_a = [[1.0, 0.0, 0.0]] * 5
    cluster_b = [[0.0, 1.0, 0.0]] * 5
    matrix = np.array(cluster_a + cluster_b, dtype=float)
    assert _select_k(matrix) == 2


def test_select_k_handles_degenerate_identical_vectors():
    matrix = np.array([[1.0, 0.0]] * 6, dtype=float)
    # silhouette_score is undefined for all-same labels → falls back to score=-1 → k stays at 2
    # or gracefully returns any k; the key invariant is: no exception raised
    result = _select_k(matrix)
    assert isinstance(result, int)
    assert result >= 1


# ---------------------------------------------------------------------------
# cluster_papers
# ---------------------------------------------------------------------------

def test_cluster_papers_empty_embeddings_returns_empty():
    papers = [_paper("p1"), _paper("p2")]
    result = cluster_papers(papers, {})
    assert result == {}


def test_cluster_papers_excludes_papers_missing_from_embeddings():
    papers = [_paper("p1"), _paper("p2"), _paper("p3")]
    embeddings = {"p1": [1.0, 0.0], "p2": [0.9, 0.1]}
    result = cluster_papers(papers, embeddings)
    all_ids = [p.paper_id for cluster in result.values() for p in cluster]
    assert "p3" not in all_ids
    assert "p1" in all_ids
    assert "p2" in all_ids


def test_cluster_papers_single_paper_returns_one_cluster():
    papers = [_paper("p1")]
    result = cluster_papers(papers, {"p1": [1.0, 0.0]})
    assert len(result) == 1
    assert result[0] == papers


def test_cluster_papers_two_clear_clusters():
    papers = [_paper(f"a{i}") for i in range(4)] + [_paper(f"b{i}") for i in range(4)]
    vecs_a = [[1.0, 0.0, 0.0]] * 4
    vecs_b = [[0.0, 1.0, 0.0]] * 4
    embeddings = _embeddings(papers, vecs_a + vecs_b)
    result = cluster_papers(papers, embeddings)
    assert len(result) == 2
    total = sum(len(v) for v in result.values())
    assert total == 8


def test_cluster_papers_all_papers_included_once():
    n = 8
    papers = [_paper(f"p{i}") for i in range(n)]
    vecs_a = [[1.0, 0.0, 0.0]] * 4
    vecs_b = [[0.0, 1.0, 0.0]] * 4
    embeddings = _embeddings(papers, vecs_a + vecs_b)
    result = cluster_papers(papers, embeddings)
    all_ids = [p.paper_id for cluster in result.values() for p in cluster]
    assert sorted(all_ids) == sorted(p.paper_id for p in papers)


def test_cluster_papers_returns_cluster_ids_as_ints():
    papers = [_paper(f"p{i}") for i in range(8)]
    vecs = [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4
    embeddings = _embeddings(papers, vecs)
    result = cluster_papers(papers, embeddings)
    for key in result:
        assert isinstance(key, int)


def test_cluster_papers_fewer_than_4_papers_returns_single_cluster():
    papers = [_paper("x"), _paper("y"), _paper("z")]
    embeddings = {"x": [1.0, 0.0], "y": [0.0, 1.0], "z": [0.5, 0.5]}
    result = cluster_papers(papers, embeddings)
    assert len(result) == 1
    assert 0 in result
    assert len(result[0]) == 3
