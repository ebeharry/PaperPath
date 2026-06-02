from unittest.mock import MagicMock, patch

import requests

from src.literature_review.data_classes import ClusterAnalysis, GapAnalysisReport, Paper
from src.literature_review.review_runner import run, run_with_analysis


_SS_PAPERS = [
    Paper(paper_id="p1", title="SS Paper", abstract="Abstract.", authors=["Author A"], year=2023, url=None, source="semantic_scholar")
]

_ARXIV_PAPERS = [
    Paper(paper_id="a1", title="ArXiv Paper", abstract="", authors=[], year=2023, url=None, source="arxiv")
]


def _dual_patch(ss_return=None, arxiv_return=None):
    ss = patch("src.literature_review.review_runner.search_semantic_scholar", return_value=ss_return or _SS_PAPERS)
    arxiv = patch("src.literature_review.review_runner.search_arxiv", return_value=arxiv_return or _ARXIV_PAPERS)
    return ss, arxiv


@patch("src.literature_review.review_runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_delegates_to_semantic_scholar(mock_ss, mock_arxiv):
    run("neural networks")
    mock_ss.assert_called_once_with("neural networks", limit=20, sort=None, year=None)


@patch("src.literature_review.review_runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_calls_search_arxiv(mock_ss, mock_arxiv):
    run("neural networks")
    mock_arxiv.assert_called_once_with("neural networks", limit=20, sort=None, year=None)


@patch("src.literature_review.review_runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_combines_results(mock_ss, mock_arxiv):
    result = run("neural networks")
    assert result == _SS_PAPERS + _ARXIV_PAPERS


@patch("src.literature_review.review_runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_routes_sorts_independently(mock_ss, mock_arxiv):
    run("neural networks", ss_sort="citationCount:desc", arxiv_sort="submittedDate:asc")
    mock_ss.assert_called_once_with("neural networks", limit=20, sort="citationCount:desc", year=None)
    mock_arxiv.assert_called_once_with("neural networks", limit=20, sort="submittedDate:asc", year=None)


@patch("src.literature_review.review_runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_passes_none_sort_to_both(mock_ss, mock_arxiv):
    run("neural networks")
    mock_ss.assert_called_once_with("neural networks", limit=20, sort=None, year=None)
    mock_arxiv.assert_called_once_with("neural networks", limit=20, sort=None, year=None)


@patch("src.literature_review.review_runner.search_arxiv")
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_deduplicates_by_title(mock_ss, mock_arxiv):
    dup = Paper(paper_id="a2", title="SS Paper", abstract="", authors=[], year=2023, url=None, source="arxiv")
    mock_arxiv.return_value = [dup]
    result = run("neural networks")
    assert len(result) == 1
    assert result[0].source == "semantic_scholar"


@patch("src.literature_review.review_runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_forwards_year(mock_ss, mock_arxiv):
    run("neural networks", year="2020:2023")
    mock_ss.assert_called_once_with("neural networks", limit=20, sort=None, year="2020:2023")
    mock_arxiv.assert_called_once_with("neural networks", limit=20, sort=None, year="2020:2023")


@patch("src.literature_review.review_runner.search_arxiv", side_effect=requests.ReadTimeout("timed out"))
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_continues_when_arxiv_times_out(mock_ss, mock_arxiv, capsys):
    result = run("neural networks")
    assert result == _SS_PAPERS
    assert "arxiv unavailable" in capsys.readouterr().err


@patch("src.literature_review.review_runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.literature_review.review_runner.search_semantic_scholar", side_effect=requests.ReadTimeout("timed out"))
def test_run_continues_when_ss_times_out(mock_ss, mock_arxiv, capsys):
    result = run("neural networks")
    assert result == _ARXIV_PAPERS
    assert "semantic scholar unavailable" in capsys.readouterr().err


@patch("src.literature_review.review_runner.search_arxiv", side_effect=requests.ReadTimeout("timed out"))
@patch("src.literature_review.review_runner.search_semantic_scholar", side_effect=requests.ReadTimeout("timed out"))
def test_run_raises_when_both_sources_fail(mock_ss, mock_arxiv):
    import pytest
    with pytest.raises(requests.ReadTimeout):
        run("neural networks")


# ---------------------------------------------------------------------------
# run_with_analysis
# ---------------------------------------------------------------------------

_MOCK_REPORT = GapAnalysisReport(
    query="neural networks",
    clusters=[ClusterAnalysis(cluster_id=0, paper_ids=["p1"], what_exists="A", what_is_contested="B", what_is_missing="C")],
    overall_what_exists="A",
    overall_what_is_contested="B",
    overall_what_is_missing="C",
)


@patch("src.literature_review.review_runner.build_gap_report", return_value=_MOCK_REPORT)
@patch("src.literature_review.review_runner.make_llm_client")
@patch("src.literature_review.review_runner.cluster_papers", return_value={0: _SS_PAPERS})
@patch("src.literature_review.review_runner.embed_papers", return_value={"p1": [0.1, 0.2]})
@patch("src.literature_review.review_runner.make_embedder")
@patch("src.literature_review.review_runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_with_analysis_returns_papers_and_report(
    mock_ss, mock_arxiv, mock_embedder, mock_embed_papers, mock_cluster, mock_llm, mock_report
):
    papers, report = run_with_analysis("neural networks")
    assert isinstance(papers, list)
    assert isinstance(report, GapAnalysisReport)


@patch("src.literature_review.review_runner.build_gap_report", return_value=_MOCK_REPORT)
@patch("src.literature_review.review_runner.make_llm_client")
@patch("src.literature_review.review_runner.cluster_papers", return_value={0: _SS_PAPERS})
@patch("src.literature_review.review_runner.embed_papers", return_value={"p1": [0.1, 0.2]})
@patch("src.literature_review.review_runner.make_embedder")
@patch("src.literature_review.review_runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_with_analysis_passes_backends(
    mock_ss, mock_arxiv, mock_embedder, mock_embed_papers, mock_cluster, mock_llm, mock_report
):
    run_with_analysis("neural networks", embed_backend="openai", llm_backend="anthropic")
    mock_embedder.assert_called_once_with("openai")
    mock_llm.assert_called_once_with("anthropic")
