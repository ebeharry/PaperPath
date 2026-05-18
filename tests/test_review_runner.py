from unittest.mock import patch

from src.literature_review.data_classes import Paper
from src.literature_review.review_runner import run


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
def test_run_forwards_sort_to_ss_only(mock_ss, mock_arxiv):
    run("neural networks", sort="citationCount:desc")
    mock_ss.assert_called_once_with("neural networks", limit=20, sort="citationCount:desc", year=None)
    mock_arxiv.assert_called_once_with("neural networks", limit=20, sort=None, year=None)


@patch("src.literature_review.review_runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_routes_arxiv_sort_to_arxiv_only(mock_ss, mock_arxiv):
    run("neural networks", sort="submittedDate:desc")
    mock_ss.assert_called_once_with("neural networks", limit=20, sort=None, year=None)
    mock_arxiv.assert_called_once_with("neural networks", limit=20, sort="submittedDate:desc", year=None)


@patch("src.literature_review.review_runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_passes_none_sort_to_both(mock_ss, mock_arxiv):
    run("neural networks", sort=None)
    mock_ss.assert_called_once_with("neural networks", limit=20, sort=None, year=None)
    mock_arxiv.assert_called_once_with("neural networks", limit=20, sort=None, year=None)


@patch("src.literature_review.review_runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.literature_review.review_runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_forwards_year(mock_ss, mock_arxiv):
    run("neural networks", year="2020:2023")
    mock_ss.assert_called_once_with("neural networks", limit=20, sort=None, year="2020:2023")
    mock_arxiv.assert_called_once_with("neural networks", limit=20, sort=None, year="2020:2023")
