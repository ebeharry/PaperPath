from unittest.mock import patch

from src.literature_review.data_classes import Paper
from src.literature_review.review_runner import run


_SAMPLE_PAPERS = [
    Paper(
        paper_id="p1",
        title="Test Paper",
        abstract="Abstract.",
        authors=["Author A"],
        year=2023,
        url=None,
        source="semantic_scholar",
    )
]


@patch("src.literature_review.review_runner.search_semantic_scholar")
def test_run_delegates_to_semantic_scholar(mock_search):
    mock_search.return_value = _SAMPLE_PAPERS
    run("neural networks")
    mock_search.assert_called_once_with(
        "neural networks", limit=20, sort=None, publication_date_or_year=None
    )


@patch("src.literature_review.review_runner.search_semantic_scholar")
def test_run_returns_papers_unchanged(mock_search):
    mock_search.return_value = _SAMPLE_PAPERS
    result = run("neural networks")
    assert result == _SAMPLE_PAPERS


@patch("src.literature_review.review_runner.search_semantic_scholar")
def test_run_forwards_sort(mock_search):
    mock_search.return_value = _SAMPLE_PAPERS
    run("neural networks", sort="citationCount:desc")
    mock_search.assert_called_once_with(
        "neural networks", limit=20, sort="citationCount:desc", publication_date_or_year=None
    )


@patch("src.literature_review.review_runner.search_semantic_scholar")
def test_run_forwards_publication_date_or_year(mock_search):
    mock_search.return_value = _SAMPLE_PAPERS
    run("neural networks", publication_date_or_year="2020:2023")
    mock_search.assert_called_once_with(
        "neural networks", limit=20, sort=None, publication_date_or_year="2020:2023"
    )
