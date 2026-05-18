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
    result = run("neural networks")
    mock_search.assert_called_once_with("neural networks")


@patch("src.literature_review.review_runner.search_semantic_scholar")
def test_run_returns_papers_unchanged(mock_search):
    mock_search.return_value = _SAMPLE_PAPERS
    result = run("neural networks")
    assert result == _SAMPLE_PAPERS
