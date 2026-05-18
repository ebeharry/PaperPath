import sys
from unittest.mock import patch

import pytest

from src.literature_review.data_classes import Paper
from src.main import main


def _make_paper(abstract="Short abstract.", url="https://example.com", authors=None, year=2023):
    return Paper(
        paper_id="p1",
        title="Test Paper",
        abstract=abstract,
        authors=authors if authors is not None else ["Author A"],
        year=year,
        url=url,
        source="semantic_scholar",
    )


@patch("src.main.run")
def test_main_calls_run_with_defaults(mock_run):
    mock_run.return_value = []
    with patch.object(sys, "argv", ["prog", "--query", "ml"]):
        main()
    mock_run.assert_called_once_with("ml", max_papers=10, sort=None, year="2023-")


@patch("src.main.run")
def test_main_calls_run_with_all_args(mock_run):
    mock_run.return_value = []
    with patch.object(sys, "argv", [
        "prog", "--query", "transformers",
        "--max-papers", "5",
        "--sort", "citationCount:desc",
        "--year", "2020:2023",
    ]):
        main()
    mock_run.assert_called_once_with(
        "transformers", max_papers=5, sort="citationCount:desc", year="2020:2023"
    )


@patch("src.main.run")
def test_main_prints_found_count(mock_run, capsys):
    mock_run.return_value = [_make_paper(), _make_paper()]
    with patch.object(sys, "argv", ["prog", "--query", "ml"]):
        main()
    assert "Found 2 papers" in capsys.readouterr().out


@patch("src.main.run")
def test_main_prints_paper_fields(mock_run, capsys):
    mock_run.return_value = [_make_paper()]
    with patch.object(sys, "argv", ["prog", "--query", "ml"]):
        main()
    out = capsys.readouterr().out
    assert "Test Paper" in out
    assert "Author A" in out
    assert "Short abstract." in out
    assert "https://example.com" in out


@patch("src.main.run")
def test_main_truncates_long_abstract(mock_run, capsys):
    long_abstract = "x" * 201
    mock_run.return_value = [_make_paper(abstract=long_abstract)]
    with patch.object(sys, "argv", ["prog", "--query", "ml"]):
        main()
    out = capsys.readouterr().out
    assert "..." in out
    assert long_abstract not in out


@patch("src.main.run")
def test_main_prints_unknown_for_no_authors(mock_run, capsys):
    mock_run.return_value = [_make_paper(authors=[])]
    with patch.object(sys, "argv", ["prog", "--query", "ml"]):
        main()
    assert "Unknown" in capsys.readouterr().out


@patch("src.main.run")
def test_main_omits_url_line_when_none(mock_run, capsys):
    mock_run.return_value = [_make_paper(url=None)]
    with patch.object(sys, "argv", ["prog", "--query", "ml"]):
        main()
    assert "URL" not in capsys.readouterr().out
