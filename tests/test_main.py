import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.literature_review.data_classes import ClusterAnalysis, GapAnalysisReport, Paper
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


_MOCK_REPORT = GapAnalysisReport(
    query="ml",
    clusters=[
        ClusterAnalysis(
            cluster_id=0,
            paper_ids=["p1"],
            what_exists="Existing work.",
            what_is_contested="Contested area.",
            what_is_missing="Missing research.",
        )
    ],
    overall_what_exists="Overall existing.",
    overall_what_is_contested="Overall contested.",
    overall_what_is_missing="Overall missing.",
)


# ---------------------------------------------------------------------------
# Phase 1 (baseline) tests — fixed from stale --sort / sort= usage
# ---------------------------------------------------------------------------

@patch("src.main.run")
def test_main_calls_run_with_defaults(mock_run):
    mock_run.return_value = []
    with patch.object(sys, "argv", ["prog", "--query", "ml"]):
        main()
    mock_run.assert_called_once_with("ml", max_papers=10, ss_sort=None, arxiv_sort=None, year="2023-")


@patch("src.main.run")
def test_main_calls_run_with_all_args(mock_run):
    mock_run.return_value = []
    with patch.object(sys, "argv", [
        "prog", "--query", "transformers",
        "--max-papers", "5",
        "--ss-sort", "citationCount:desc",
        "--arxiv-sort", "submittedDate:desc",
        "--year", "2020:2023",
    ]):
        main()
    mock_run.assert_called_once_with(
        "transformers",
        max_papers=5,
        ss_sort="citationCount:desc",
        arxiv_sort="submittedDate:desc",
        year="2020:2023",
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


@patch("src.main.run")
def test_main_accepts_ss_sort_choice(mock_run):
    mock_run.return_value = []
    with patch.object(sys, "argv", ["prog", "--query", "ml", "--ss-sort", "citationCount:desc"]):
        main()
    mock_run.assert_called_once_with("ml", max_papers=10, ss_sort="citationCount:desc", arxiv_sort=None, year="2023-")


@patch("src.main.run")
def test_main_accepts_arxiv_sort_choice(mock_run):
    mock_run.return_value = []
    with patch.object(sys, "argv", ["prog", "--query", "ml", "--arxiv-sort", "submittedDate:desc"]):
        main()
    mock_run.assert_called_once_with("ml", max_papers=10, ss_sort=None, arxiv_sort="submittedDate:desc", year="2023-")


# ---------------------------------------------------------------------------
# Phase 2 (--analyse) tests
# ---------------------------------------------------------------------------

@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_flag_calls_run_with_analysis(mock_rwa):
    with patch.object(sys, "argv", ["prog", "--query", "ml", "--analyse"]):
        main()
    mock_rwa.assert_called_once()


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_passes_embed_backend(mock_rwa):
    with patch.object(sys, "argv", ["prog", "--query", "ml", "--analyse", "--embed-backend", "openai"]):
        main()
    call_kwargs = mock_rwa.call_args[1]
    assert call_kwargs["embed_backend"] == "openai"


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_passes_llm_backend(mock_rwa):
    with patch.object(sys, "argv", ["prog", "--query", "ml", "--analyse", "--llm-backend", "anthropic"]):
        main()
    call_kwargs = mock_rwa.call_args[1]
    assert call_kwargs["llm_backend"] == "anthropic"


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_prints_gap_report_summary(mock_rwa, capsys):
    with patch.object(sys, "argv", ["prog", "--query", "ml", "--analyse"]):
        main()
    out = capsys.readouterr().out
    assert "Gap Analysis" in out
    assert "What Exists" in out
    assert "What Is Missing" in out


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_writes_json_to_output_file(mock_rwa, tmp_path):
    out_file = tmp_path / "report.json"
    with patch.object(sys, "argv", ["prog", "--query", "ml", "--analyse", "--output", str(out_file)]):
        main()
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["query"] == "ml"
    assert "clusters" in data


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_prints_json_to_stdout_when_no_output_file(mock_rwa, capsys):
    with patch.object(sys, "argv", ["prog", "--query", "ml", "--analyse"]):
        main()
    out = capsys.readouterr().out
    # JSON should be present somewhere in stdout
    assert '"query"' in out


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_default_embed_backend_is_local(mock_rwa):
    with patch.object(sys, "argv", ["prog", "--query", "ml", "--analyse"]):
        main()
    call_kwargs = mock_rwa.call_args[1]
    assert call_kwargs["embed_backend"] == "local"


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_default_llm_backend_is_openai(mock_rwa):
    with patch.object(sys, "argv", ["prog", "--query", "ml", "--analyse"]):
        main()
    call_kwargs = mock_rwa.call_args[1]
    assert call_kwargs["llm_backend"] == "openai"
