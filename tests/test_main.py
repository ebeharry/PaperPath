import json
import sys
from unittest.mock import patch

import yaml

from src.data_classes import AbstractDraft, ClusterAnalysis, ConferenceMatch, ConferenceMatchReport, DraftReport, GapAnalysisReport, Paper, RelatedWorkDraft
from src.main import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _write_cfg(tmp_path, **fields) -> str:
    cfg = {"query": "ml", **fields}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)


_MOCK_REPORT = GapAnalysisReport(
    input="ml",
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

_MOCK_DRAFT = DraftReport(
    input="ml",
    related_work=RelatedWorkDraft(
        subsections=[],
        full_text="### A Theme\n\nA paragraph.",
    ),
    abstract=AbstractDraft(
        background="Background.",
        prior_work_summary="Prior.",
        gap="Gap.",
        proposed_approach="[FILL IN]",
        expected_contribution="Contribution.",
        full_text="Full abstract paragraph.",
    ),
)


# ---------------------------------------------------------------------------
# search mode
# ---------------------------------------------------------------------------

@patch("src.main.run")
def test_search_mode_calls_run(mock_run, tmp_path):
    mock_run.return_value = []
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    mock_run.assert_called_once_with("ml", max_papers=10, ss_sort=None, arxiv_sort=None, year="2023-")


@patch("src.main.run")
def test_search_mode_empty_results_prints_found_zero(mock_run, tmp_path, capsys):
    mock_run.return_value = []
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert "Found 0 papers" in capsys.readouterr().out


@patch("src.main.run")
def test_search_mode_respects_yaml_fields(mock_run, tmp_path):
    mock_run.return_value = []
    cfg = _write_cfg(tmp_path, mode="search", max_papers=5, year="2020:2023",
                     ss_sort="citationCount:desc", arxiv_sort="submittedDate:desc")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    mock_run.assert_called_once_with(
        "ml", max_papers=5, ss_sort="citationCount:desc",
        arxiv_sort="submittedDate:desc", year="2020:2023",
    )


@patch("src.main.run")
def test_search_mode_prints_found_count(mock_run, tmp_path, capsys):
    mock_run.return_value = [_make_paper(), _make_paper()]
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert "Found 2 papers" in capsys.readouterr().out


@patch("src.main.run")
def test_search_mode_prints_paper_fields(mock_run, tmp_path, capsys):
    mock_run.return_value = [_make_paper()]
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "Test Paper" in out
    assert "Author A" in out
    assert "Short abstract." in out
    assert "https://example.com" in out


@patch("src.main.run")
def test_search_mode_truncates_long_abstract(mock_run, tmp_path, capsys):
    long_abstract = "x" * 201
    mock_run.return_value = [_make_paper(abstract=long_abstract)]
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "..." in out
    assert long_abstract not in out


@patch("src.main.run")
def test_search_mode_prints_unknown_for_no_authors(mock_run, tmp_path, capsys):
    mock_run.return_value = [_make_paper(authors=[])]
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert "Unknown" in capsys.readouterr().out


@patch("src.main.run")
def test_search_mode_omits_url_line_when_none(mock_run, tmp_path, capsys):
    mock_run.return_value = [_make_paper(url=None)]
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert "URL" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# analyse mode
# ---------------------------------------------------------------------------

@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_mode_calls_run_with_analysis(mock_rwa, tmp_path):
    cfg = _write_cfg(tmp_path, mode="analyse")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    mock_rwa.assert_called_once()


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_mode_passes_embed_backend(mock_rwa, tmp_path):
    cfg = _write_cfg(tmp_path, mode="analyse", embed_backend="openai")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwa.call_args[1]["embed_backend"] == "openai"


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_mode_passes_llm_backend(mock_rwa, tmp_path):
    cfg = _write_cfg(tmp_path, mode="analyse", llm_backend="anthropic")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwa.call_args[1]["llm_backend"] == "anthropic"


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_mode_default_backends(mock_rwa, tmp_path):
    cfg = _write_cfg(tmp_path, mode="analyse")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwa.call_args[1]["embed_backend"] == "local"
    assert mock_rwa.call_args[1]["llm_backend"] == "openrouter"


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_mode_prints_gap_report(mock_rwa, tmp_path, capsys):
    cfg = _write_cfg(tmp_path, mode="analyse")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "Gap Analysis" in out
    assert "What Exists" in out
    assert "What Is Missing" in out


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_mode_writes_json_to_output_file(mock_rwa, tmp_path):
    out_file = tmp_path / "report.json"
    cfg = _write_cfg(tmp_path, mode="analyse", output=str(out_file))
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert isinstance(data["papers"], list)
    assert isinstance(data["gap_report"], dict)
    assert data["gap_report"]["input"] == "ml"
    assert isinstance(data["gap_report"]["clusters"], list)
    assert "overall_what_exists" in data["gap_report"]


@patch("src.main.run_with_analysis", return_value=([_make_paper()], _MOCK_REPORT))
def test_analyse_mode_passes_project_description(mock_rwa, tmp_path):
    cfg = _write_cfg(tmp_path, mode="analyse", project_description="Detailed description.")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwa.call_args[1]["project_description"] == "Detailed description."


# ---------------------------------------------------------------------------
# draft mode
# ---------------------------------------------------------------------------

@patch("src.main.run_with_drafts", return_value=([_make_paper()], _MOCK_REPORT, _MOCK_DRAFT))
def test_draft_mode_calls_run_with_drafts(mock_rwd, tmp_path):
    cfg = _write_cfg(tmp_path, mode="draft")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    mock_rwd.assert_called_once()


@patch("src.main.run_with_drafts", return_value=([_make_paper()], _MOCK_REPORT, _MOCK_DRAFT))
def test_draft_mode_is_default_mode(mock_rwd, tmp_path):
    cfg = _write_cfg(tmp_path)  # no mode key
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    mock_rwd.assert_called_once()


@patch("src.main.run_with_drafts", return_value=([_make_paper()], _MOCK_REPORT, _MOCK_DRAFT))
def test_draft_mode_prints_abstract_section(mock_rwd, tmp_path, capsys):
    cfg = _write_cfg(tmp_path, mode="draft")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert "Abstract Draft" in capsys.readouterr().out


@patch("src.main.run_with_drafts", return_value=([_make_paper()], _MOCK_REPORT, _MOCK_DRAFT))
def test_draft_mode_prints_related_work_section(mock_rwd, tmp_path, capsys):
    cfg = _write_cfg(tmp_path, mode="draft")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert "Related Work Draft" in capsys.readouterr().out


@patch("src.main.run_with_drafts", return_value=([_make_paper()], _MOCK_REPORT, _MOCK_DRAFT))
def test_draft_mode_writes_combined_json_to_output(mock_rwd, tmp_path):
    out_file = tmp_path / "draft.json"
    cfg = _write_cfg(tmp_path, mode="draft", output=str(out_file))
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "papers" in data
    assert "gap_report" in data
    assert "draft_report" in data


@patch("src.main.run_with_drafts", return_value=([_make_paper()], _MOCK_REPORT, _MOCK_DRAFT))
def test_draft_mode_default_top_k_is_5(mock_rwd, tmp_path):
    cfg = _write_cfg(tmp_path, mode="draft")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwd.call_args[1]["top_k"] == 5


@patch("src.main.run_with_drafts", return_value=([_make_paper()], _MOCK_REPORT, _MOCK_DRAFT))
def test_draft_mode_passes_top_k_from_yaml(mock_rwd, tmp_path):
    cfg = _write_cfg(tmp_path, mode="draft", top_k=3)
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwd.call_args[1]["top_k"] == 3


@patch("src.main.run_with_drafts", return_value=([_make_paper()], _MOCK_REPORT, _MOCK_DRAFT))
def test_draft_mode_passes_project_description(mock_rwd, tmp_path):
    cfg = _write_cfg(tmp_path, mode="draft", project_description="My project description.")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwd.call_args[1]["project_description"] == "My project description."


# ---------------------------------------------------------------------------
# match mode
# ---------------------------------------------------------------------------

_MOCK_MATCH_REPORT = ConferenceMatchReport(
    input="Full abstract paragraph.",
    matches=[
        ConferenceMatch(
            conference_id="neurips-2030",
            name="Neural Information Processing Systems",
            short_name="NeurIPS",
            similarity=0.95,
            deadline="2030-01-01 23:59:59",
            abstract_deadline="2029-12-15 23:59:59",
            link="https://neurips.cc/",
            subject_areas=["ML"],
        )
    ],
    top_n=10,
)

_MATCH_RETURN = ([_make_paper()], _MOCK_REPORT, _MOCK_DRAFT, _MOCK_MATCH_REPORT)


@patch("src.main.run_with_conference_matching", return_value=_MATCH_RETURN)
def test_match_mode_calls_run_with_conference_matching(mock_rwcm, tmp_path):
    cfg = _write_cfg(tmp_path, mode="match")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    mock_rwcm.assert_called_once()


@patch("src.main.run_with_conference_matching", return_value=_MATCH_RETURN)
def test_match_mode_passes_top_n_from_yaml(mock_rwcm, tmp_path):
    cfg = _write_cfg(tmp_path, mode="match", top_n=7)
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwcm.call_args[1]["top_n"] == 7


@patch("src.main.run_with_conference_matching", return_value=_MATCH_RETURN)
def test_match_mode_prints_conference_heading(mock_rwcm, tmp_path, capsys):
    cfg = _write_cfg(tmp_path, mode="match")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "Conference Matches" in out
    assert "NeurIPS" in out


@patch("src.main.run_with_conference_matching", return_value=_MATCH_RETURN)
def test_match_mode_prints_deadline_and_similarity(mock_rwcm, tmp_path, capsys):
    cfg = _write_cfg(tmp_path, mode="match")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "sim" in out
    assert "deadline" in out


@patch("src.main.run_with_conference_matching", return_value=_MATCH_RETURN)
def test_match_mode_writes_json_to_output(mock_rwcm, tmp_path):
    out_file = tmp_path / "results.json"
    cfg = _write_cfg(tmp_path, mode="match", output=str(out_file))
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "papers" in data
    assert "match_report" in data
    assert data["match_report"]["top_n"] == 10


@patch("src.main.run_with_conference_matching", return_value=_MATCH_RETURN)
def test_match_mode_prints_note_when_no_output(mock_rwcm, tmp_path, capsys):
    cfg = _write_cfg(tmp_path, mode="match")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "no output configured" in out
