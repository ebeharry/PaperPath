import json
import sys
from unittest.mock import patch

import yaml

from src.data_classes import AbstractDraft, ClusterAnalysis, ConferenceMatch, ConferenceMatchReport, DraftReport, GapAnalysisReport, Paper, RelatedWorkDraft
from src.main import main, _papers_json, _print_caveats, _search_caveats


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

_ANALYSIS_RETURN = ([_make_paper()], _MOCK_REPORT)
_DRAFT_RETURN = ([_make_paper()], _MOCK_REPORT, _MOCK_DRAFT)


# ---------------------------------------------------------------------------
# search mode
# ---------------------------------------------------------------------------

@patch("src.main.run")
def test_search_mode_calls_run(mock_run, tmp_path):
    mock_run.return_value = []
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    mock_run.assert_called_once_with("ml", search_limit=10, ss_sort=None, arxiv_sort=None, year="2023-")


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
    cfg = _write_cfg(tmp_path, mode="search", search_limit=5, year="2020:2023",
                     ss_sort="citationCount:desc", arxiv_sort="submittedDate:desc")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    mock_run.assert_called_once_with(
        "ml", search_limit=5, ss_sort="citationCount:desc",
        arxiv_sort="submittedDate:desc", year="2020:2023",
    )


@patch("src.main.make_embedder")
@patch("src.main.rank_papers", side_effect=lambda q, papers, emb: papers)
@patch("src.main.run")
def test_search_mode_prints_found_count(mock_run, mock_rank, mock_embedder, tmp_path, capsys):
    mock_run.return_value = [_make_paper(), _make_paper()]
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert "Found 2 papers" in capsys.readouterr().out


@patch("src.main.make_embedder")
@patch("src.main.rank_papers", side_effect=lambda q, papers, emb: papers)
@patch("src.main.run")
def test_search_mode_prints_paper_fields(mock_run, mock_rank, mock_embedder, tmp_path, capsys):
    mock_run.return_value = [_make_paper()]
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "Test Paper" in out
    assert "Author A" in out
    assert "Short abstract." in out
    assert "https://example.com" in out


@patch("src.main.make_embedder")
@patch("src.main.rank_papers", side_effect=lambda q, papers, emb: papers)
@patch("src.main.run")
def test_search_mode_truncates_long_abstract(mock_run, mock_rank, mock_embedder, tmp_path, capsys):
    long_abstract = "x" * 201
    mock_run.return_value = [_make_paper(abstract=long_abstract)]
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "..." in out
    assert long_abstract not in out


@patch("src.main.make_embedder")
@patch("src.main.rank_papers", side_effect=lambda q, papers, emb: papers)
@patch("src.main.run")
def test_search_mode_prints_unknown_for_no_authors(mock_run, mock_rank, mock_embedder, tmp_path, capsys):
    mock_run.return_value = [_make_paper(authors=[])]
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert "Unknown" in capsys.readouterr().out


@patch("src.main.make_embedder")
@patch("src.main.rank_papers", side_effect=lambda q, papers, emb: papers)
@patch("src.main.run")
def test_search_mode_omits_url_line_when_none(mock_run, mock_rank, mock_embedder, tmp_path, capsys):
    mock_run.return_value = [_make_paper(url=None)]
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert "URL" not in capsys.readouterr().out


def test_papers_json_includes_final_score():
    p = _make_paper()
    p_scored = p.model_copy(update={"final_score": 0.7531})
    result = _papers_json([p_scored])
    assert result[0]["final_score"] == 0.7531


def test_papers_json_final_score_is_null_when_not_ranked():
    result = _papers_json([_make_paper()])
    assert result[0]["final_score"] is None


@patch("src.main.make_embedder")
@patch("src.main.rank_papers",
       side_effect=lambda q, papers, emb: [p.model_copy(update={"final_score": 0.812}) for p in papers])
@patch("src.main.run")
def test_search_mode_saves_final_score_to_json(mock_run, mock_rank, mock_embedder, tmp_path):
    mock_run.return_value = [_make_paper()]
    out_dir = tmp_path / "out"
    cfg = _write_cfg(tmp_path, mode="search", output=str(out_dir))
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    papers = json.loads((out_dir / "papers.json").read_text())
    assert papers[0]["final_score"] == 0.812


# ---------------------------------------------------------------------------
# analyse mode
# ---------------------------------------------------------------------------

@patch("src.main.run_with_analysis", return_value=_ANALYSIS_RETURN)
def test_analyse_mode_calls_run_with_analysis(mock_rwa, tmp_path):
    cfg = _write_cfg(tmp_path, mode="analyse")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    mock_rwa.assert_called_once()


@patch("src.main.run_with_analysis", return_value=_ANALYSIS_RETURN)
def test_analyse_mode_passes_embed_backend(mock_rwa, tmp_path):
    cfg = _write_cfg(tmp_path, mode="analyse", embed_backend="openai")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwa.call_args[1]["embed_backend"] == "openai"


@patch("src.main.run_with_analysis", return_value=_ANALYSIS_RETURN)
def test_analyse_mode_passes_llm_backend(mock_rwa, tmp_path):
    cfg = _write_cfg(tmp_path, mode="analyse", llm_backend="anthropic")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwa.call_args[1]["llm_backend"] == "anthropic"


@patch("src.main.run_with_analysis", return_value=_ANALYSIS_RETURN)
def test_analyse_mode_default_backends(mock_rwa, tmp_path):
    cfg = _write_cfg(tmp_path, mode="analyse")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwa.call_args[1]["embed_backend"] == "local"
    assert mock_rwa.call_args[1]["llm_backend"] == "openrouter"


@patch("src.main.run_with_analysis", return_value=_ANALYSIS_RETURN)
def test_analyse_mode_prints_gap_report(mock_rwa, tmp_path, capsys):
    cfg = _write_cfg(tmp_path, mode="analyse")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "Gap Analysis" in out
    assert "What Exists" in out
    assert "What Is Missing" in out


@patch("src.main.run_with_analysis", return_value=_ANALYSIS_RETURN)
def test_analyse_mode_writes_output_files(mock_rwa, tmp_path):
    out_dir = tmp_path / "results"
    cfg = _write_cfg(tmp_path, mode="analyse", output=str(out_dir))
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    papers = json.loads((out_dir / "papers.json").read_text())
    gap = json.loads((out_dir / "gap_analysis.json").read_text())
    assert isinstance(papers, list)
    assert gap["input"] == "ml"
    assert "clusters" in gap
    assert "overall_what_exists" in gap


@patch("src.main.run_with_analysis", return_value=_ANALYSIS_RETURN)
def test_analyse_mode_passes_project_description(mock_rwa, tmp_path):
    cfg = _write_cfg(tmp_path, mode="analyse", project_description="Detailed description.")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwa.call_args[1]["project_description"] == "Detailed description."


# ---------------------------------------------------------------------------
# draft mode
# ---------------------------------------------------------------------------

@patch("src.main.run_with_drafts", return_value=_DRAFT_RETURN)
def test_draft_mode_calls_run_with_drafts(mock_rwd, tmp_path):
    cfg = _write_cfg(tmp_path, mode="draft")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    mock_rwd.assert_called_once()


@patch("src.main.run_with_drafts", return_value=_DRAFT_RETURN)
def test_draft_mode_is_default_mode(mock_rwd, tmp_path):
    cfg = _write_cfg(tmp_path)  # no mode key
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    mock_rwd.assert_called_once()


@patch("src.main.run_with_drafts", return_value=_DRAFT_RETURN)
def test_draft_mode_prints_complete(mock_rwd, tmp_path, capsys):
    cfg = _write_cfg(tmp_path, mode="draft")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert "Draft complete" in capsys.readouterr().out


@patch("src.main.run_with_drafts", return_value=_DRAFT_RETURN)
def test_draft_mode_writes_output_files(mock_rwd, tmp_path):
    out_dir = tmp_path / "draft_out"
    cfg = _write_cfg(tmp_path, mode="draft", output=str(out_dir))
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    papers = json.loads((out_dir / "papers.json").read_text())
    gap = json.loads((out_dir / "gap_analysis.json").read_text())
    draft = json.loads((out_dir / "draft.json").read_text())
    assert isinstance(papers, list)
    assert "clusters" in gap
    assert "related_work" in draft
    assert "abstract" in draft


@patch("src.main.run_with_drafts", return_value=_DRAFT_RETURN)
def test_draft_mode_default_top_k_is_5(mock_rwd, tmp_path):
    cfg = _write_cfg(tmp_path, mode="draft")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwd.call_args[1]["top_k"] == 5


@patch("src.main.run_with_drafts", return_value=_DRAFT_RETURN)
def test_draft_mode_passes_top_k_from_yaml(mock_rwd, tmp_path):
    cfg = _write_cfg(tmp_path, mode="draft", top_k=3)
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    assert mock_rwd.call_args[1]["top_k"] == 3


@patch("src.main.run_with_drafts", return_value=_DRAFT_RETURN)
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
def test_match_mode_writes_output_files(mock_rwcm, tmp_path):
    out_dir = tmp_path / "match_out"
    cfg = _write_cfg(tmp_path, mode="match", output=str(out_dir))
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    papers = json.loads((out_dir / "papers.json").read_text())
    conferences = json.loads((out_dir / "conferences.json").read_text())
    assert isinstance(papers, list)
    assert conferences["top_n"] == 10


@patch("src.main.run_with_conference_matching", return_value=_MATCH_RETURN)
def test_match_mode_prints_note_when_no_output(mock_rwcm, tmp_path, capsys):
    cfg = _write_cfg(tmp_path, mode="match")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "no output configured" in out


# ---------------------------------------------------------------------------
# _print_caveats
# ---------------------------------------------------------------------------

def test_print_caveats_outputs_formatted_list(capsys):
    _print_caveats(["Only arXiv papers found; venue fit is low confidence.", "3 papers have no abstract."])
    out = capsys.readouterr().out
    assert "Quality Notes" in out
    assert "- Only arXiv" in out
    assert "- 3 papers" in out


def test_print_caveats_silent_when_empty(capsys):
    _print_caveats([])
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# _search_caveats
# ---------------------------------------------------------------------------

def test_search_caveats_all_arxiv():
    papers = [_make_paper(year=2023), _make_paper(year=2022)]
    for p in papers:
        object.__setattr__(p, "source", "arxiv")
    # Rebuild properly
    from src.data_classes import Paper as _Paper
    arxiv_papers = [_Paper(paper_id=f"a{i}", title="T", abstract="A", authors=[], year=2023, url=None, source="arxiv") for i in range(2)]
    caveats = _search_caveats(arxiv_papers)
    assert any("arXiv" in c for c in caveats)


def test_search_caveats_mixed_sources_no_source_caveat():
    from src.data_classes import Paper as _Paper
    papers = [
        _Paper(paper_id="ss", title="T", abstract="A", authors=[], year=2023, url=None, source="semantic_scholar"),
        _Paper(paper_id="ax", title="T", abstract="A", authors=[], year=2023, url=None, source="arxiv"),
    ]
    caveats = _search_caveats(papers)
    assert not any("arXiv" in c or "Semantic Scholar" in c for c in caveats)


def test_search_caveats_missing_year():
    from src.data_classes import Paper as _Paper
    papers = [_Paper(paper_id="p1", title="T", abstract="A", authors=[], year=None, url=None, source="arxiv")]
    caveats = _search_caveats(papers)
    assert any("year" in c for c in caveats)


def test_search_caveats_empty_list():
    assert _search_caveats([]) == []


@patch("src.main.make_embedder")
@patch("src.main.rank_papers", side_effect=lambda q, papers, emb: papers)
@patch("src.main.run")
def test_search_mode_prints_quality_notes_when_caveats(mock_run, mock_rank, mock_embedder, tmp_path, capsys):
    from src.data_classes import Paper as _Paper
    # all-arxiv → should trigger caveat
    mock_run.return_value = [_Paper(paper_id="a1", title="T", abstract="A", authors=[], year=2023, url=None, source="arxiv")]
    cfg = _write_cfg(tmp_path, mode="search")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "Quality Notes" in out
    assert "arXiv" in out


@patch("src.main.run_with_analysis", return_value=_ANALYSIS_RETURN)
def test_analyse_mode_prints_caveats_when_present(mock_rwa, tmp_path, capsys):
    # _MOCK_REPORT has caveats=[], so Quality Notes should NOT appear
    cfg = _write_cfg(tmp_path, mode="analyse")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "Quality Notes" not in out


@patch("src.main.run_with_analysis", return_value=(
    [_make_paper()],
    GapAnalysisReport(
        input="ml",
        clusters=[ClusterAnalysis(cluster_id=0, paper_ids=["p1"], what_exists="A", what_is_contested="B", what_is_missing="C")],
        overall_what_exists="A",
        overall_what_is_contested="B",
        overall_what_is_missing="C",
        caveats=["Only arXiv papers found; venue fit is low confidence."],
    )
))
def test_analyse_mode_prints_non_empty_caveats(mock_rwa, tmp_path, capsys):
    cfg = _write_cfg(tmp_path, mode="analyse")
    with patch.object(sys, "argv", ["prog", cfg]):
        main()
    out = capsys.readouterr().out
    assert "Quality Notes" in out
    assert "arXiv" in out
