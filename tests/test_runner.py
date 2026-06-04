import pytest
from unittest.mock import MagicMock, patch

import requests

from src.data_classes import AbstractDraft, ClusterAnalysis, ConferenceMatch, ConferenceMatchReport, DraftReport, GapAnalysisReport, Paper, RelatedWorkDraft
from src.runner import run, run_with_analysis, run_with_conference_matching, run_with_drafts, _analysis_caveats


_SS_PAPERS = [
    Paper(paper_id="p1", title="SS Paper", abstract="Abstract.", authors=["Author A"], year=2023, url=None, source="semantic_scholar")
]

_ARXIV_PAPERS = [
    Paper(paper_id="a1", title="ArXiv Paper", abstract="", authors=[], year=2023, url=None, source="arxiv")
]

_RANK_SIDE_EFFECT = lambda q, papers, emb: papers


@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_delegates_to_semantic_scholar(mock_ss, mock_arxiv):
    run("neural networks")
    mock_ss.assert_called_once_with("neural networks", limit=20, sort=None, year=None)


@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_calls_search_arxiv(mock_ss, mock_arxiv):
    run("neural networks")
    mock_arxiv.assert_called_once_with("neural networks", limit=20, sort=None, year=None)


@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_combines_results(mock_ss, mock_arxiv):
    result = run("neural networks")
    assert result == _SS_PAPERS + _ARXIV_PAPERS


@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_routes_sorts_independently(mock_ss, mock_arxiv):
    run("neural networks", ss_sort="citationCount:desc", arxiv_sort="submittedDate:asc")
    mock_ss.assert_called_once_with("neural networks", limit=20, sort="citationCount:desc", year=None)
    mock_arxiv.assert_called_once_with("neural networks", limit=20, sort="submittedDate:asc", year=None)


@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_passes_none_sort_to_both(mock_ss, mock_arxiv):
    run("neural networks")
    mock_ss.assert_called_once_with("neural networks", limit=20, sort=None, year=None)
    mock_arxiv.assert_called_once_with("neural networks", limit=20, sort=None, year=None)


@patch("src.runner.search_arxiv")
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_deduplicates_by_title(mock_ss, mock_arxiv):
    dup = Paper(paper_id="a2", title="SS Paper", abstract="", authors=[], year=2023, url=None, source="arxiv")
    mock_arxiv.return_value = [dup]
    result = run("neural networks")
    assert len(result) == 1
    assert result[0].source == "semantic_scholar"


@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_forwards_year(mock_ss, mock_arxiv):
    run("neural networks", year="2020:2023")
    mock_ss.assert_called_once_with("neural networks", limit=20, sort=None, year="2020:2023")
    mock_arxiv.assert_called_once_with("neural networks", limit=20, sort=None, year="2020:2023")


@patch("src.runner.search_arxiv", side_effect=requests.ReadTimeout("timed out"))
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_continues_when_arxiv_times_out(mock_ss, mock_arxiv, capsys):
    result = run("neural networks")
    assert result == _SS_PAPERS
    assert "arxiv unavailable" in capsys.readouterr().err


@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", side_effect=requests.ReadTimeout("timed out"))
def test_run_continues_when_ss_times_out(mock_ss, mock_arxiv, capsys):
    result = run("neural networks")
    assert result == _ARXIV_PAPERS
    assert "semantic scholar unavailable" in capsys.readouterr().err


@patch("src.runner.search_arxiv", side_effect=requests.ReadTimeout("timed out"))
@patch("src.runner.search_semantic_scholar", side_effect=requests.ReadTimeout("timed out"))
def test_run_raises_when_both_sources_fail(mock_ss, mock_arxiv):
    with pytest.raises(requests.ReadTimeout):
        run("neural networks")


# ---------------------------------------------------------------------------
# run_with_analysis
# ---------------------------------------------------------------------------

_MOCK_REPORT = GapAnalysisReport(
    input="neural networks",
    clusters=[ClusterAnalysis(cluster_id=0, paper_ids=["p1"], what_exists="A", what_is_contested="B", what_is_missing="C")],
    overall_what_exists="A",
    overall_what_is_contested="B",
    overall_what_is_missing="C",
)


@patch("src.runner.build_gap_report", return_value=_MOCK_REPORT)
@patch("src.runner.make_llm_client")
@patch("src.runner.cluster_papers", return_value={0: _SS_PAPERS})
@patch("src.runner.embed_papers", return_value={"p1": [0.1, 0.2]})
@patch("src.runner.rank_papers", side_effect=_RANK_SIDE_EFFECT)
@patch("src.runner.make_embedder")
@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_with_analysis_returns_papers_and_report(
    mock_ss, mock_arxiv, mock_embedder, mock_rank, mock_embed_papers, mock_cluster, mock_llm, mock_report
):
    papers, report = run_with_analysis("neural networks")
    assert isinstance(papers, list)
    assert isinstance(report, GapAnalysisReport)


@patch("src.runner.build_gap_report", return_value=_MOCK_REPORT)
@patch("src.runner.make_llm_client")
@patch("src.runner.cluster_papers", return_value={0: _SS_PAPERS})
@patch("src.runner.embed_papers", return_value={"p1": [0.1, 0.2]})
@patch("src.runner.rank_papers", side_effect=_RANK_SIDE_EFFECT)
@patch("src.runner.make_embedder")
@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_with_analysis_passes_backends(
    mock_ss, mock_arxiv, mock_embedder, mock_rank, mock_embed_papers, mock_cluster, mock_llm, mock_report
):
    run_with_analysis("neural networks", embed_backend="openai", llm_backend="anthropic")
    mock_embedder.assert_called_once_with("openai")
    mock_llm.assert_called_once_with("anthropic")


# ---------------------------------------------------------------------------
# run_with_drafts
# ---------------------------------------------------------------------------

_MOCK_DRAFT = DraftReport(
    input="neural networks",
    related_work=RelatedWorkDraft(subsections=[], full_text=""),
    abstract=AbstractDraft(
        background="B",
        prior_work_summary="P",
        gap="G",
        proposed_approach="[FILL IN]",
        expected_contribution="E",
        full_text="Full abstract paragraph.",
    ),
)


@patch("src.runner.build_draft_report", return_value=_MOCK_DRAFT)
@patch("src.runner.build_gap_report", return_value=_MOCK_REPORT)
@patch("src.runner.make_llm_client")
@patch("src.runner.cluster_papers", return_value={0: _SS_PAPERS})
@patch("src.runner.embed_papers", return_value={"p1": [0.1, 0.2]})
@patch("src.runner.rank_papers", side_effect=_RANK_SIDE_EFFECT)
@patch("src.runner.make_embedder")
@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_with_drafts_returns_three_tuple(
    mock_ss, mock_arxiv, mock_embedder, mock_rank, mock_embed_papers, mock_cluster, mock_llm, mock_report, mock_draft
):
    result = run_with_drafts("neural networks")
    assert len(result) == 3
    papers, gap_report, draft_report = result
    assert isinstance(papers, list)
    assert isinstance(gap_report, GapAnalysisReport)
    assert isinstance(draft_report, DraftReport)


@patch("src.runner.build_draft_report", return_value=_MOCK_DRAFT)
@patch("src.runner.build_gap_report", return_value=_MOCK_REPORT)
@patch("src.runner.make_llm_client")
@patch("src.runner.cluster_papers", return_value={0: _SS_PAPERS})
@patch("src.runner.embed_papers", return_value={"p1": [0.1, 0.2]})
@patch("src.runner.rank_papers", side_effect=_RANK_SIDE_EFFECT)
@patch("src.runner.make_embedder")
@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_with_drafts_calls_make_embedder_once(
    mock_ss, mock_arxiv, mock_embedder, mock_rank, mock_embed_papers, mock_cluster, mock_llm, mock_report, mock_draft
):
    run_with_drafts("neural networks")
    mock_embedder.assert_called_once()


@patch("src.runner.build_draft_report", return_value=_MOCK_DRAFT)
@patch("src.runner.build_gap_report", return_value=_MOCK_REPORT)
@patch("src.runner.make_llm_client")
@patch("src.runner.cluster_papers", return_value={0: _SS_PAPERS})
@patch("src.runner.embed_papers", return_value={"p1": [0.1, 0.2]})
@patch("src.runner.rank_papers", side_effect=_RANK_SIDE_EFFECT)
@patch("src.runner.make_embedder")
@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_with_drafts_passes_top_k(
    mock_ss, mock_arxiv, mock_embedder, mock_rank, mock_embed_papers, mock_cluster, mock_llm, mock_report, mock_draft
):
    run_with_drafts("neural networks", top_k=3)
    call_kwargs = mock_draft.call_args[1]
    assert call_kwargs["top_k"] == 3


# ---------------------------------------------------------------------------
# run_with_conference_matching
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
            subject_areas=["ML"],
        )
    ],
    top_n=10,
)


def _full_patch_for_matching(mock_match_report=None):
    return [
        patch("src.runner.build_conference_match_report", return_value=mock_match_report or _MOCK_MATCH_REPORT),
        patch("src.runner.filter_future_conferences", return_value=[]),
        patch("src.runner.fetch_conferences", return_value=[]),
        patch("src.runner.build_draft_report", return_value=_MOCK_DRAFT),
        patch("src.runner.build_gap_report", return_value=_MOCK_REPORT),
        patch("src.runner.make_llm_client"),
        patch("src.runner.cluster_papers", return_value={0: _SS_PAPERS}),
        patch("src.runner.embed_papers", return_value={"p1": [0.1, 0.2]}),
        patch("src.runner.rank_papers", side_effect=_RANK_SIDE_EFFECT),
        patch("src.runner.make_embedder"),
        patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS),
        patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS),
    ]


def test_run_with_conference_matching_returns_four_tuple():
    from contextlib import ExitStack
    with ExitStack() as stack:
        for p in _full_patch_for_matching():
            stack.enter_context(p)
        result = run_with_conference_matching("neural networks")
    assert len(result) == 4
    papers, gap_report, draft_report, match_report = result
    assert isinstance(papers, list)
    assert isinstance(gap_report, GapAnalysisReport)
    assert isinstance(draft_report, DraftReport)
    assert isinstance(match_report, ConferenceMatchReport)


def test_run_with_conference_matching_passes_top_n():
    from contextlib import ExitStack
    with ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in _full_patch_for_matching()]
        run_with_conference_matching("neural networks", top_n=5)
    mock_match = mocks[0]
    mock_match.assert_called_once()
    assert mock_match.call_args[0][3] == 5


def test_run_with_conference_matching_uses_abstract_full_text():
    from contextlib import ExitStack
    with ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in _full_patch_for_matching()]
        run_with_conference_matching("neural networks")
    abstract_arg = mocks[0].call_args[0][0]
    assert abstract_arg == "Full abstract paragraph."


def _empty_abstract_draft():
    return DraftReport(
        input="neural networks",
        related_work=RelatedWorkDraft(subsections=[], full_text=""),
        abstract=AbstractDraft(
            background="", prior_work_summary="", gap="", proposed_approach="", expected_contribution="",
            full_text="",
        ),
    )


def test_run_with_conference_matching_falls_back_to_project_description():
    from contextlib import ExitStack
    base_patches = _full_patch_for_matching()
    with ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in base_patches]
        mocks[3].return_value = _empty_abstract_draft()
        run_with_conference_matching("neural networks", project_description="My project desc")
    abstract_arg = mocks[0].call_args[0][0]
    assert abstract_arg == "My project desc"


def test_run_with_conference_matching_falls_back_to_query():
    from contextlib import ExitStack
    base_patches = _full_patch_for_matching()
    with ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in base_patches]
        mocks[3].return_value = _empty_abstract_draft()
        run_with_conference_matching("neural networks", project_description=None)
    abstract_arg = mocks[0].call_args[0][0]
    assert abstract_arg == "neural networks"


# ---------------------------------------------------------------------------
# _analysis_caveats
# ---------------------------------------------------------------------------

def _make_paper(paper_id: str, source: str = "semantic_scholar", abstract: str = "Abstract.") -> Paper:
    return Paper(paper_id=paper_id, title="T", abstract=abstract, authors=[], year=2023, url=None, source=source)


def test_analysis_caveats_all_arxiv():
    papers = [_make_paper("a1", "arxiv"), _make_paper("a2", "arxiv")]
    caveats = _analysis_caveats(papers, {0: papers})
    assert any("arXiv" in c for c in caveats)
    assert any("venue fit" in c for c in caveats)


def test_analysis_caveats_all_ss():
    papers = [_make_paper("p1"), _make_paper("p2")]
    caveats = _analysis_caveats(papers, {0: papers[:1], 1: papers[1:]})
    assert any("Semantic Scholar" in c for c in caveats)


def test_analysis_caveats_mixed_sources_no_source_caveat():
    papers = [_make_paper("p1", "semantic_scholar"), _make_paper("a1", "arxiv")]
    caveats = _analysis_caveats(papers, {0: papers[:1], 1: papers[1:]})
    assert not any("arXiv" in c or "Semantic Scholar" in c for c in caveats)


def test_analysis_caveats_single_cluster():
    papers = [_make_paper("p1"), _make_paper("p2")]
    caveats = _analysis_caveats(papers, {0: papers})
    assert any("single cluster" in c.lower() or "one cluster" in c.lower() for c in caveats)


def test_analysis_caveats_missing_abstracts():
    papers = [_make_paper("p1", abstract=""), _make_paper("p2", abstract="Has content.")]
    caveats = _analysis_caveats(papers, {0: papers[:1], 1: papers[1:]})
    assert any("no abstract" in c for c in caveats)


def test_analysis_caveats_all_singleton_clusters_returns_caveat():
    papers = [_make_paper("p1", "semantic_scholar"), _make_paper("a1", "arxiv")]
    clusters = {0: [papers[0]], 1: [papers[1]]}
    caveats = _analysis_caveats(papers, clusters)
    assert any("cluster" in c.lower() for c in caveats)


@patch("src.runner.build_gap_report", return_value=GapAnalysisReport(
    input="q", clusters=[ClusterAnalysis(cluster_id=0, paper_ids=["p1"], what_exists="A", what_is_contested="B", what_is_missing="C")],
    overall_what_exists="A", overall_what_is_contested="B", overall_what_is_missing="C",
))
@patch("src.runner.make_llm_client")
@patch("src.runner.cluster_papers", return_value={0: _SS_PAPERS})
@patch("src.runner.embed_papers", return_value={"p1": [0.1, 0.2]})
@patch("src.runner.rank_papers", side_effect=lambda q, papers, emb: papers)
@patch("src.runner.make_embedder")
@patch("src.runner.search_arxiv", return_value=_ARXIV_PAPERS)
@patch("src.runner.search_semantic_scholar", return_value=_SS_PAPERS)
def test_run_with_analysis_report_has_caveats(
    mock_ss, mock_arxiv, mock_embedder, mock_rank, mock_embed, mock_cluster, mock_llm, mock_report
):
    _, report = run_with_analysis("q")
    assert isinstance(report.caveats, list)
    # single cluster → caveat expected
    assert any("cluster" in c.lower() for c in report.caveats)
