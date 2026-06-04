from __future__ import annotations
import json

import numpy as np
import pytest

from src.data_classes import (
    AbstractDraft,
    CitationStatement,
    ClusterAnalysis,
    DraftReport,
    GapAnalysisReport,
    Paper,
    RelatedWorkSubsection,
)
from src.drafting.draft_writer import (
    _build_abstract_prompt,
    _build_citation_key_map,
    _build_related_work_prompt,
    _draft_caveats,
    _extract_citation_statements,
    _format_citation_key,
    _format_paper_snippet,
    _parse_draft_response,
    build_draft_report,
    draft_abstract,
    draft_related_work_subsection,
    retrieve_top_k,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paper(
    paper_id: str = "p1",
    title: str = "Title",
    abstract: str = "Abstract text.",
    authors: list[str] | None = None,
    year: int | None = 2023,
) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=authors if authors is not None else ["Alice Smith", "Bob Jones"],
        year=year,
        url=None,
        source="semantic_scholar",
    )


def _cluster(cluster_id: int = 0, paper_ids: list[str] | None = None) -> ClusterAnalysis:
    return ClusterAnalysis(
        cluster_id=cluster_id,
        paper_ids=paper_ids or ["p1"],
        what_exists="Prior work exists.",
        what_is_contested="Contested area.",
        what_is_missing="Missing research.",
    )


def _gap_report(clusters: list[ClusterAnalysis] | None = None) -> GapAnalysisReport:
    return GapAnalysisReport(
        input="test query",
        clusters=clusters or [_cluster()],
        overall_what_exists="Overall exists.",
        overall_what_is_contested="Overall contested.",
        overall_what_is_missing="Overall missing.",
    )


class _MockLLMClient:
    def __init__(self, response: str | None = None):
        self._response = response or json.dumps({
            "theme": "Test Theme",
            "paragraph": "Para [Smith et al., 2023].",
            "citations": [{"key": "[Smith et al., 2023]", "snippet": "Key finding from abstract."}],
        })

    def complete(self, prompt: str) -> str:
        return self._response


class _AbstractLLMClient:
    def complete(self, prompt: str) -> str:
        return json.dumps(
            {
                "background": "Background text.",
                "prior_work_summary": "Prior work summary.",
                "gap": "Gap text.",
                "proposed_approach": "[FILL IN: describe the approach your paper proposes to address the gap]",
                "expected_contribution": "Expected contribution.",
                "full_text": "Full abstract paragraph.",
            }
        )


class _MalformedLLMClient:
    def complete(self, prompt: str) -> str:
        return "This is definitely not JSON."


class _MockEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


# ---------------------------------------------------------------------------
# retrieve_top_k
# ---------------------------------------------------------------------------

def test_retrieve_top_k_returns_k_results():
    embeddings = {"p1": [1.0, 0.0], "p2": [0.0, 1.0], "p3": [0.7, 0.7]}
    result = retrieve_top_k([1.0, 0.0], embeddings, ["p1", "p2", "p3"], k=2)
    assert len(result) == 2


def test_retrieve_top_k_ranks_by_cosine_similarity():
    # p1 is identical to query; p2 is orthogonal
    embeddings = {"p1": [1.0, 0.0], "p2": [0.0, 1.0]}
    result = retrieve_top_k([1.0, 0.0], embeddings, ["p1", "p2"], k=2)
    assert result[0] == "p1"


def test_retrieve_top_k_returns_fewer_when_less_than_k_candidates():
    embeddings = {"p1": [1.0, 0.0]}
    result = retrieve_top_k([1.0, 0.0], embeddings, ["p1"], k=5)
    assert result == ["p1"]


def test_retrieve_top_k_empty_candidate_ids_returns_empty():
    embeddings = {"p1": [1.0, 0.0]}
    result = retrieve_top_k([1.0, 0.0], embeddings, [], k=5)
    assert result == []


def test_retrieve_top_k_candidate_not_in_embeddings_skipped():
    embeddings = {"p1": [1.0, 0.0]}
    result = retrieve_top_k([1.0, 0.0], embeddings, ["p1", "missing"], k=2)
    assert "missing" not in result
    assert "p1" in result


def test_retrieve_top_k_normalizes_unnormalized_input():
    # unnormalized vectors — ranking should still be correct
    embeddings = {"p1": [10.0, 0.0], "p2": [0.0, 10.0]}
    result = retrieve_top_k([5.0, 0.0], embeddings, ["p1", "p2"], k=2)
    assert result[0] == "p1"


# ---------------------------------------------------------------------------
# _format_citation_key
# ---------------------------------------------------------------------------

def test_format_citation_key_multiple_authors():
    p = _paper(authors=["Alice Smith", "Bob Jones"], year=2023)
    assert _format_citation_key(p) == "[Smith et al., 2023]"


def test_format_citation_key_single_author():
    p = _paper(authors=["Alice Smith"], year=2021)
    assert _format_citation_key(p) == "[Smith, 2021]"


def test_format_citation_key_no_authors():
    p = _paper(authors=[], year=2022)
    assert _format_citation_key(p) == "[Unknown, 2022]"


def test_format_citation_key_missing_year():
    p = _paper(authors=["Alice Smith"], year=None)
    assert _format_citation_key(p) == "[Smith, n.d.]"


def test_format_citation_key_single_word_author():
    p = _paper(authors=["Plato"], year=2020)
    assert _format_citation_key(p) == "[Plato, 2020]"


# ---------------------------------------------------------------------------
# _format_paper_snippet
# ---------------------------------------------------------------------------

def test_format_paper_snippet_includes_title():
    p = _paper(title="My Paper", abstract="Some abstract.")
    snippet = _format_paper_snippet(p)
    assert "My Paper" in snippet


def test_format_paper_snippet_truncates_abstract():
    long_abstract = "x" * 500
    p = _paper(abstract=long_abstract)
    snippet = _format_paper_snippet(p)
    assert len(snippet) < len(long_abstract) + 100


def test_format_paper_snippet_handles_empty_abstract():
    p = _paper(abstract="")
    snippet = _format_paper_snippet(p)
    assert ":" not in snippet or "Title" in snippet.split(":")[0]


def test_format_paper_snippet_includes_citation_key():
    p = _paper(authors=["Alice Smith"], year=2023, abstract="A.")
    snippet = _format_paper_snippet(p)
    assert "[Smith, 2023]" in snippet


# ---------------------------------------------------------------------------
# _build_related_work_prompt
# ---------------------------------------------------------------------------

def test_build_related_work_prompt_contains_what_is_missing():
    cluster = _cluster()
    prompt = _build_related_work_prompt(cluster, [_paper()])
    assert "Missing research." in prompt


def test_build_related_work_prompt_contains_what_is_contested():
    cluster = _cluster()
    prompt = _build_related_work_prompt(cluster, [_paper()])
    assert "Contested area." in prompt


def test_build_related_work_prompt_contains_paper_snippets():
    cluster = _cluster()
    prompt = _build_related_work_prompt(cluster, [_paper(title="Unique Title XYZ")])
    assert "Unique Title XYZ" in prompt


def test_build_related_work_prompt_requests_theme_paragraph_citations_keys():
    cluster = _cluster()
    prompt = _build_related_work_prompt(cluster, [_paper()])
    assert '"theme"' in prompt
    assert '"paragraph"' in prompt
    assert '"citations"' in prompt


def test_build_related_work_prompt_no_hallucination_instruction():
    cluster = _cluster()
    prompt = _build_related_work_prompt(cluster, [_paper()])
    assert "NOT invent" in prompt


# ---------------------------------------------------------------------------
# _build_abstract_prompt
# ---------------------------------------------------------------------------

def test_build_abstract_prompt_contains_overall_what_is_missing():
    report = _gap_report()
    prompt = _build_abstract_prompt(report)
    assert "Overall missing." in prompt


def test_build_abstract_prompt_contains_overall_what_exists():
    report = _gap_report()
    prompt = _build_abstract_prompt(report)
    assert "Overall exists." in prompt


def test_build_abstract_prompt_requests_six_json_keys():
    report = _gap_report()
    prompt = _build_abstract_prompt(report)
    for key in ["background", "prior_work_summary", "gap", "proposed_approach", "expected_contribution", "full_text"]:
        assert key in prompt


def test_build_abstract_prompt_uses_project_description_when_present():
    report = _gap_report()
    prompt = _build_abstract_prompt(report)
    assert "test query" in prompt
    assert "FILL IN" not in prompt


def test_build_abstract_prompt_placeholder_when_input_empty():
    report = GapAnalysisReport(
        input="",
        clusters=[],
        overall_what_exists="",
        overall_what_is_contested="",
        overall_what_is_missing="",
    )
    prompt = _build_abstract_prompt(report)
    assert "FILL IN" in prompt


# ---------------------------------------------------------------------------
# _parse_draft_response
# ---------------------------------------------------------------------------

def test_parse_draft_response_valid_json():
    response = json.dumps({"theme": "T", "paragraph": "P"})
    result = _parse_draft_response(response)
    assert result["theme"] == "T"
    assert result["paragraph"] == "P"


def test_parse_draft_response_json_in_code_fence():
    response = '```json\n{"theme": "T", "paragraph": "P"}\n```'
    result = _parse_draft_response(response)
    assert result["theme"] == "T"


def test_parse_draft_response_malformed_returns_raw():
    response = "not json at all"
    result = _parse_draft_response(response)
    assert "_raw" in result
    assert result["_raw"] == response


def test_parse_draft_response_partial_json_returns_available_keys():
    response = '{"theme": "T"}'
    result = _parse_draft_response(response)
    assert result.get("theme") == "T"


# ---------------------------------------------------------------------------
# draft_related_work_subsection
# ---------------------------------------------------------------------------

def test_draft_related_work_subsection_returns_correct_type():
    p = _paper("p1")
    cluster = _cluster(paper_ids=["p1"])
    embeddings = {"p1": [1.0, 0.0]}
    result = draft_related_work_subsection(cluster, {"p1": p}, embeddings, _MockEmbedder(), _MockLLMClient())
    assert isinstance(result, RelatedWorkSubsection)


def test_draft_related_work_subsection_cluster_id_preserved():
    p = _paper("p1")
    cluster = _cluster(cluster_id=3, paper_ids=["p1"])
    embeddings = {"p1": [1.0, 0.0]}
    result = draft_related_work_subsection(cluster, {"p1": p}, embeddings, _MockEmbedder(), _MockLLMClient())
    assert result.cluster_id == 3


def test_draft_related_work_subsection_cited_paper_ids_from_retrieval():
    p = _paper("p1")
    cluster = _cluster(paper_ids=["p1"])
    embeddings = {"p1": [1.0, 0.0]}
    result = draft_related_work_subsection(cluster, {"p1": p}, embeddings, _MockEmbedder(), _MockLLMClient())
    assert "p1" in result.cited_paper_ids


def test_draft_related_work_subsection_handles_malformed_llm_response():
    p = _paper("p1")
    cluster = _cluster(paper_ids=["p1"])
    embeddings = {"p1": [1.0, 0.0]}
    result = draft_related_work_subsection(cluster, {"p1": p}, embeddings, _MockEmbedder(), _MalformedLLMClient())
    assert isinstance(result, RelatedWorkSubsection)
    assert result.paragraph == "This is definitely not JSON."
    assert result.theme == "Cluster 0"


def test_draft_related_work_subsection_empty_cluster_paper_ids():
    cluster = ClusterAnalysis(
        cluster_id=0,
        paper_ids=[],
        what_exists="Exists.",
        what_is_contested="Contested.",
        what_is_missing="Missing.",
    )
    result = draft_related_work_subsection(cluster, {}, {}, _MockEmbedder(), _MockLLMClient())
    assert isinstance(result, RelatedWorkSubsection)
    assert result.cited_paper_ids == []


def test_draft_related_work_subsection_falls_back_to_what_exists_when_missing_empty():
    p = _paper("p1")
    cluster = ClusterAnalysis(
        cluster_id=0,
        paper_ids=["p1"],
        what_exists="Exists.",
        what_is_contested="Contested.",
        what_is_missing="",
    )
    embeddings = {"p1": [1.0, 0.0]}
    result = draft_related_work_subsection(cluster, {"p1": p}, embeddings, _MockEmbedder(), _MockLLMClient())
    assert isinstance(result, RelatedWorkSubsection)


def test_draft_related_work_subsection_populates_citation_statements():
    p = _paper("p1")
    cluster = _cluster(paper_ids=["p1"])
    embeddings = {"p1": [1.0, 0.0]}
    result = draft_related_work_subsection(cluster, {"p1": p}, embeddings, _MockEmbedder(), _MockLLMClient())
    assert len(result.citation_statements) == 1
    assert result.citation_statements[0].paper_id == "p1"
    assert result.citation_statements[0].snippet == "Key finding from abstract."


def test_draft_related_work_subsection_empty_citation_statements_on_malformed():
    p = _paper("p1")
    cluster = _cluster(paper_ids=["p1"])
    embeddings = {"p1": [1.0, 0.0]}
    result = draft_related_work_subsection(cluster, {"p1": p}, embeddings, _MockEmbedder(), _MalformedLLMClient())
    assert result.citation_statements == []


def test_draft_related_work_subsection_empty_citation_statements_when_key_missing():
    p = _paper("p1")
    cluster = _cluster(paper_ids=["p1"])
    embeddings = {"p1": [1.0, 0.0]}
    llm = _MockLLMClient(json.dumps({"theme": "T", "paragraph": "P"}))
    result = draft_related_work_subsection(cluster, {"p1": p}, embeddings, _MockEmbedder(), llm)
    assert result.citation_statements == []


# ---------------------------------------------------------------------------
# _build_citation_key_map
# ---------------------------------------------------------------------------

def test_build_citation_key_map_returns_correct_mapping():
    p = _paper("p1", authors=["Alice Smith"], year=2023)
    result = _build_citation_key_map([p])
    assert result == {"[Smith, 2023]": "p1"}


def test_build_citation_key_map_multiple_papers():
    p1 = _paper("p1", authors=["Alice Smith"], year=2023)
    p2 = _paper("p2", authors=["Bob Jones", "Carol Lee"], year=2021)
    result = _build_citation_key_map([p1, p2])
    assert result["[Smith, 2023]"] == "p1"
    assert result["[Jones et al., 2021]"] == "p2"


# ---------------------------------------------------------------------------
# _extract_citation_statements
# ---------------------------------------------------------------------------

def test_extract_citation_statements_matches_bracketed_key():
    key_to_id = {"[Smith, 2023]": "p1"}
    result = _extract_citation_statements([{"key": "[Smith, 2023]", "snippet": "A snippet."}], key_to_id)
    assert len(result) == 1
    assert result[0].paper_id == "p1"
    assert result[0].snippet == "A snippet."


def test_extract_citation_statements_normalises_key_without_brackets():
    key_to_id = {"[Smith, 2023]": "p1"}
    result = _extract_citation_statements([{"key": "Smith, 2023", "snippet": "A snippet."}], key_to_id)
    assert len(result) == 1
    assert result[0].paper_id == "p1"


def test_extract_citation_statements_skips_unknown_keys():
    key_to_id = {"[Smith, 2023]": "p1"}
    result = _extract_citation_statements([{"key": "[Ghost, 1900]", "snippet": "Hallucinated."}], key_to_id)
    assert result == []


def test_extract_citation_statements_skips_empty_snippet():
    key_to_id = {"[Smith, 2023]": "p1"}
    result = _extract_citation_statements([{"key": "[Smith, 2023]", "snippet": ""}], key_to_id)
    assert result == []


def test_extract_citation_statements_skips_non_dict_entries():
    key_to_id = {"[Smith, 2023]": "p1"}
    result = _extract_citation_statements(["not a dict", None, 42], key_to_id)
    assert result == []


def test_extract_citation_statements_empty_input():
    assert _extract_citation_statements([], {}) == []


# ---------------------------------------------------------------------------
# draft_abstract
# ---------------------------------------------------------------------------

def test_draft_abstract_returns_abstract_draft():
    report = _gap_report()
    result = draft_abstract(report, _AbstractLLMClient())
    assert isinstance(result, AbstractDraft)


def test_draft_abstract_fields_populated():
    report = _gap_report()
    result = draft_abstract(report, _AbstractLLMClient())
    assert result.background == "Background text."
    assert result.prior_work_summary == "Prior work summary."
    assert result.gap == "Gap text."
    assert "FILL IN" in result.proposed_approach
    assert result.expected_contribution == "Expected contribution."
    assert result.full_text == "Full abstract paragraph."


def test_draft_abstract_handles_malformed_llm_response():
    report = _gap_report()
    result = draft_abstract(report, _MalformedLLMClient())
    assert isinstance(result, AbstractDraft)
    assert result.background == "This is definitely not JSON."
    assert result.gap == ""


def test_draft_abstract_empty_overall_fields_graceful():
    report = GapAnalysisReport(
        input="q",
        clusters=[],
        overall_what_exists="",
        overall_what_is_contested="",
        overall_what_is_missing="",
    )
    result = draft_abstract(report, _AbstractLLMClient())
    assert isinstance(result, AbstractDraft)


# ---------------------------------------------------------------------------
# build_draft_report
# ---------------------------------------------------------------------------

def test_build_draft_report_returns_draft_report():
    p = _paper("p1")
    report = _gap_report([_cluster(paper_ids=["p1"])])
    embeddings = {"p1": [1.0, 0.0]}
    result = build_draft_report("query", report, [p], embeddings, _MockEmbedder(), _AbstractLLMClient())
    assert isinstance(result, DraftReport)


def test_build_draft_report_query_preserved():
    p = _paper("p1")
    report = _gap_report([_cluster(paper_ids=["p1"])])
    embeddings = {"p1": [1.0, 0.0]}
    result = build_draft_report("my query", report, [p], embeddings, _MockEmbedder(), _AbstractLLMClient())
    assert result.input == "my query"


def test_build_draft_report_subsection_count_matches_clusters():
    papers = [_paper("p1"), _paper("p2")]
    clusters = [_cluster(0, ["p1"]), _cluster(1, ["p2"])]
    report = _gap_report(clusters)
    embeddings = {"p1": [1.0, 0.0], "p2": [0.0, 1.0]}
    result = build_draft_report("q", report, papers, embeddings, _MockEmbedder(), _MockLLMClient())
    assert len(result.related_work.subsections) == 2


def test_build_draft_report_full_text_contains_themes():
    p = _paper("p1")
    report = _gap_report([_cluster(paper_ids=["p1"])])
    embeddings = {"p1": [1.0, 0.0]}
    result = build_draft_report("q", report, [p], embeddings, _MockEmbedder(), _MockLLMClient())
    assert "Test Theme" in result.related_work.full_text


def test_build_draft_report_single_cluster():
    p = _paper("p1")
    report = _gap_report([_cluster(0, ["p1"])])
    embeddings = {"p1": [1.0, 0.0]}
    result = build_draft_report("q", report, [p], embeddings, _MockEmbedder(), _MockLLMClient())
    assert len(result.related_work.subsections) == 1
    assert "###" in result.related_work.full_text


def test_build_draft_report_full_text_uses_double_newline_separator():
    papers = [_paper("p1"), _paper("p2")]
    clusters = [_cluster(0, ["p1"]), _cluster(1, ["p2"])]
    report = _gap_report(clusters)
    embeddings = {"p1": [1.0, 0.0], "p2": [0.0, 1.0]}
    result = build_draft_report("q", report, papers, embeddings, _MockEmbedder(), _MockLLMClient())
    assert "\n\n" in result.related_work.full_text


# ---------------------------------------------------------------------------
# llm_fallback field
# ---------------------------------------------------------------------------

def test_draft_related_work_subsection_sets_llm_fallback_false_on_valid():
    p = _paper("p1")
    cluster = _cluster(paper_ids=["p1"])
    embeddings = {"p1": [1.0, 0.0]}
    result = draft_related_work_subsection(cluster, {"p1": p}, embeddings, _MockEmbedder(), _MockLLMClient())
    assert result.llm_fallback is False


def test_draft_related_work_subsection_sets_llm_fallback_true_on_malformed():
    p = _paper("p1")
    cluster = _cluster(paper_ids=["p1"])
    embeddings = {"p1": [1.0, 0.0]}
    result = draft_related_work_subsection(cluster, {"p1": p}, embeddings, _MockEmbedder(), _MalformedLLMClient())
    assert result.llm_fallback is True


# ---------------------------------------------------------------------------
# _draft_caveats
# ---------------------------------------------------------------------------

def _subsection(llm_fallback: bool = False, cited_paper_ids: list[str] | None = None) -> RelatedWorkSubsection:
    return RelatedWorkSubsection(
        cluster_id=0,
        theme="T",
        paragraph="P",
        cited_paper_ids=cited_paper_ids or ["p1"],
        llm_fallback=llm_fallback,
    )


def test_draft_caveats_includes_paper_count():
    papers = [_paper("p1"), _paper("p2")]
    caveats = _draft_caveats(papers, [_subsection()])
    assert any("2 papers" in c for c in caveats)


def test_draft_caveats_mentions_missing_abstracts():
    papers = [_paper("p1", abstract=""), _paper("p2", abstract="Content.")]
    caveats = _draft_caveats(papers, [_subsection()])
    assert any("missing abstracts" in c for c in caveats)
    assert any("1" in c for c in caveats)


def test_draft_caveats_no_missing_abstract_message_when_all_present():
    papers = [_paper("p1"), _paper("p2")]
    caveats = _draft_caveats(papers, [_subsection()])
    assert not any("missing" in c for c in caveats)


def test_draft_caveats_mentions_fallback_cluster():
    papers = [_paper("p1")]
    caveats = _draft_caveats(papers, [_subsection(llm_fallback=True)])
    assert any("unparseable" in c or "malformed" in c.lower() or "raw text" in c for c in caveats)


def test_draft_caveats_no_fallback_message_when_none():
    papers = [_paper("p1")]
    caveats = _draft_caveats(papers, [_subsection(llm_fallback=False)])
    assert not any("unparseable" in c for c in caveats)


def test_build_draft_report_has_caveats():
    p = _paper("p1")
    report = _gap_report([_cluster(paper_ids=["p1"])])
    embeddings = {"p1": [1.0, 0.0]}
    result = build_draft_report("q", report, [p], embeddings, _MockEmbedder(), _MockLLMClient())
    assert isinstance(result.caveats, list)
    assert len(result.caveats) > 0  # always includes paper count caveat
