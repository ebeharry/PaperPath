from __future__ import annotations

import math

import pytest

from src.conference.matcher import _build_scope_string, build_conference_match_report, filter_by_category, rank_conferences
from src.data_classes import ConferenceMatchReport, RawConferenceEntry


class _MockEmbedder:
    def __init__(self, vectors: list[list[float]] | None = None):
        self._vectors = vectors
        self.call_count = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        if self._vectors is not None:
            return self._vectors[: len(texts)]
        # distinct vectors with increasing magnitude so ranking tests using the default produce non-identical scores
        return [[float(i + 1), 0.0] for i in range(len(texts))]


def _entry(
    eid: str = "conf-2030",
    name: str = "CONF",
    full_name: str = "Conference on ML",
    paper_deadline: str = "2030-01-01 23:59:59",
    category: str | None = "ML",
) -> RawConferenceEntry:
    return RawConferenceEntry(
        id=eid, name=name, full_name=full_name, paper_deadline=paper_deadline, category=category
    )


# --- _build_scope_string ---

def test_build_scope_string_with_category():
    e = _entry(full_name="Neural Information Processing Systems", category="ML")
    assert _build_scope_string(e) == "Neural Information Processing Systems ML"


def test_build_scope_string_multiple_categories():
    e = _entry(full_name="SIGKDD", category="DM, ML")
    assert _build_scope_string(e) == "SIGKDD DM ML"


def test_build_scope_string_none_category():
    e = _entry(full_name="Some Conference", category=None)
    assert _build_scope_string(e) == "Some Conference"


def test_build_scope_string_empty_category():
    e = _entry(full_name="Some Conference", category="")
    assert _build_scope_string(e) == "Some Conference"


def test_build_scope_string_whitespace_category():
    e = _entry(full_name="Some Conference", category="  ,  ")
    assert _build_scope_string(e) == "Some Conference"


# --- rank_conferences ---

def test_rank_conferences_sorted_descending():
    # abstract vec: [1, 0]
    # conf 0 scope: [1, 0]  → similarity 1.0
    # conf 1 scope: [0, 1]  → similarity 0.0
    # conf 2 scope: [0.707, 0.707] → similarity ~0.707
    abstract_vec = [1.0, 0.0]
    conf_vecs = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.707, 0.707],
    ]
    all_vecs = [abstract_vec] + conf_vecs
    embedder = _MockEmbedder(vectors=all_vecs)
    entries = [_entry(eid=f"c{i}") for i in range(3)]

    result = rank_conferences("abstract", entries, embedder, top_n=3)
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0][0].id == "c0"


def test_rank_conferences_capped_at_top_n():
    all_vecs = [[1.0, 0.0]] * 6  # 1 abstract + 5 conferences
    embedder = _MockEmbedder(vectors=all_vecs)
    entries = [_entry(eid=f"c{i}") for i in range(5)]
    result = rank_conferences("abstract", entries, embedder, top_n=3)
    assert len(result) == 3


def test_rank_conferences_fewer_than_top_n():
    all_vecs = [[1.0, 0.0]] * 3  # 1 abstract + 2 conferences
    embedder = _MockEmbedder(vectors=all_vecs)
    entries = [_entry(eid=f"c{i}") for i in range(2)]
    result = rank_conferences("abstract", entries, embedder, top_n=10)
    assert len(result) == 2


def test_rank_conferences_empty_entries():
    embedder = _MockEmbedder()
    result = rank_conferences("abstract", [], embedder, top_n=5)
    assert result == []
    assert embedder.call_count == 0


def test_rank_conferences_zero_norm_abstract():
    all_vecs = [[0.0, 0.0], [1.0, 0.0]]
    embedder = _MockEmbedder(vectors=all_vecs)
    entries = [_entry()]
    # Should not raise even with zero-norm abstract vector
    result = rank_conferences("abstract", entries, embedder, top_n=1)
    assert len(result) == 1


# --- build_conference_match_report ---

def test_build_conference_match_report_returns_correct_type():
    all_vecs = [[1.0, 0.0], [1.0, 0.0]]
    embedder = _MockEmbedder(vectors=all_vecs)
    report = build_conference_match_report("my abstract", [_entry()], embedder, top_n=5)
    assert isinstance(report, ConferenceMatchReport)


def test_build_conference_match_report_sets_input():
    all_vecs = [[1.0, 0.0], [1.0, 0.0]]
    embedder = _MockEmbedder(vectors=all_vecs)
    report = build_conference_match_report("test abstract", [_entry()], embedder, top_n=5)
    assert report.input == "test abstract"


def test_build_conference_match_report_respects_top_n():
    n = 5
    all_vecs = [[1.0, 0.0]] * (n + 1)
    embedder = _MockEmbedder(vectors=all_vecs)
    entries = [_entry(eid=f"c{i}") for i in range(n)]
    report = build_conference_match_report("abstract", entries, embedder, top_n=3)
    assert report.top_n == 3
    assert len(report.matches) == 3


def test_build_conference_match_report_calls_embedder_once():
    all_vecs = [[1.0, 0.0]] * 4
    embedder = _MockEmbedder(vectors=all_vecs)
    entries = [_entry(eid=f"c{i}") for i in range(3)]
    build_conference_match_report("abstract", entries, embedder, top_n=3)
    assert embedder.call_count == 1


def test_build_conference_match_report_match_fields():
    all_vecs = [[1.0, 0.0], [1.0, 0.0]]
    embedder = _MockEmbedder(vectors=all_vecs)
    e = RawConferenceEntry(
        id="neurips-2030",
        name="NeurIPS",
        full_name="Neural Information Processing Systems",
        paper_deadline="2030-01-01 23:59:59",
        abstract_deadline="2029-12-15 23:59:59",
        homepage="https://neurips.cc/",
        category="ML",
    )
    report = build_conference_match_report("abstract", [e], embedder, top_n=1)
    m = report.matches[0]
    assert m.conference_id == "neurips-2030"
    assert m.short_name == "NeurIPS"
    assert m.name == "Neural Information Processing Systems"
    assert m.subject_areas == ["ML"]
    assert m.deadline == "2030-01-01 23:59:59"
    assert m.abstract_deadline == "2029-12-15 23:59:59"
    assert m.link == "https://neurips.cc/"


def test_build_conference_match_report_subject_areas_from_category():
    all_vecs = [[1.0, 0.0], [1.0, 0.0]]
    embedder = _MockEmbedder(vectors=all_vecs)
    e = _entry(category="DM, ML, CV")
    report = build_conference_match_report("abstract", [e], embedder, top_n=1)
    assert report.matches[0].subject_areas == ["DM", "ML", "CV"]


def test_build_conference_match_report_none_category_yields_empty_areas():
    all_vecs = [[1.0, 0.0], [1.0, 0.0]]
    embedder = _MockEmbedder(vectors=all_vecs)
    e = _entry(category=None)
    report = build_conference_match_report("abstract", [e], embedder, top_n=1)
    assert report.matches[0].subject_areas == []


# --- filter_by_category ---

def test_filter_by_category_keeps_matching():
    nlp = _entry(eid="acl", category="NLP")
    cv = _entry(eid="cvpr", category="CV")
    abstract = "We propose a new language model for natural language understanding."
    result = filter_by_category([nlp, cv], abstract)
    assert len(result) == 1
    assert result[0].id == "acl"


def test_filter_by_category_keeps_multi_category_overlap():
    ml_nlp = _entry(eid="aaai", category="ML, NLP")
    cv = _entry(eid="cvpr", category="CV")
    abstract = "Deep learning for text classification using transformer models."
    result = filter_by_category([ml_nlp, cv], abstract)
    assert len(result) == 1
    assert result[0].id == "aaai"


def test_filter_by_category_fallback_when_no_keywords_match():
    nlp = _entry(eid="acl", category="NLP")
    cv = _entry(eid="cvpr", category="CV")
    abstract = "A study of cooking techniques in molecular gastronomy."
    result = filter_by_category([nlp, cv], abstract)
    assert len(result) == 2


def test_filter_by_category_empty_entries():
    result = filter_by_category([], "language model for summarization")
    assert result == []


def test_filter_by_category_excludes_no_category():
    no_cat = _entry(eid="misc", category=None)
    nlp = _entry(eid="acl", category="NLP")
    abstract = "A retrieval-augmented generation system for scientific summarization."
    result = filter_by_category([no_cat, nlp], abstract)
    assert len(result) == 1
    assert result[0].id == "acl"


def test_filter_by_category_multiple_matches_all_returned():
    acl = _entry(eid="acl", category="NLP")
    emnlp = _entry(eid="emnlp", category="NLP")
    cv = _entry(eid="cvpr", category="CV")
    abstract = "natural language processing with transformer models"
    result = filter_by_category([acl, emnlp, cv], abstract)
    assert len(result) == 2
    assert {e.id for e in result} == {"acl", "emnlp"}


def test_rank_conferences_tied_scores():
    # All conference vectors identical to abstract → all similarity scores equal
    all_vecs = [[1.0, 0.0]] * 4  # abstract + 3 conferences
    embedder = _MockEmbedder(vectors=all_vecs)
    entries = [_entry(eid=f"c{i}") for i in range(3)]
    result = rank_conferences("abstract", entries, embedder, top_n=3)
    assert len(result) == 3
    scores = [s for _, s in result]
    assert all(s == scores[0] for s in scores)


def test_rank_conferences_empty_abstract_string():
    # Empty abstract produces a zero-norm vector; function should not raise
    all_vecs = [[0.0, 0.0], [1.0, 0.0]]  # abstract (zero norm) + 1 conference
    embedder = _MockEmbedder(vectors=all_vecs)
    entries = [_entry()]
    result = rank_conferences("", entries, embedder, top_n=1)
    assert len(result) == 1
