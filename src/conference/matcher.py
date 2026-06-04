from __future__ import annotations

import numpy as np

from src.data_classes import ConferenceMatch, ConferenceMatchReport, RawConferenceEntry
from src.literature_review.embedder import EmbedderProtocol
from src.utils import l2_normalize, l2_normalize_vector

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "NLP": ["language model", "natural language", "nlp", "text", "rag",
            "retrieval", "summarization", "translation", "dialogue",
            "question answering", "large language model", "bert", "transformer", "chatbot"],
    "ML":  ["machine learning", "deep learning", "neural network",
            "hallucination", "reinforcement learning",
            "classification", "regression", "generalization", "optimization", "training"],
    "CV":  ["computer vision", "image", "video", "visual",
            "detection", "segmentation", "recognition"],
    "DM":  ["data mining", "knowledge graph", "recommendation",
            "information retrieval", "web search", "knowledge base"],
    "SE":  ["software engineering", "code generation", "program synthesis",
            "static analysis", "debugging", "testing"],
}


def _parse_categories(entry: RawConferenceEntry) -> list[str]:
    return [t.strip() for t in (entry.category or "").split(",") if t.strip()]


def filter_by_category(
    entries: list[RawConferenceEntry],
    abstract_text: str,
) -> list[RawConferenceEntry]:
    lowered = abstract_text.lower()
    matched: dict[str, list[str]] = {}
    for code, keywords in _CATEGORY_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in lowered]
        if hits:
            matched[code] = hits

    if not matched:
        return entries
    matched_codes = set(matched)
    filtered = [
        e for e in entries
        if matched_codes & set(_parse_categories(e))
    ]
    return filtered if filtered else entries


def _build_scope_string(entry: RawConferenceEntry) -> str:
    parts = [entry.full_name] + _parse_categories(entry)
    return " ".join(parts)


def rank_conferences(
    abstract_text: str,
    entries: list[RawConferenceEntry],
    embedder: EmbedderProtocol,
    top_n: int = 10,
) -> list[tuple[RawConferenceEntry, float]]:
    if not entries:
        return []

    scope_strings = [_build_scope_string(e) for e in entries]
    all_vecs = embedder.embed([abstract_text] + scope_strings)

    abstract_vec = np.array(all_vecs[0], dtype=float)
    conf_matrix = np.array(all_vecs[1:], dtype=float)

    conf_matrix = l2_normalize(conf_matrix)
    abstract_vec = l2_normalize_vector(abstract_vec)

    scores = conf_matrix @ abstract_vec
    n = min(top_n, len(entries))
    top_indices = np.argsort(scores)[::-1][:n]
    return [(entries[i], float(scores[i])) for i in top_indices]


def build_conference_match_report(
    abstract_text: str,
    entries: list[RawConferenceEntry],
    embedder: EmbedderProtocol,
    top_n: int = 10,
) -> ConferenceMatchReport:
    ranked = rank_conferences(abstract_text, entries, embedder, top_n)
    matches = [
        ConferenceMatch(
            conference_id=entry.id,
            name=entry.full_name,
            short_name=entry.name,
            similarity=score,
            deadline=entry.paper_deadline,
            abstract_deadline=entry.abstract_deadline,
            link=entry.homepage,
            subject_areas=_parse_categories(entry),
        )
        for entry, score in ranked
    ]
    return ConferenceMatchReport(input=abstract_text, matches=matches, top_n=top_n)
