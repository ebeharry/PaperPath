from __future__ import annotations
import datetime

import numpy as np

from src.data_classes import Paper
from src.literature_review.embedder import EmbedderProtocol
from src.utils import l2_normalize, l2_normalize_vector


def rank_papers(
    query: str,
    papers: list[Paper],
    embedder: EmbedderProtocol,
    current_year: int | None = None,
) -> list[Paper]:
    """Re-rank papers by a weighted hybrid score, returning Papers with score fields populated.

    score = 0.35*title_score + 0.35*abstract_score + 0.20*recency_score + 0.10*citation_score
    """
    if not papers:
        return []

    if current_year is None:
        current_year = datetime.date.today().year

    # Batch: [query, title_0, abstract_0, title_1, abstract_1, ...]
    texts: list[str] = [query]
    for p in papers:
        texts.append(p.title)
        texts.append(p.abstract)

    raw = embedder.embed(texts)
    all_vecs = np.array(raw, dtype=float)

    query_vec = l2_normalize_vector(all_vecs[0])
    title_vecs = l2_normalize(all_vecs[1::2])    # shape (n, dim)
    abstract_vecs = l2_normalize(all_vecs[2::2])  # shape (n, dim)

    title_sims = np.clip(title_vecs @ query_vec, 0.0, 1.0)
    abstract_sims = np.clip(abstract_vecs @ query_vec, 0.0, 1.0)

    n = len(papers)
    recency_scores = np.array([
        max(0.0, 1.0 - (current_year - p.year) / 10.0) if p.year is not None else 0.0
        for p in papers
    ])

    citation_scores = np.zeros(n)
    has_count = [(i, p.citation_count) for i, p in enumerate(papers) if p.citation_count is not None]
    if len(has_count) > 1:
        sorted_by_count = sorted(has_count, key=lambda x: x[1])
        n_with = len(sorted_by_count)
        for rank_pos, (idx, _) in enumerate(sorted_by_count):
            citation_scores[idx] = rank_pos / (n_with - 1)

    final = (
        0.35 * title_sims
        + 0.35 * abstract_sims
        + 0.20 * recency_scores
        + 0.10 * citation_scores
    )

    combined = sorted(
        zip(papers, title_sims, abstract_sims, recency_scores, citation_scores, final),
        key=lambda x: x[5],
        reverse=True,
    )

    result: list[Paper] = []
    for i, (paper, t_s, a_s, r_s, c_s, f_s) in enumerate(combined):
        result.append(paper.model_copy(update={
            "rank": i + 1,
            "title_score": round(float(t_s), 4),
            "abstract_score": round(float(a_s), 4),
            "recency_score": round(float(r_s), 4),
            "citation_score": round(float(c_s), 4),
            "final_score": round(float(f_s), 4),
        }))
    return result
