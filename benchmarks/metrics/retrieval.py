from __future__ import annotations

import numpy as np

from src.data_classes import Paper, RelatedWorkSubsection
from src.literature_review.embedder import EmbedderProtocol


def retrieval_relevance_at_k(
    papers: list[Paper],
    gold_abstract: str,
    embedder: EmbedderProtocol,
    k: int,
) -> float:
    candidates = papers[:k]
    if not candidates or not gold_abstract:
        return 0.0
    texts = [f"{p.title}. {p.abstract}" for p in candidates] + [gold_abstract]
    raw = embedder.embed(texts)
    vecs = np.array(raw, dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs = vecs / norms
    paper_vecs = vecs[:-1]
    gold_vec = vecs[-1]
    return float(np.mean(paper_vecs @ gold_vec))


def hallucinated_citation_rate(
    subsections: list[RelatedWorkSubsection],
    retrieved_paper_ids: set[str],
) -> float:
    all_cited = [pid for s in subsections for pid in s.cited_paper_ids]
    if not all_cited:
        return 0.0
    hallucinated = sum(1 for pid in all_cited if pid not in retrieved_paper_ids)
    return hallucinated / len(all_cited)
