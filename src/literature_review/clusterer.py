from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from src.literature_review.data_classes import Paper

_MAX_CLUSTERS = 8

def _select_k(embeddings: np.ndarray) -> int:
    n = len(embeddings)
    if n < 4:
        return 1
    best_k, best_score = 2, -1.0
    for k in range(2, min(n, _MAX_CLUSTERS + 1)):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(embeddings)
        try:
            score = silhouette_score(embeddings, labels)
        except ValueError:
            score = -1.0
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_papers(
    papers: list[Paper],
    embeddings: dict[str, list[float]],
) -> dict[int, list[Paper]]:
    ordered = [p for p in papers if p.paper_id in embeddings]
    if not ordered:
        return {}

    matrix = np.array([embeddings[p.paper_id] for p in ordered], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms

    k = _select_k(matrix)
    if k == 1:
        return {0: ordered}

    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(matrix)
    clusters: dict[int, list[Paper]] = {}
    for paper, label in zip(ordered, labels):
        clusters.setdefault(int(label), []).append(paper)
    return clusters
