from __future__ import annotations

import numpy as np

from src.literature_review.embedder import EmbedderProtocol
from src.utils import l2_normalize_vector


def abstract_embedding_similarity(
    generated: str,
    gold: str,
    embedder: EmbedderProtocol,
) -> float:
    vectors = embedder.embed([generated, gold])
    a = l2_normalize_vector(np.array(vectors[0], dtype=float))
    b = l2_normalize_vector(np.array(vectors[1], dtype=float))
    return float(np.dot(a, b))
