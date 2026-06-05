from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.data_classes import DraftReport, Paper
from src.literature_review.embedder import EmbedderProtocol

from benchmarks.metrics.abstract_quality import abstract_embedding_similarity
from benchmarks.metrics.latex_check import compile_success as _compile_success
from benchmarks.metrics.retrieval import (
    hallucinated_citation_rate,
    retrieval_relevance_at_k as _retrieval_relevance_at_k,
)


@dataclass
class BenchmarkResult:
    gold_paper_id: str
    gold_title: str
    retrieval_relevance_at_k: float | None
    hallucinated_citation_rate: float | None
    abstract_embedding_similarity: float | None
    compile_success: bool | None


def run_benchmark(
    gold_entry: dict,
    papers: list[Paper],
    draft_report: DraftReport | None = None,
    embedder: EmbedderProtocol | None = None,
    tex_path: str | None = None,
    k: int = 10,
) -> BenchmarkResult:
    retrieved_set = set(p.paper_id for p in papers)

    # retrieval relevance@k (mean cosine similarity of top-k retrieved to gold abstract)
    gold_abstract = gold_entry.get("gold_abstract", "")
    rrak = None
    if embedder is not None and gold_abstract:
        rrak = _retrieval_relevance_at_k(papers, gold_abstract, embedder, k)

    # hallucinated citation rate
    hcr = None
    if draft_report is not None:
        hcr = hallucinated_citation_rate(
            draft_report.related_work.subsections, retrieved_set
        )

    # abstract embedding similarity
    aes = None
    if draft_report is not None and embedder is not None and gold_abstract:
        aes = abstract_embedding_similarity(
            draft_report.abstract.full_text, gold_abstract, embedder
        )

    # compile success
    cs = None
    if tex_path is not None:
        cs = _compile_success(tex_path)

    return BenchmarkResult(
        gold_paper_id=gold_entry["paper_id"],
        gold_title=gold_entry.get("title", ""),
        retrieval_relevance_at_k=rrak,
        hallucinated_citation_rate=hcr,
        abstract_embedding_similarity=aes,
        compile_success=cs,
    )


def load_gold_entry(path: str) -> dict:
    return json.loads(Path(path).read_text())
