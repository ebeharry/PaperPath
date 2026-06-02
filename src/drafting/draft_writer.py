from __future__ import annotations
import json
import logging
import re

import numpy as np

from src.data_classes import (
    AbstractDraft,
    ClusterAnalysis,
    DraftReport,
    GapAnalysisReport,
    Paper,
    RelatedWorkDraft,
    RelatedWorkSubsection,
)
from src.literature_review.embedder import EmbedderProtocol
from src.literature_review.gap_analysis import LLMClientProtocol

logger = logging.getLogger(__name__)

_ABSTRACT_LIMIT = 400
_DEFAULT_TOP_K = 5


def retrieve_top_k(
    query_embedding: list[float],
    embeddings: dict[str, list[float]],
    candidate_paper_ids: list[str],
    k: int = _DEFAULT_TOP_K,
) -> list[str]:
    valid_ids = [pid for pid in candidate_paper_ids if pid in embeddings]
    if not valid_ids:
        return []

    matrix = np.array([embeddings[pid] for pid in valid_ids], dtype=float)
    query = np.array(query_embedding, dtype=float)

    # L2-normalize
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms = np.where(row_norms == 0, 1.0, row_norms)
    matrix = matrix / row_norms

    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        q_norm = 1.0
    query = query / q_norm

    scores = matrix @ query
    top_indices = np.argsort(scores)[::-1][: min(k, len(valid_ids))]
    return [valid_ids[i] for i in top_indices]


def _format_citation_key(paper: Paper) -> str:
    year = str(paper.year) if paper.year is not None else "n.d."
    if not paper.authors:
        return f"[Unknown, {year}]"
    last_name = paper.authors[0].rsplit(" ", 1)[-1]
    if len(paper.authors) >= 2:
        return f"[{last_name} et al., {year}]"
    return f"[{last_name}, {year}]"


def _format_paper_snippet(paper: Paper) -> str:
    key = _format_citation_key(paper)
    if paper.abstract:
        snippet = paper.abstract[:_ABSTRACT_LIMIT]
        return f"{key} {paper.title}: {snippet}"
    return f"{key} {paper.title}"


def _build_related_work_prompt(
    cluster_analysis: ClusterAnalysis,
    context_papers: list[Paper],
) -> str:
    paper_lines = "\n".join(_format_paper_snippet(p) for p in context_papers)
    return (
        f"You are an academic writing assistant. Draft one subsection of a Related Work section.\n\n"
        f"Research gap context:\n"
        f"  - What currently exists: {cluster_analysis.what_exists}\n"
        f"  - What is contested: {cluster_analysis.what_is_contested}\n"
        f"  - What is missing: {cluster_analysis.what_is_missing}\n\n"
        f"Representative papers:\n{paper_lines}\n\n"
        f"Instructions:\n"
        f"1. Infer a concise thematic label (5-10 words) for this cluster.\n"
        f"2. Write a 3-5 sentence paragraph that names the theme, cites at least 2 papers "
        f"inline using their bracketed citation keys exactly as shown, notes contested aspects, "
        f"and closes with the gap.\n"
        f"3. Do NOT invent citations beyond those provided.\n\n"
        f'Respond with ONLY a JSON object (no markdown, no code fences) with exactly these two keys:\n'
        f'{{"theme": "...", "paragraph": "..."}}'
    )


def _build_abstract_prompt(gap_report: GapAnalysisReport) -> str:
    proposed_approach_instruction = (
        f'For "proposed_approach", base it on this project description: {gap_report.input}'
        if gap_report.input
        else f'For "proposed_approach", output this placeholder verbatim: '
             f'"[FILL IN: describe the approach your paper proposes to address the gap]"'
    )
    return (
        f"You are an academic writing assistant drafting a structured abstract.\n\n"
        f"Prior work landscape:\n"
        f"  - What prior work has done: {gap_report.overall_what_exists}\n"
        f"  - What remains contested: {gap_report.overall_what_is_contested}\n"
        f"  - The key gap: {gap_report.overall_what_is_missing}\n\n"
        f"Instructions:\n"
        f"Write five distinct fields, each 1-3 sentences.\n"
        f"{proposed_approach_instruction}\n\n"
        f"Respond with ONLY a JSON object (no markdown, no code fences) with exactly these six keys:\n"
        f'{{"background": "...", "prior_work_summary": "...", "gap": "...", '
        f'"proposed_approach": "...", "expected_contribution": "...", '
        f'"full_text": "A single cohesive abstract paragraph (4-6 sentences) that flows naturally '
        f'through background, prior work, gap, proposed approach, and expected contribution."}}'
    )


def _parse_draft_response(response: str) -> dict[str, str]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return {k: str(v) for k, v in data.items()}
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse LLM JSON response; using raw text fallback")
    return {"_raw": response}


def draft_related_work_subsection(
    cluster_analysis: ClusterAnalysis,
    papers_by_id: dict[str, Paper],
    embeddings: dict[str, list[float]],
    embedder: EmbedderProtocol,
    llm: LLMClientProtocol,
    top_k: int = _DEFAULT_TOP_K,
) -> RelatedWorkSubsection:
    query_text = cluster_analysis.what_is_missing or cluster_analysis.what_exists or "research gap"
    query_vec = embedder.embed([query_text])[0]

    retrieved_ids = retrieve_top_k(query_vec, embeddings, cluster_analysis.paper_ids, k=top_k)

    if retrieved_ids:
        context_papers = [papers_by_id[pid] for pid in retrieved_ids if pid in papers_by_id]
    else:
        context_papers = [papers_by_id[pid] for pid in cluster_analysis.paper_ids if pid in papers_by_id]

    prompt = _build_related_work_prompt(cluster_analysis, context_papers)
    raw = llm.complete(prompt)
    parsed = _parse_draft_response(raw)

    if "_raw" in parsed:
        return RelatedWorkSubsection(
            cluster_id=cluster_analysis.cluster_id,
            theme=f"Cluster {cluster_analysis.cluster_id}",
            paragraph=parsed["_raw"],
            cited_paper_ids=retrieved_ids,
        )

    return RelatedWorkSubsection(
        cluster_id=cluster_analysis.cluster_id,
        theme=parsed.get("theme", f"Cluster {cluster_analysis.cluster_id}"),
        paragraph=parsed.get("paragraph", ""),
        cited_paper_ids=retrieved_ids,
    )


def draft_abstract(
    gap_report: GapAnalysisReport,
    llm: LLMClientProtocol,
) -> AbstractDraft:
    prompt = _build_abstract_prompt(gap_report)
    raw = llm.complete(prompt)
    parsed = _parse_draft_response(raw)

    if "_raw" in parsed:
        return AbstractDraft(
            background=parsed["_raw"],
            prior_work_summary="",
            gap="",
            proposed_approach="",
            expected_contribution="",
            full_text="",
        )

    return AbstractDraft(
        background=parsed.get("background", ""),
        prior_work_summary=parsed.get("prior_work_summary", ""),
        gap=parsed.get("gap", ""),
        proposed_approach=parsed.get(
            "proposed_approach",
            "[FILL IN: describe the approach your paper proposes to address the gap]",
        ),
        expected_contribution=parsed.get("expected_contribution", ""),
        full_text=parsed.get("full_text", ""),
    )


def build_draft_report(
    query: str,
    gap_report: GapAnalysisReport,
    papers: list[Paper],
    embeddings: dict[str, list[float]],
    embedder: EmbedderProtocol,
    llm: LLMClientProtocol,
    top_k: int = _DEFAULT_TOP_K,
) -> DraftReport:
    papers_by_id = {p.paper_id: p for p in papers}

    subsections = [
        draft_related_work_subsection(cluster, papers_by_id, embeddings, embedder, llm, top_k)
        for cluster in gap_report.clusters
    ]

    full_text = "\n\n".join(f"### {s.theme}\n\n{s.paragraph}" for s in subsections)
    related_work = RelatedWorkDraft(subsections=subsections, full_text=full_text)

    abstract = draft_abstract(gap_report, llm)

    return DraftReport(input=query, related_work=related_work, abstract=abstract)
