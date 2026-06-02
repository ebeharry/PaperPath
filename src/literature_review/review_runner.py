import sys

import requests

from src.literature_review.clients import search_arxiv, search_semantic_scholar
from src.literature_review.data_classes import GapAnalysisReport, Paper
from src.literature_review.embedder import embed_papers, make_embedder
from src.literature_review.clusterer import cluster_papers
from src.literature_review.gap_analysis import build_gap_report, make_llm_client


def run(project_description: str, max_papers: int = 20, ss_sort: str | None = None, arxiv_sort: str | None = None, year: str | None = None) -> list[Paper]:
    """
    Search Semantic Scholar and arXiv and return combined results.

    max_papers is applied per source, so up to 2 * max_papers papers may be returned.
    If one source is unavailable, results from the other are returned with a warning.
    """
    ss_papers: list[Paper] = []
    arxiv_papers: list[Paper] = []
    last_exc: Exception | None = None

    try:
        ss_papers = search_semantic_scholar(project_description, limit=max_papers, sort=ss_sort, year=year)
    except requests.RequestException as e:
        print(f"warning: semantic scholar unavailable: {e}", file=sys.stderr)
        last_exc = e

    try:
        arxiv_papers = search_arxiv(project_description, limit=max_papers, sort=arxiv_sort, year=year)
    except requests.RequestException as e:
        print(f"warning: arxiv unavailable: {e}", file=sys.stderr)
        last_exc = e

    if not ss_papers and not arxiv_papers:
        raise last_exc  # type: ignore[misc]

    seen: set[str] = set()
    result: list[Paper] = []
    for paper in ss_papers + arxiv_papers:
        key = paper.title.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(paper)
    return result


def run_with_analysis(
    project_description: str,
    max_papers: int = 20,
    ss_sort: str | None = None,
    arxiv_sort: str | None = None,
    year: str | None = None,
    embed_backend: str = "local",
    llm_backend: str = "openai",
) -> tuple[list[Paper], GapAnalysisReport]:
    papers = run(project_description, max_papers, ss_sort, arxiv_sort, year)
    embedder = make_embedder(embed_backend)
    embeddings = embed_papers(papers, embedder)
    clusters = cluster_papers(papers, embeddings)
    llm = make_llm_client(llm_backend)
    report = build_gap_report(project_description, clusters, llm)
    return papers, report
