import sys
from dataclasses import dataclass, field

import requests

from src.literature_review.clients import search_arxiv, search_semantic_scholar
from src.data_classes import ConferenceMatchReport, DraftReport, GapAnalysisReport, Paper
from src.literature_review.embedder import EmbedderProtocol, embed_papers, make_embedder
from src.conference.fetcher import fetch_conferences, filter_future_conferences
from src.conference.matcher import build_conference_match_report, filter_by_category
from src.literature_review.clusterer import cluster_papers
from src.literature_review.gap_analysis import build_gap_report, make_llm_client
from src.drafting.draft_writer import build_draft_report


@dataclass
class SearchParams:
    ss_sort: str | None = None
    arxiv_sort: str | None = None
    year: str | None = None


def run(query: str, max_papers: int = 20, ss_sort: str | None = None, arxiv_sort: str | None = None, year: str | None = None) -> list[Paper]:
    """
    Search Semantic Scholar and arXiv and return combined results.

    max_papers is applied per source, so up to 2 * max_papers papers may be returned.
    If one source is unavailable, results from the other are returned with a warning.
    """
    ss_papers: list[Paper] = []
    arxiv_papers: list[Paper] = []
    last_exc: Exception | None = None

    try:
        ss_papers = search_semantic_scholar(query, limit=max_papers, sort=ss_sort, year=year)
    except requests.RequestException as e:
        print(f"warning: semantic scholar unavailable: {e}", file=sys.stderr)
        last_exc = e

    try:
        arxiv_papers = search_arxiv(query, limit=max_papers, sort=arxiv_sort, year=year)
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


def _progress(msg: str) -> None:
    print(f"  {msg}", flush=True)


def run_with_analysis(
    query: str,
    project_description: str | None = None,
    max_papers: int = 20,
    search_params: SearchParams | None = None,
    embed_backend: str = "local",
    llm_backend: str = "openrouter",
) -> tuple[list[Paper], GapAnalysisReport]:
    sp = search_params or SearchParams()
    llm_context = project_description or query
    _progress("Searching Papers...")
    papers = run(query, max_papers, sp.ss_sort, sp.arxiv_sort, sp.year)
    _progress(f"Found {len(papers)} Papers — Analysing Gaps...")
    embedder = make_embedder(embed_backend)
    embeddings = embed_papers(papers, embedder)
    clusters = cluster_papers(papers, embeddings)
    llm = make_llm_client(llm_backend)
    report = build_gap_report(llm_context, clusters, llm)
    return papers, report


def run_with_drafts(
    query: str,
    project_description: str | None = None,
    max_papers: int = 20,
    search_params: SearchParams | None = None,
    embed_backend: str = "local",
    llm_backend: str = "openrouter",
    top_k: int = 5,
    embedder: EmbedderProtocol | None = None,
) -> tuple[list[Paper], GapAnalysisReport, DraftReport]:
    sp = search_params or SearchParams()
    llm_context = project_description or query
    _progress("Searching Papers...")
    papers = run(query, max_papers, sp.ss_sort, sp.arxiv_sort, sp.year)
    _progress(f"Found {len(papers)} Papers — Analysing Gaps...")
    if embedder is None:
        embedder = make_embedder(embed_backend)
    embeddings = embed_papers(papers, embedder)
    clusters = cluster_papers(papers, embeddings)
    llm = make_llm_client(llm_backend)
    gap_report = build_gap_report(llm_context, clusters, llm)
    _progress("Drafting Abstract and Related Work...")
    draft_report = build_draft_report(llm_context, gap_report, papers, embeddings, embedder, llm, top_k=top_k)
    return papers, gap_report, draft_report


def run_with_conference_matching(
    query: str,
    project_description: str | None = None,
    max_papers: int = 20,
    search_params: SearchParams | None = None,
    embed_backend: str = "local",
    llm_backend: str = "openrouter",
    top_k: int = 5,
    top_n: int = 10,
) -> tuple[list[Paper], GapAnalysisReport, DraftReport, ConferenceMatchReport]:
    embedder = make_embedder(embed_backend)
    papers, gap_report, draft_report = run_with_drafts(
        query, project_description, max_papers, search_params,
        embed_backend, llm_backend, top_k, embedder=embedder,
    )
    _progress("Matching Conferences...")
    if draft_report.abstract.full_text:
        abstract_text = draft_report.abstract.full_text
    elif project_description:
        abstract_text = project_description
    else:
        abstract_text = query
    raw_entries = fetch_conferences()
    future_entries = filter_future_conferences(raw_entries)
    category_entries = filter_by_category(future_entries, abstract_text)
    match_report = build_conference_match_report(abstract_text, category_entries, embedder, top_n)
    return papers, gap_report, draft_report, match_report
