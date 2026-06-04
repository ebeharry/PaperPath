import sys
from dataclasses import dataclass, field

import requests

from src.literature_review.clients import search_arxiv, search_semantic_scholar
from src.data_classes import ConferenceMatch, ConferenceMatchReport, DraftReport, GapAnalysisReport, Paper
from src.literature_review.embedder import EmbedderProtocol, embed_papers, make_embedder
from src.literature_review.ranker import rank_papers
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


def run(query: str, search_limit: int = 20, ss_sort: str | None = None, arxiv_sort: str | None = None, year: str | None = None) -> list[Paper]:
    """
    Search Semantic Scholar and arXiv and return combined results.

    search_limit is applied per source, so up to 2 * search_limit papers may be returned.
    If one source is unavailable, results from the other are returned with a warning.
    """
    ss_papers: list[Paper] = []
    arxiv_papers: list[Paper] = []
    last_exc: Exception | None = None

    try:
        ss_papers = search_semantic_scholar(query, limit=search_limit, sort=ss_sort, year=year)
    except requests.HTTPError as e:
        label = "rate-limited (429)" if e.response is not None and e.response.status_code == 429 else f"error {e.response.status_code if e.response is not None else '?'}"
        print(f"warning: semantic scholar {label}: {e}", file=sys.stderr)
        last_exc = e
    except requests.RequestException as e:
        print(f"warning: semantic scholar unavailable: {e}", file=sys.stderr)
        last_exc = e

    try:
        arxiv_papers = search_arxiv(query, limit=search_limit, sort=arxiv_sort, year=year)
    except requests.HTTPError as e:
        label = "rate-limited (429)" if e.response is not None and e.response.status_code == 429 else f"error {e.response.status_code if e.response is not None else '?'}"
        print(f"warning: arxiv {label}: {e}", file=sys.stderr)
        last_exc = e
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


def _analysis_caveats(
    papers: list[Paper],
    clusters: dict[int, list[Paper]],
) -> list[str]:
    caveats: list[str] = []

    sources = {p.source for p in papers}
    if len(sources) == 1:
        if "arxiv" in sources:
            caveats.append("Only arXiv papers found; venue fit is low confidence.")
        else:
            caveats.append("Only Semantic Scholar papers found; may miss recent preprints.")

    no_abstract = sum(1 for p in papers if not p.abstract.strip())
    if no_abstract:
        caveats.append(
            f"{no_abstract} of {len(papers)} papers have no abstract; clustering may be imprecise."
        )

    if len(clusters) == 1:
        caveats.append("No thematic separation detected; all papers assigned to one cluster.")
    else:
        singleton_count = sum(1 for ps in clusters.values() if len(ps) <= 1)
        if singleton_count > len(clusters) // 2:
            caveats.append(
                f"No clear dominant cluster; {singleton_count} of {len(clusters)} clusters contain only 1 paper."
            )

    return caveats


def _rank_and_trim(
    query: str,
    papers: list[Paper],
    embedder: EmbedderProtocol,
    rank_limit: int | None,
) -> list[Paper]:
    ranked = rank_papers(query, papers, embedder)
    if rank_limit is not None:
        ranked = ranked[:rank_limit]
    return ranked


def _progress(msg: str) -> None:
    print(f"  {msg}", flush=True)


def run_with_analysis(
    query: str,
    project_description: str | None = None,
    search_limit: int = 20,
    search_params: SearchParams | None = None,
    embed_backend: str = "local",
    llm_backend: str = "openrouter",
    rank_limit: int | None = None,
) -> tuple[list[Paper], GapAnalysisReport]:
    sp = search_params or SearchParams()
    llm_context = project_description or query
    _progress("Searching Papers...")
    embedder = make_embedder(embed_backend)
    papers = run(query, search_limit, sp.ss_sort, sp.arxiv_sort, sp.year)
    papers = _rank_and_trim(query, papers, embedder, rank_limit)
    _progress(f"Found {len(papers)} Papers — Analysing Gaps...")
    embeddings = embed_papers(papers, embedder)
    clusters = cluster_papers(papers, embeddings)
    llm = make_llm_client(llm_backend)
    report = build_gap_report(llm_context, clusters, llm)
    report = report.model_copy(update={"caveats": _analysis_caveats(papers, clusters)})
    return papers, report


def run_with_drafts(
    query: str,
    project_description: str | None = None,
    search_limit: int = 20,
    search_params: SearchParams | None = None,
    embed_backend: str = "local",
    llm_backend: str = "openrouter",
    top_k: int = 5,
    embedder: EmbedderProtocol | None = None,
    rank_limit: int | None = None,
) -> tuple[list[Paper], GapAnalysisReport, DraftReport]:
    sp = search_params or SearchParams()
    llm_context = project_description or query
    _progress("Searching Papers...")
    if embedder is None:
        embedder = make_embedder(embed_backend)
    papers = run(query, search_limit, sp.ss_sort, sp.arxiv_sort, sp.year)
    papers = _rank_and_trim(query, papers, embedder, rank_limit)
    _progress(f"Found {len(papers)} Papers — Analysing Gaps...")
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
    search_limit: int = 20,
    search_params: SearchParams | None = None,
    embed_backend: str = "local",
    llm_backend: str = "openrouter",
    top_k: int = 5,
    top_n: int = 10,
    rank_limit: int | None = None,
) -> tuple[list[Paper], GapAnalysisReport, DraftReport, ConferenceMatchReport]:
    embedder = make_embedder(embed_backend)
    papers, gap_report, draft_report = run_with_drafts(
        query, project_description, search_limit, search_params,
        embed_backend, llm_backend, top_k, embedder=embedder, rank_limit=rank_limit,
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


def run_with_latex(
    query: str,
    project_description: str | None = None,
    search_limit: int = 20,
    search_params: SearchParams | None = None,
    embed_backend: str = "local",
    llm_backend: str = "openrouter",
    top_k: int = 5,
    top_n: int = 10,
    rank_limit: int | None = None,
) -> tuple[list[Paper], GapAnalysisReport, DraftReport, ConferenceMatchReport, str, str, dict[str, bytes], ConferenceMatch]:
    """Run the full pipeline and return LaTeX-ready content.

    Returns:
        papers, gap_report, draft_report, match_report,
        populated_tex, bibtex_str, style_files, top_conference
    """
    from src.latex.template_fetcher import fetch_latex_template
    from src.latex.bib_generator import generate_bib_entries
    from src.latex.populator import populate_template

    papers, gap_report, draft_report, match_report = run_with_conference_matching(
        query, project_description, search_limit, search_params,
        embed_backend, llm_backend, top_k, top_n, rank_limit=rank_limit,
    )

    top_conference = match_report.matches[0]
    _progress(f"Fetching LaTeX template for {top_conference.short_name}...")
    template_tex, style_files = fetch_latex_template(top_conference)

    subsections = draft_report.related_work.subsections
    cited_ids: set[str] = set()
    for sub in subsections:
        cited_ids.update(sub.cited_paper_ids)

    bibtex_str, id_to_key = generate_bib_entries(papers, cited_ids)

    abstract_text = draft_report.abstract.full_text
    populated_tex = populate_template(template_tex, abstract_text, subsections, papers, id_to_key)

    return papers, gap_report, draft_report, match_report, populated_tex, bibtex_str, style_files, top_conference
