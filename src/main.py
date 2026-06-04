import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from src.runner import run, run_with_analysis, run_with_drafts, run_with_conference_matching, run_with_latex, SearchParams
from src.data_classes import ConferenceMatchReport, DraftReport, GapAnalysisReport, Paper
from src.config import load_config
from src.literature_review.ranker import rank_papers
from src.literature_review.embedder import make_embedder


_DISPLAY_ABSTRACT_LIMIT = 200


def _save_json(output_dir: str, filename: str, data) -> None:
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_text(output_dir: str, filename: str, content: str) -> None:
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _save_bytes(output_dir: str, filename: str, content: bytes) -> None:
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _papers_json(papers: list[Paper]) -> list[dict]:
    return [p.model_dump() for p in papers]


def _print_caveats(caveats: list[str]) -> None:
    if not caveats:
        return
    print("=== Quality Notes ===")
    for caveat in caveats:
        print(f"  - {caveat}")
    print()


def _search_caveats(papers: list[Paper]) -> list[str]:
    if not papers:
        return []
    caveats: list[str] = []
    sources = {p.source for p in papers}
    if len(sources) == 1:
        if "arxiv" in sources:
            caveats.append("Only arXiv papers found; venue fit is low confidence.")
        else:
            caveats.append("Only Semantic Scholar papers found; may miss recent preprints.")
    no_abstract = sum(1 for p in papers if not p.abstract.strip())
    if no_abstract:
        caveats.append(f"{no_abstract} of {len(papers)} papers have no abstract; scores are estimates.")
    no_year = sum(1 for p in papers if p.year is None)
    if no_year:
        caveats.append(f"{no_year} papers have no year; recency scores set to 0.")
    return caveats


def _print_papers(papers: list[Paper], *, start: int = 1) -> None:
    print(f"Found {len(papers)} papers\n")
    for i, paper in enumerate(papers, start):
        score_str = f"  [score: {paper.final_score:.3f}]" if paper.final_score is not None else ""
        authors = ", ".join(paper.authors) if paper.authors else "Unknown"
        print(f"{i}. {paper.title} ({paper.year}){score_str}")
        print(f"   Authors: {authors}")
        print(f"   Source: {paper.source}")
        if len(paper.abstract) > _DISPLAY_ABSTRACT_LIMIT:
            print(f"\tAbstract: {paper.abstract[:_DISPLAY_ABSTRACT_LIMIT]}...")
        else:
            print(f"\tAbstract: {paper.abstract}")
        if paper.url:
            print(f"\tURL: {paper.url}")
        print()


def _print_draft_report(draft: DraftReport) -> None:
    print(f"\n=== Related Work Draft: {draft.input} ===\n")
    for sub in draft.related_work.subsections:
        print(f"--- {sub.theme} (Cluster {sub.cluster_id}) ---")
        print(sub.paragraph)
        print()
    print("=== Abstract Draft ===")
    print(draft.abstract.full_text)
    print()


def _print_conference_matches(report: ConferenceMatchReport) -> None:
    print(f"\n=== Conference Matches ({len(report.matches)}/{report.top_n}) ===\n")
    for i, m in enumerate(report.matches, 1):
        areas = ", ".join(m.subject_areas) if m.subject_areas else "—"
        deadline = m.deadline or "TBD"
        sim = f"{m.similarity:.3f}"
        link = f"  {m.link}" if m.link else ""
        print(f"{i:2}. {m.short_name:<14} {areas:<8} deadline {deadline:<12} sim {sim}{link}")
    print()


def _print_gap_report(report: GapAnalysisReport, papers: list[Paper]) -> None:
    id_to_title = {p.paper_id: p.title for p in papers}
    print(f"\n=== Gap Analysis: {report.input} ===\n")
    for cluster in report.clusters:
        titles = ", ".join(id_to_title.get(pid, pid) for pid in cluster.paper_ids)
        print(f"Cluster {cluster.cluster_id} ({len(cluster.paper_ids)} papers):")
        print(f"  Papers: {titles}")
        print(f"  What Exists: {cluster.what_exists}")
        print(f"  What Is Contested: {cluster.what_is_contested}")
        print(f"  What Is Missing: {cluster.what_is_missing}")
        print()
    print("=== Overall ===")
    print(f"  What Exists: {report.overall_what_exists}")
    print(f"  What Is Contested: {report.overall_what_is_contested}")
    print(f"  What Is Missing: {report.overall_what_is_missing}")
    print()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the PaperPath pipeline from a YAML config file.")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"error: invalid YAML in {args.config}: {e}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"error: invalid config in {args.config}:\n{e}", file=sys.stderr)
        sys.exit(1)

    sp = SearchParams(ss_sort=cfg.ss_sort, arxiv_sort=cfg.arxiv_sort, year=cfg.year)

    if cfg.mode == "draft":
        papers, gap_report, draft_report = run_with_drafts(
            cfg.query,
            project_description=cfg.project_description,
            search_limit=cfg.search_limit,
            search_params=sp,
            embed_backend=cfg.embed_backend,
            llm_backend=cfg.llm_backend,
            top_k=cfg.top_k,
            rank_limit=cfg.rank_limit,
        )
        print("  Draft complete.", flush=True)
        _print_caveats(gap_report.caveats + draft_report.caveats)
        if cfg.output:
            _save_json(cfg.output, "papers.json", _papers_json(papers))
            _save_json(cfg.output, "gap_analysis.json", gap_report.model_dump())
            _save_json(cfg.output, "draft.json", draft_report.model_dump())

    elif cfg.mode == "analyse":
        papers, report = run_with_analysis(
            cfg.query,
            project_description=cfg.project_description,
            search_limit=cfg.search_limit,
            search_params=sp,
            embed_backend=cfg.embed_backend,
            llm_backend=cfg.llm_backend,
            rank_limit=cfg.rank_limit,
        )
        _print_gap_report(report, papers)
        _print_caveats(report.caveats)
        if cfg.output:
            _save_json(cfg.output, "papers.json", _papers_json(papers))
            _save_json(cfg.output, "gap_analysis.json", report.model_dump())

    elif cfg.mode == "match":
        papers, gap_report, draft_report, match_report = run_with_conference_matching(
            cfg.query,
            project_description=cfg.project_description,
            search_limit=cfg.search_limit,
            search_params=sp,
            embed_backend=cfg.embed_backend,
            llm_backend=cfg.llm_backend,
            top_k=cfg.top_k,
            top_n=cfg.top_n,
            rank_limit=cfg.rank_limit,
        )
        _print_conference_matches(match_report)
        _print_caveats(gap_report.caveats + draft_report.caveats)
        if cfg.output:
            _save_json(cfg.output, "papers.json", _papers_json(papers))
            _save_json(cfg.output, "gap_analysis.json", gap_report.model_dump())
            _save_json(cfg.output, "draft.json", draft_report.model_dump())
            _save_json(cfg.output, "conferences.json", match_report.model_dump())
        else:
            print("(no output configured — set in config to save full results)")

    elif cfg.mode == "latex":
        papers, gap_report, draft_report, match_report, populated_tex, bibtex_str, style_files, top_conf = run_with_latex(
            cfg.query,
            project_description=cfg.project_description,
            search_limit=cfg.search_limit,
            search_params=sp,
            embed_backend=cfg.embed_backend,
            llm_backend=cfg.llm_backend,
            top_k=cfg.top_k,
            top_n=cfg.top_n,
            rank_limit=cfg.rank_limit,
        )
        print(f"\nTop conference: {top_conf.name} ({top_conf.short_name})")
        _print_conference_matches(match_report)
        _print_caveats(gap_report.caveats + draft_report.caveats)
        if cfg.output:
            _save_json(cfg.output, "papers.json", _papers_json(papers))
            _save_json(cfg.output, "gap_analysis.json", gap_report.model_dump())
            _save_json(cfg.output, "draft.json", draft_report.model_dump())
            _save_json(cfg.output, "conferences.json", match_report.model_dump())
            _save_text(cfg.output, "paper.tex", populated_tex)
            _save_text(cfg.output, "references.bib", bibtex_str)
            for fname, fbytes in style_files.items():
                _save_bytes(cfg.output, fname, fbytes)
            print(f"Output written to: {Path(cfg.output).resolve()}/")
            print(f"  paper.tex         ({len(populated_tex)} chars)")
            print(f"  references.bib    ({len(bibtex_str)} chars)")
            for fname in style_files:
                print(f"  {fname}")

    else:  # search
        papers = run(
            cfg.query,
            search_limit=cfg.search_limit,
            ss_sort=sp.ss_sort,
            arxiv_sort=sp.arxiv_sort,
            year=sp.year,
        )
        embedder = make_embedder(cfg.embed_backend)
        ranked_papers = rank_papers(cfg.query, papers, embedder)
        _print_papers(ranked_papers)
        _print_caveats(_search_caveats(ranked_papers))
        if cfg.output:
            _save_json(cfg.output, "papers.json", _papers_json(ranked_papers))


if __name__ == "__main__":
    main()
