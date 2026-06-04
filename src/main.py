import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from src.runner import run, run_with_analysis, run_with_drafts, run_with_conference_matching, SearchParams
from src.data_classes import ConferenceMatchReport, DraftReport, GapAnalysisReport, Paper
from src.config import load_config


_DISPLAY_ABSTRACT_LIMIT = 200


def _save(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _papers_json(papers: list[Paper]) -> list[dict]:
    return [p.model_dump() for p in papers]


def _print_papers(papers: list[Paper], *, start: int = 1) -> None:
    print(f"Found {len(papers)} papers\n")
    for i, paper in enumerate(papers, start):
        authors = ", ".join(paper.authors) if paper.authors else "Unknown"
        print(f"{i}. {paper.title} ({paper.year})")
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
            max_papers=cfg.max_papers,
            search_params=sp,
            embed_backend=cfg.embed_backend,
            llm_backend=cfg.llm_backend,
            top_k=cfg.top_k,
        )
        _print_draft_report(draft_report)
        if cfg.output:
            _save(cfg.output, {
                "papers": _papers_json(papers),
                "gap_report": gap_report.model_dump(),
                "draft_report": draft_report.model_dump(),
            })

    elif cfg.mode == "analyse":
        papers, report = run_with_analysis(
            cfg.query,
            project_description=cfg.project_description,
            max_papers=cfg.max_papers,
            search_params=sp,
            embed_backend=cfg.embed_backend,
            llm_backend=cfg.llm_backend,
        )
        _print_gap_report(report, papers)
        if cfg.output:
            _save(cfg.output, {
                "papers": _papers_json(papers),
                "gap_report": report.model_dump(),
            })

    elif cfg.mode == "match":
        papers, gap_report, draft_report, match_report = run_with_conference_matching(
            cfg.query,
            project_description=cfg.project_description,
            max_papers=cfg.max_papers,
            search_params=sp,
            embed_backend=cfg.embed_backend,
            llm_backend=cfg.llm_backend,
            top_k=cfg.top_k,
            top_n=cfg.top_n,
        )
        _print_conference_matches(match_report)
        if cfg.output:
            _save(cfg.output, {
                "papers": _papers_json(papers),
                "gap_report": gap_report.model_dump(),
                "draft_report": draft_report.model_dump(),
                "match_report": match_report.model_dump(),
            })
        else:
            print("(no output configured — set in config to save full results)")

    else:  # search
        papers = run(
            cfg.query,
            max_papers=cfg.max_papers,
            ss_sort=sp.ss_sort,
            arxiv_sort=sp.arxiv_sort,
            year=sp.year,
        )
        _print_papers(papers)
        if cfg.output:
            _save(cfg.output, {"papers": _papers_json(papers)})


if __name__ == "__main__":
    main()
