import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.literature_review import run
from src.runner import run_with_analysis, run_with_drafts
from src.data_classes import DraftReport, GapAnalysisReport
from src.config import load_config


def _print_papers(papers, *, start: int = 1) -> None:
    print(f"Found {len(papers)} papers\n")
    for i, paper in enumerate(papers, start):
        authors = ", ".join(paper.authors) if paper.authors else "Unknown"
        print(f"{i}. {paper.title} ({paper.year})")
        print(f"   Authors: {authors}")
        print(f"   Source: {paper.source}")
        if len(paper.abstract) > 200:
            print(f"\tAbstract: {paper.abstract[:200]}...")
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


def _print_gap_report(report: GapAnalysisReport, papers) -> None:
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
    parser = argparse.ArgumentParser(description="Run the PaperPath pipeline from a YAML config file.")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if cfg.mode == "draft":
        papers, gap_report, draft_report = run_with_drafts(
            cfg.query,
            project_description=cfg.project_description,
            max_papers=cfg.max_papers,
            ss_sort=cfg.ss_sort,
            arxiv_sort=cfg.arxiv_sort,
            year=cfg.year,
            embed_backend=cfg.embed_backend,
            llm_backend=cfg.llm_backend,
            top_k=cfg.top_k,
        )
        _print_papers(papers)
        _print_gap_report(gap_report, papers)
        _print_draft_report(draft_report)
        combined = {
            "gap_report": json.loads(gap_report.model_dump_json()),
            "draft_report": json.loads(draft_report.model_dump_json()),
        }
        json_out = json.dumps(combined, indent=2)
        if cfg.draft_output:
            Path(cfg.draft_output).parent.mkdir(parents=True, exist_ok=True)
            with open(cfg.draft_output, "w") as f:
                f.write(json_out)
        else:
            print(json_out)

    elif cfg.mode == "analyse":
        papers, report = run_with_analysis(
            cfg.query,
            project_description=cfg.project_description,
            max_papers=cfg.max_papers,
            ss_sort=cfg.ss_sort,
            arxiv_sort=cfg.arxiv_sort,
            year=cfg.year,
            embed_backend=cfg.embed_backend,
            llm_backend=cfg.llm_backend,
        )
        _print_papers(papers)
        _print_gap_report(report, papers)
        json_out = report.model_dump_json(indent=2)
        if cfg.output:
            Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
            with open(cfg.output, "w") as f:
                f.write(json_out)
        else:
            print(json_out)

    else:  # search
        papers = run(
            cfg.query,
            max_papers=cfg.max_papers,
            ss_sort=cfg.ss_sort,
            arxiv_sort=cfg.arxiv_sort,
            year=cfg.year,
        )
        _print_papers(papers)


if __name__ == "__main__":
    main()
