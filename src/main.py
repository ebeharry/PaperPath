import argparse
import json

from dotenv import load_dotenv
load_dotenv()

from src.literature_review import run
from src.literature_review.review_runner import run_with_analysis
from src.literature_review.data_classes import GapAnalysisReport

_SS_SORT_CHOICES = [
    "paperId:asc", "paperId:desc",
    "publicationDate:asc", "publicationDate:desc",
    "citationCount:asc", "citationCount:desc",
]

_ARXIV_SORT_CHOICES = [
    "relevance:asc", "relevance:desc",
    "lastUpdatedDate:asc", "lastUpdatedDate:desc",
    "submittedDate:asc", "submittedDate:desc",
]

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


def _print_gap_report(report: GapAnalysisReport, papers) -> None:
    id_to_title = {p.paper_id: p.title for p in papers}
    print(f"\n=== Gap Analysis: {report.query} ===\n")
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
    parser = argparse.ArgumentParser(description="Search academic literature.")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--max-papers", type=int, default=10, help="Max papers to return per source (default: 10)")
    parser.add_argument("--ss-sort", choices=_SS_SORT_CHOICES, default=None, help="Sort Semantic Scholar results (e.g. citationCount:desc)")
    parser.add_argument("--arxiv-sort", choices=_ARXIV_SORT_CHOICES, default=None, help="Sort arXiv results (e.g. submittedDate:desc)")
    parser.add_argument("--year", default="2023-", help="Filter by year range (e.g. 2020:2023 or 2023-)")
    parser.add_argument("--analyse", action="store_true", help="Run clustering and gap analysis (Phase 2)")
    parser.add_argument("--embed-backend", choices=["openai", "local"], default="local", help="Embedding backend (default: local)")
    parser.add_argument("--llm-backend", choices=["openai", "anthropic", "openrouter"], default="openai", help="LLM backend for gap analysis (default: openai)")
    parser.add_argument("--output", default=None, help="Write JSON gap analysis report to this file path")
    args = parser.parse_args()

    if args.analyse:
        papers, report = run_with_analysis(
            args.query,
            max_papers=args.max_papers,
            ss_sort=args.ss_sort,
            arxiv_sort=args.arxiv_sort,
            year=args.year,
            embed_backend=args.embed_backend,
            llm_backend=args.llm_backend,
        )
        _print_papers(papers)
        _print_gap_report(report, papers)
        json_out = report.model_dump_json(indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(json_out)
        else:
            print(json_out)
    else:
        papers = run(
            args.query,
            max_papers=args.max_papers,
            ss_sort=args.ss_sort,
            arxiv_sort=args.arxiv_sort,
            year=args.year,
        )
        _print_papers(papers)


if __name__ == "__main__":
    main()
