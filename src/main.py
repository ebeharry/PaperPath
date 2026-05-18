import argparse

from src.literature_review import run

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


def main():
    parser = argparse.ArgumentParser(description="Search academic literature.")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--max-papers", type=int, default=10, help="Max papers to return per source (default: 10)")
    parser.add_argument("--ss-sort", choices=_SS_SORT_CHOICES, default=None, help="Sort Semantic Scholar results (e.g. citationCount:desc)")
    parser.add_argument("--arxiv-sort", choices=_ARXIV_SORT_CHOICES, default=None, help="Sort arXiv results (e.g. submittedDate:desc)")
    parser.add_argument("--year", default="2023-", help="Filter by year range (e.g. 2020:2023 or 2023-)")
    args = parser.parse_args()

    papers = run(
        args.query,
        max_papers=args.max_papers,
        ss_sort=args.ss_sort,
        arxiv_sort=args.arxiv_sort,
        year=args.year,
    )

    print(f"Found {len(papers)} papers\n")
    for i, paper in enumerate(papers, 1):
        authors = ", ".join(paper.authors) if paper.authors else "Unknown"
        print(f"{i}. {paper.title} ({paper.year})")
        print(f"   Authors: {authors}")
        if len(paper.abstract) > 200:
            print(f"\tAbstract: {paper.abstract[:200]}...")
        else:
            print(f"\tAbstract: {paper.abstract}")
        if paper.url:
            print(f"\tURL: {paper.url}")
        print()


if __name__ == "__main__":
    main()
