import argparse

from src.literature_review import run


def main():
    parser = argparse.ArgumentParser(description="Search academic literature.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-papers", type=int, default=20, help="Max papers to return (default: 20)")
    args = parser.parse_args()

    papers = run(args.query, max_papers=args.max_papers)

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
