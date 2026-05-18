from src.literature_review.clients import search_arxiv, search_semantic_scholar
from src.literature_review.data_classes import Paper

def run(project_description: str, max_papers: int = 20, ss_sort: str | None = None, arxiv_sort: str | None = None, year: str | None = None) -> list[Paper]:
    """
    Search Semantic Scholar and arXiv and return combined results.

    max_papers is applied per source, so up to 2 * max_papers papers may be returned.
    """
    ss_papers = search_semantic_scholar(project_description, limit=max_papers, sort=ss_sort, year=year)
    arxiv_papers = search_arxiv(project_description, limit=max_papers, sort=arxiv_sort, year=year)
    seen: set[str] = set()
    result: list[Paper] = []
    for paper in ss_papers + arxiv_papers:
        key = paper.title.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(paper)
    return result
