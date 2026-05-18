from src.literature_review.clients import search_arxiv, search_semantic_scholar
from src.literature_review.data_classes import Paper

_SS_SORT_FIELDS = {"paperId", "publicationDate", "citationCount"}
_ARXIV_SORT_FIELDS = {"relevance", "lastUpdatedDate", "submittedDate"}


def run(project_description: str, max_papers: int = 20, sort: str | None = None, year: str | None = None) -> list[Paper]:
    """
    Search Semantic Scholar and arXiv and return combined results.

    max_papers is applied per source, so up to 2 * max_papers papers may be returned.
    Sort fields are routed to the source that owns them: Semantic Scholar accepts
    paperId, publicationDate, citationCount; arXiv accepts relevance, lastUpdatedDate,
    submittedDate. An unrecognized sort field is silently ignored by both sources.
    """
    sort_field = sort.split(":")[0] if sort else None
    ss_sort = sort if sort_field in _SS_SORT_FIELDS else None
    arxiv_sort = sort if sort_field in _ARXIV_SORT_FIELDS else None

    ss_papers = search_semantic_scholar(project_description, limit=max_papers, sort=ss_sort, year=year)
    arxiv_papers = search_arxiv(project_description, limit=max_papers, sort=arxiv_sort, year=year)
    return ss_papers + arxiv_papers
