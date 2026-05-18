from src.literature_review.models import Paper


def search_semantic_scholar(query: str, limit: int = 20) -> list[Paper]:
    raise NotImplementedError


def search_arxiv(query: str, limit: int = 20) -> list[Paper]:
    raise NotImplementedError
