from src.literature_review.clients import search_semantic_scholar
from src.literature_review.data_classes import Paper


def run(project_description: str, max_papers: int = 20) -> list[Paper]:
    papers = search_semantic_scholar(project_description, limit=max_papers)
    return papers
