from src.literature_review.clients import search_semantic_scholar
from src.literature_review.data_classes import Paper


def run(project_description: str) -> list[Paper]:
    papers = search_semantic_scholar(project_description)
    return papers
