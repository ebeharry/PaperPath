from src.literature_review.clients import search_arxiv, search_semantic_scholar
from src.literature_review.data_classes import Paper
from paperpath.shared.models import Cluster


def run(project_description: str) -> list[Cluster]:
    papers: list[Paper] = (
        search_semantic_scholar(project_description)
        + search_arxiv(project_description)
    )
    raise NotImplementedError  # clustering to be implemented
