from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: Optional[int] = None
    url: Optional[str] = None
    source: str  # "semantic_scholar" | "arxiv"


class ClusterAnalysis(BaseModel):
    cluster_id: int
    paper_ids: list[str]
    what_exists: str
    what_is_contested: str
    what_is_missing: str


class GapAnalysisReport(BaseModel):
    query: str
    clusters: list[ClusterAnalysis]
    overall_what_exists: str
    overall_what_is_contested: str
    overall_what_is_missing: str
