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
    input: str
    clusters: list[ClusterAnalysis]
    overall_what_exists: str
    overall_what_is_contested: str
    overall_what_is_missing: str


class RelatedWorkSubsection(BaseModel):
    cluster_id: int
    theme: str
    paragraph: str
    cited_paper_ids: list[str]


class RelatedWorkDraft(BaseModel):
    subsections: list[RelatedWorkSubsection]
    full_text: str


class AbstractDraft(BaseModel):
    background: str
    prior_work_summary: str
    gap: str
    proposed_approach: str
    expected_contribution: str
    full_text: str


class DraftReport(BaseModel):
    input: str
    related_work: RelatedWorkDraft
    abstract: AbstractDraft
