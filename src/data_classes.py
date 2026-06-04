from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, model_validator


class Paper(BaseModel):
    model_config = ConfigDict(extra="ignore")
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: int | None = None
    url: str | None = None
    source: Literal["semantic_scholar", "arxiv"]


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


class RawConferenceEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    full_name: str | None = None
    paper_deadline: str
    abstract_deadline: str | None = None
    deadline_timezone: str | None = None
    homepage: str | None = None
    category: str | None = None

    @model_validator(mode="after")
    def _default_full_name(self) -> RawConferenceEntry:
        if not self.full_name:
            self.full_name = self.name
        return self


class ConferenceMatch(BaseModel):
    conference_id: str
    name: str
    short_name: str
    similarity: float
    deadline: str
    abstract_deadline: str | None = None
    link: str | None = None
    subject_areas: list[str]


class ConferenceMatchReport(BaseModel):
    input: str
    matches: list[ConferenceMatch]
    top_n: int
