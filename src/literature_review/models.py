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
