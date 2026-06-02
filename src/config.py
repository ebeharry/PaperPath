from __future__ import annotations
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, model_validator


class PipelineConfig(BaseModel):
    query: str
    project_description: str = ""
    mode: Literal["search", "analyse", "draft"] = "draft"
    max_papers: int = 10
    year: str = "2023-"
    ss_sort: Optional[str] = None
    arxiv_sort: Optional[str] = None
    llm_backend: str = "openrouter"
    embed_backend: str = "local"
    top_k: int = 5
    output: Optional[str] = None
    draft_output: Optional[str] = None

    @model_validator(mode="after")
    def _default_description(self) -> PipelineConfig:
        if not self.project_description:
            self.project_description = self.query
        return self


def load_config(path: str) -> PipelineConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return PipelineConfig(**data)
