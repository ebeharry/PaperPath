from __future__ import annotations
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class PipelineConfig(BaseModel):
    query: str
    project_description: str = ""
    mode: Literal["search", "analyse", "draft", "match"] = "draft"
    max_papers: int = 10
    year: str = "2023-"
    ss_sort: str | None = None
    arxiv_sort: str | None = None
    llm_backend: str = "openrouter"
    embed_backend: str = "local"
    top_k: int = 5
    top_n: int = 10
    output: str | None = None

    @model_validator(mode="after")
    def _default_description(self) -> PipelineConfig:
        if not self.project_description:
            self.project_description = self.query
        return self


def load_config(path: str) -> PipelineConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return PipelineConfig(**data)
