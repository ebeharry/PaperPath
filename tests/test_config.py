import yaml
import pytest
from pydantic import ValidationError

from src.config import PipelineConfig, load_config


def _write_yaml(tmp_path, data: dict) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


def test_load_config_reads_yaml_file(tmp_path):
    path = _write_yaml(tmp_path, {"query": "transformers", "mode": "search"})
    cfg = load_config(path)
    assert cfg.query == "transformers"
    assert cfg.mode == "search"


def test_load_config_defaults_project_description_to_query(tmp_path):
    path = _write_yaml(tmp_path, {"query": "graph neural networks"})
    cfg = load_config(path)
    assert cfg.project_description == "graph neural networks"


def test_load_config_explicit_project_description_preserved(tmp_path):
    path = _write_yaml(tmp_path, {
        "query": "GNNs",
        "project_description": "We propose a novel graph neural network architecture.",
    })
    cfg = load_config(path)
    assert cfg.project_description == "We propose a novel graph neural network architecture."


def test_load_config_missing_query_raises(tmp_path):
    path = _write_yaml(tmp_path, {"mode": "search"})
    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_invalid_mode_raises(tmp_path):
    path = _write_yaml(tmp_path, {"query": "ml", "mode": "invalid"})
    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_defaults_applied(tmp_path):
    path = _write_yaml(tmp_path, {"query": "ml"})
    cfg = load_config(path)
    assert cfg.mode == "draft"
    assert cfg.llm_backend == "openrouter"
    assert cfg.embed_backend == "local"
    assert cfg.search_limit == 10
    assert cfg.year == "2023-"
    assert cfg.top_k == 5
    assert cfg.output is None
    assert cfg.ss_sort is None
    assert cfg.arxiv_sort is None


def test_load_config_all_fields(tmp_path):
    data = {
        "query": "RAG",
        "project_description": "Our project.",
        "mode": "analyse",
        "search_limit": 20,
        "year": "2022-",
        "ss_sort": "citationCount:desc",
        "arxiv_sort": "submittedDate:desc",
        "llm_backend": "anthropic",
        "embed_backend": "openai",
        "top_k": 8,
        "output": "out.json",
    }
    path = _write_yaml(tmp_path, data)
    cfg = load_config(path)
    assert cfg.query == "RAG"
    assert cfg.project_description == "Our project."
    assert cfg.mode == "analyse"
    assert cfg.search_limit == 20
    assert cfg.year == "2022-"
    assert cfg.ss_sort == "citationCount:desc"
    assert cfg.arxiv_sort == "submittedDate:desc"
    assert cfg.llm_backend == "anthropic"
    assert cfg.embed_backend == "openai"
    assert cfg.top_k == 8
    assert cfg.output == "out.json"


def test_pipeline_config_direct_construction():
    cfg = PipelineConfig(query="test")
    assert cfg.project_description == "test"
    assert cfg.mode == "draft"


def test_load_config_match_mode_accepted(tmp_path):
    path = _write_yaml(tmp_path, {"query": "ml", "mode": "match"})
    cfg = load_config(path)
    assert cfg.mode == "match"


def test_load_config_top_n_defaults_to_10(tmp_path):
    path = _write_yaml(tmp_path, {"query": "ml"})
    cfg = load_config(path)
    assert cfg.top_n == 10


def test_load_config_top_n_preserved_from_yaml(tmp_path):
    path = _write_yaml(tmp_path, {"query": "ml", "top_n": 7})
    cfg = load_config(path)
    assert cfg.top_n == 7


def test_load_config_output_defaults_to_none(tmp_path):
    path = _write_yaml(tmp_path, {"query": "ml"})
    cfg = load_config(path)
    assert cfg.output is None


def test_load_config_output_preserved_from_yaml(tmp_path):
    path = _write_yaml(tmp_path, {"query": "ml", "output": "outputs/results.json"})
    cfg = load_config(path)
    assert cfg.output == "outputs/results.json"
