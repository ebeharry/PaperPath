"""
Run the full pipeline for a gold fixture, then evaluate it.

Usage:
  python benchmarks/run_gold_bench.py attention_is_all_you_need
  python benchmarks/run_gold_bench.py bert
  python benchmarks/run_gold_bench.py rag
  python benchmarks/run_gold_bench.py all

Options (env vars):
  LLM_BACKEND     openrouter | openai | anthropic  (default: openrouter)
  SEARCH_LIMIT    number of papers to fetch per source  (default: 20)
  K               k for recall@k  (default: 10)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_HERE = Path(__file__).parent
_GOLD_DIR = _HERE / "gold"
_ROOT = _HERE.parent

# Ensure project root is on sys.path so `src` is importable when this
# script is invoked directly (e.g. python benchmarks/run_gold_bench.py).
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_gold(name: str) -> list[tuple[str, dict]]:
    if name == "all":
        paths = sorted(_GOLD_DIR.glob("*.json"))
        if not paths:
            print("No gold fixtures found. Run: python benchmarks/gold/create_gold.py")
            sys.exit(1)
        return [(p.stem, json.loads(p.read_text())) for p in paths]
    path = _GOLD_DIR / f"{name}.json"
    if not path.exists():
        print(f"Gold fixture not found: {path}")
        sys.exit(1)
    return [(name, json.loads(path.read_text()))]


def _run_pipeline(name: str, entry: dict, llm_backend: str, search_limit: int) -> Path:
    output_dir = _ROOT / "outputs" / f"bench_{name}"
    config = {
        "query": entry["query"],
        "project_description": entry.get("gold_abstract", ""),
        "mode": "latex",
        "search_limit": search_limit,
        # only search papers published before the gold paper, simulating
        # the literature available at the time of writing
        "year": f"2010:{entry['year']}" if entry.get("year") else None,
        "llm_backend": llm_backend,
        "embed_backend": "local",
        "top_k": 5,
        "top_n": 10,
        "rank_limit": None,
        "output": str(output_dir),
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=_ROOT
    ) as f:
        yaml.dump(config, f)
        config_path = f.name

    print(f"\n{'='*60}")
    print(f"Running pipeline for: {entry['title']}")
    print(f"  query:   {entry['query']}")
    print(f"  output:  {output_dir}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.main", config_path],
            cwd=_ROOT,
        )
        if result.returncode != 0:
            print(f"Pipeline failed for {name} (exit {result.returncode})")
    finally:
        Path(config_path).unlink(missing_ok=True)

    return output_dir


def _run_benchmark(name: str, entry: dict, output_dir: Path, k: int) -> None:
    from src.data_classes import DraftReport, Paper
    from src.literature_review.embedder import make_embedder
    from benchmarks.runner import run_benchmark

    papers_path = output_dir / "papers.json"
    draft_path = output_dir / "draft.json"
    tex_path = output_dir / "paper.tex"

    papers: list[Paper] = []
    if papers_path.exists():
        papers = [Paper(**p) for p in json.loads(papers_path.read_text())]

    draft_report: DraftReport | None = None
    if draft_path.exists():
        draft_report = DraftReport.model_validate(json.loads(draft_path.read_text()))

    embedder = make_embedder("local") if (papers or draft_report) else None
    tex = str(tex_path) if tex_path.exists() else None

    result = run_benchmark(entry, papers, draft_report, embedder, tex, k)

    # Save abstract comparison file
    if draft_report is not None:
        comparison = {
            "gold_title": entry.get("title", ""),
            "gold_abstract": entry.get("gold_abstract", ""),
            "generated_abstract": draft_report.abstract.full_text,
            "generated_abstract_fields": {
                "background": draft_report.abstract.background,
                "prior_work_summary": draft_report.abstract.prior_work_summary,
                "gap": draft_report.abstract.gap,
                "proposed_approach": draft_report.abstract.proposed_approach,
                "expected_contribution": draft_report.abstract.expected_contribution,
            },
            "metrics": {
                f"retrieval_relevance@{k}": result.retrieval_relevance_at_k,
                "hallucinated_citation_rate": result.hallucinated_citation_rate,
                "abstract_embedding_similarity": result.abstract_embedding_similarity,
                "compile_success": result.compile_success,
            },
        }
        (output_dir / "benchmark.json").write_text(json.dumps(comparison, indent=2))

    def _fmt(v) -> str:
        if v is None:
            return "—"
        if isinstance(v, bool):
            return str(v)
        return f"{v:.3f}"

    print(f"\nBenchmark results for: {entry['title']}")
    print(f"  ret_relevance@{k}:  {_fmt(result.retrieval_relevance_at_k)}")
    print(f"  hall_rate:          {_fmt(result.hallucinated_citation_rate)}")
    print(f"  abs_sim:            {_fmt(result.abstract_embedding_similarity)}")
    print(f"  compile:            {_fmt(result.compile_success)}")
    if draft_report:
        print(f"  saved to:    {output_dir}/benchmark.json")


def main() -> None:
    import os

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    name = sys.argv[1]
    llm_backend = os.environ.get("LLM_BACKEND", "openrouter")
    search_limit = int(os.environ.get("SEARCH_LIMIT", "20"))
    k = int(os.environ.get("K", "10"))

    entries = _load_gold(name)
    for stem, entry in entries:
        output_dir = _run_pipeline(stem, entry, llm_backend, search_limit)
        _run_benchmark(stem, entry, output_dir, k)


if __name__ == "__main__":
    main()
