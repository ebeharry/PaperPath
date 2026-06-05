from __future__ import annotations

import json
from pathlib import Path

import click

from src.data_classes import DraftReport, Paper
from src.literature_review.embedder import make_embedder

from benchmarks.runner import BenchmarkResult, load_gold_entry, run_benchmark

_GOLD_DIR = Path(__file__).parent / "gold"


def _load_gold_entries(gold: str) -> list[dict]:
    if gold == "all":
        paths = sorted(_GOLD_DIR.glob("*.json"))
        if not paths:
            raise click.ClickException(
                f"No gold fixtures found in {_GOLD_DIR}. "
                "Run: python benchmarks/gold/create_gold.py"
            )
        return [load_gold_entry(str(p)) for p in paths]
    path = _GOLD_DIR / f"{gold}.json"
    if not path.exists():
        raise click.ClickException(f"Gold fixture not found: {path}")
    return [load_gold_entry(str(path))]


def _print_table(results: list[BenchmarkResult]) -> None:
    def _fmt(v: float | bool | None) -> str:
        if v is None:
            return "—"
        if isinstance(v, bool):
            return str(v)
        return f"{v:.3f}"

    header = f"{'gold':<32}  {'recall@k':>8}  {'hall_rate':>9}  {'abs_sim':>7}  {'compile':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        name = r.gold_title[:30] if r.gold_title else r.gold_paper_id[:30]
        print(
            f"{name:<32}  "
            f"{_fmt(r.recall_at_k):>8}  "
            f"{_fmt(r.hallucinated_citation_rate):>9}  "
            f"{_fmt(r.abstract_embedding_similarity):>7}  "
            f"{_fmt(r.compile_success):>7}"
        )


@click.command()
@click.option("--gold", default="all", help="Fixture name or 'all'.")
@click.option(
    "--output-dir",
    default=None,
    help="Directory with papers.json, draft.json, paper.tex.",
)
@click.option("--k", default=10, help="k for recall@k.")
@click.option(
    "--embed-backend",
    default="local",
    type=click.Choice(["local", "openai"]),
    help="Embedder for abstract similarity.",
)
def cli(gold: str, output_dir: str | None, k: int, embed_backend: str) -> None:
    entries = _load_gold_entries(gold)

    papers: list[Paper] = []
    draft_report: DraftReport | None = None
    tex_path: str | None = None

    if output_dir:
        out = Path(output_dir)
        papers_path = out / "papers.json"
        if papers_path.exists():
            papers = [Paper(**p) for p in json.loads(papers_path.read_text())]

        draft_path = out / "draft.json"
        if draft_path.exists():
            draft_report = DraftReport.model_validate(
                json.loads(draft_path.read_text())
            )

        tex = out / "paper.tex"
        if tex.exists():
            tex_path = str(tex)

    embedder = make_embedder(embed_backend) if papers or draft_report else None

    results = [
        run_benchmark(entry, papers, draft_report, embedder, tex_path, k)
        for entry in entries
    ]
    _print_table(results)
