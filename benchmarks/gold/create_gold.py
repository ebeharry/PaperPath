"""
One-time script to generate gold fixture JSON files from Semantic Scholar.
Usage: python benchmarks/gold/create_gold.py

Writes resnet.json, bert.json, vit.json to this directory.
Requires network access. Set SEMANTIC_SCHOLAR_API_KEY for higher rate limits.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

_HERE = Path(__file__).parent
_BASE = "https://api.semanticscholar.org/graph/v1/paper"
_DELAY = 5.0
_MAX_RETRIES = 6
_BACKOFF_BASE = 30.0

_TARGETS = [
    {
        # ResNet — He et al. 2015. Cites AlexNet, VGGNet, BatchNorm, Dropout, etc.
        # Query targets background literature, not ResNet-specific terminology.
        "paper_id": "2c03df8b48bf3fa39054345bafabfeff15bfd11d",
        "query": "convolutional neural networks image recognition deep learning",
        "output_file": "resnet.json",
    },
    {
        # BERT — Devlin et al. 2018. Cites ELMo, word2vec, language models, etc.
        "paper_id": "df2b0e26d0599ce3e70df8a9da02e51594e0e992",
        "query": "language model pre-training natural language processing transformer",
        "output_file": "bert.json",
    },
    {
        # ViT — Dosovitskiy et al. 2020. Cites ResNet, EfficientNet, BiT, BERT, etc.
        # Query targets pre-ViT image classification literature.
        "paper_id": "268d347e8a55b5eb82fb5e7d2f800e33c75ab18a",
        "query": "image classification transformer neural network convolutional deep learning",
        "output_file": "vit.json",
    },
]


def _headers() -> dict[str, str]:
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    return {"x-api-key": key} if key else {}


def _get(url: str, params: dict) -> dict:
    for attempt in range(_MAX_RETRIES + 1):
        resp = requests.get(url, params=params, headers=_headers(), timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            if attempt == _MAX_RETRIES:
                print(
                    "\n\nStill rate limited after all retries. "
                    "Get a free API key at https://www.semanticscholar.org/product/api "
                    "and set SEMANTIC_SCHOLAR_API_KEY=<key> before running again."
                )
                resp.raise_for_status()
            wait = float(resp.headers.get("Retry-After", _BACKOFF_BASE * (attempt + 1)))
            print(f"\n  rate limited, waiting {wait:.0f}s ...", end=" ", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError("unreachable")  # pragma: no cover


def fetch_gold(paper_id: str, query: str) -> dict:
    paper_url = f"{_BASE}/{paper_id}"
    paper_data = _get(paper_url, {"fields": "paperId,title,abstract,year"})
    time.sleep(_DELAY)

    refs_url = f"{_BASE}/{paper_id}/references"
    refs_data = _get(refs_url, {"fields": "paperId", "limit": 100})
    time.sleep(_DELAY)

    ref_ids = [
        r["citedPaper"]["paperId"]
        for r in refs_data.get("data", [])
        if r.get("citedPaper", {}).get("paperId")
    ]

    return {
        "paper_id": paper_data["paperId"],
        "title": paper_data["title"],
        "year": paper_data.get("year"),
        "query": query,
        "gold_abstract": paper_data.get("abstract", ""),
        "gold_references": ref_ids,
    }


def main() -> None:
    for target in _TARGETS:
        out_path = _HERE / target["output_file"]
        print(f"Fetching {target['paper_id']} ...", end=" ", flush=True)
        entry = fetch_gold(target["paper_id"], target["query"])
        # Sanity check: title should be non-empty
        if not entry["title"]:
            print(f"WARNING: empty title for {target['paper_id']}, skipping")
            continue
        out_path.write_text(json.dumps(entry, indent=2))
        print(f"done ({len(entry['gold_references'])} refs) -> {out_path.name}")


if __name__ == "__main__":
    main()
