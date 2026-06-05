# PaperPath

An agentic pipeline that turns a research query into a submission-ready LaTeX paper skeleton — abstract, related work, and bibliography — formatted for the best-matched upcoming conference.

## The Problem

The first week of writing a research paper is rarely spent writing. It's spent reading: crawling through dozens of papers, identifying what exists, what's contested, and where your work fits. Then formatting a related work section, hunting for a conference, downloading its template, and wrestling LaTeX from scratch.

PaperPath automates that entire front-end. Give it your research idea and it handles retrieval, clustering, gap analysis, drafting, conference matching, and template population in a single command.

## Pipeline

```
Query + Description
  └─ Search (Semantic Scholar + arXiv, parallel)
       └─ Rank (hybrid score: relevance + recency + citations)
            └─ Cluster (agglomerative, silhouette k-selection)
                 └─ Gap Analysis (LLM per cluster: what exists / contested / missing)
                      └─ Draft (abstract + related work via RAG)
                           └─ Conference Match (embedding similarity against scope)
                                └─ LaTeX Output (.tex + .bib + .sty)
```

### Ranking

Papers are scored with a weighted hybrid:

```
score = 0.35 × title_sim + 0.35 × abstract_sim + 0.20 × recency + 0.10 × citations
```

Similarity uses L2-normalized embeddings (local sentence-transformers or OpenAI). Recency and citation count are min-max normalized within the result set.

### Clustering

Title + abstract embeddings are L2-normalized and passed to agglomerative clustering (Ward linkage). k is auto-selected in the range [2, 8] by silhouette score.

### Gap Analysis

An LLM reads each cluster and returns three fields: what the cluster establishes, what's contested within it, and what's missing or underexplored.

### Drafting (RAG)

The gap statements are embedded and used to retrieve the top-k most relevant papers per cluster. Those papers ground the LLM when writing each related work subsection and when drafting the structured abstract.

### Conference Matching

The generated abstract is embedded and compared via cosine similarity against conference scope strings. Results are filtered to conferences with future deadlines and subject-area relevance, then ranked by similarity.

## Modes

| Mode | What it runs | What it saves |
|------|-------------|---------------|
| `search` | retrieval + ranking | `papers.json` |
| `analyse` | + clustering + gap analysis | + `gap_analysis.json` |
| `draft` | + abstract + related work | + `draft.json` |
| `match` | + conference ranking | + `conferences.json` |
| `latex` | full pipeline | + `paper.tex`, `references.bib`, style files |

## Setup

```bash
git clone https://github.com/ebeharry/PaperPath
cd PaperPath
pip install -r requirements.txt
```

Copy the environment template and fill in the keys you need:

```bash
cp .env.example .env
```

| Variable | Required for | Where to get it |
|----------|-------------|-----------------|
| `OPENROUTER_API_KEY` | `llm_backend: openrouter` (default) | openrouter.ai |
| `OPENAI_API_KEY` | `llm_backend: openai` or `embed_backend: openai` | platform.openai.com |
| `ANTHROPIC_API_KEY` | `llm_backend: anthropic` | console.anthropic.com |
| `SEMANTIC_SCHOLAR_API_KEY` | higher rate limits on Semantic Scholar | semanticscholar.org/product/api |

`embed_backend: local` (default) uses sentence-transformers and requires no API key.

## Usage

```yaml
# config.yaml
query: "retrieval augmented generation"

project_description: |
  We are developing a novel retrieval-augmented generation system that addresses
  hallucination in large language models. Our approach combines dense retrieval
  with a re-ranking step that uses query-document relevance scores to improve
  factual accuracy. We target the domain of scientific literature summarization,
  where current RAG systems struggle with long-context reasoning and citation
  faithfulness.

mode: latex         # search | analyse | draft | match | latex
search_limit: 15    # papers fetched per source; up to 2× after dedup
year: "2022-"       # exact (2023), range (2020:2023), or open-ended (2022-)
llm_backend: openrouter
embed_backend: local
top_k: 5            # papers retrieved per cluster for LLM context (RAG)
top_n: 10           # conferences to surface in match/latex modes
rank_limit: null    # trim to top N after ranking; null keeps all
output: outputs/my_run
```

```bash
python -m src.main config.yaml
```

Progress is printed to stdout:

```
  Searching Papers...
  Found 24 Papers — Analysing Gaps...
  Drafting Abstract and Related Work...
  Matching Conferences...
  Fetching LaTeX template for NeurIPS...
```

## Output Files

| File | Description |
|------|-------------|
| `papers.json` | Ranked paper list with titles, abstracts, authors, year, source |
| `gap_analysis.json` | Per-cluster and overall gap analysis (exists / contested / missing) |
| `draft.json` | Structured abstract fields + related work subsections with citation IDs |
| `conferences.json` | Ranked conference matches with deadlines and submission links |
| `paper.tex` | Populated LaTeX document (abstract + related work + `\cite{}` calls) |
| `references.bib` | BibTeX entries for all cited papers |
| `*.sty` / `*.cls` | Conference style files fetched from the venue website |

## Evaluation

The benchmark suite measures pipeline quality against gold-standard papers with known references.

### Metrics

| Metric | Description |
|--------|-------------|
| **Retrieval relevance@K** | Mean cosine similarity between the top-K retrieved paper embeddings and the gold abstract embedding — measures how topically relevant the retrieved set is to the target paper |
| **Hallucinated citation rate** | Fraction of cited paper IDs in the draft that are not in the retrieved set |
| **Abstract embedding similarity** | Cosine similarity between the generated abstract and the published abstract |
| **LaTeX compile success** | Whether `paper.tex` compiles without errors via `pdflatex` |

### Gold Cases

Gold entries are in `benchmarks/gold/` and include the paper ID, published abstract, and ground-truth reference list. Current gold cases: **BERT**, **ResNet**, **ViT**.

### Running Benchmarks

```bash
SEMANTIC_SCHOLAR_API_KEY=<key> python benchmarks/run_gold_bench.py all
# or a single fixture:
SEMANTIC_SCHOLAR_API_KEY=<key> python benchmarks/run_gold_bench.py bert
```

## Limitations

- Abstract and related work quality depend on the LLM backend and model. Output should be treated as a first draft, not a finished product.
- Conference template scraping can fail for venues with JavaScript-heavy submission portals; the pipeline falls back to a generic ACM-style template.
- Retrieval recall is bounded by Semantic Scholar and arXiv coverage. Papers behind paywalls or in non-indexed venues will not appear.
- Citation correctness is verified by paper ID, not by reading the actual paper. The LLM may misattribute a claim to a correctly retrieved paper.
- The ML-Conferences deadline list covers primarily ML/AI venues.

## Tests

```bash
python -m pytest --cov=src --cov-report=term-missing
```

13 test modules covering all pipeline stages.

## AI Disclosure

This project was co-authored by **Claude Code** (Anthropic), used for planning, implementing, and reviewing modular components. Generative AI was also used to refine descriptions and brainstorm project scope. All code has been reviewed and validated by the author.

The pipeline itself uses LLM backends (OpenAI, Anthropic, or OpenRouter) as a core functional component for gap analysis and draft generation.

## Credits

- [Semantic Scholar API](https://api.semanticscholar.org/) — paper search and metadata
- [arXiv](https://arxiv.org/) — open access preprint search; thank you to arXiv for use of its open access interoperability
- [ai-deadlines](https://github.com/paperswithcode/ai-deadlines) — conference deadline data
- [sentence-transformers](https://www.sbert.net/) — local embedding model
- [scikit-learn](https://scikit-learn.org/) — agglomerative clustering and silhouette scoring
