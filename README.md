# PaperPath
A CS153 final project that provides an agentic pipeline that receives a research project description and retrieves and clusters relevant literature, drafts a positioned paper skeleton, and auto-formats it for the best-matched conference given a specified deadline timeline.

This repo is co-authored with Claude Code, which is used to plan and implement modular components of the code. Generative AI is also used in refining text and descriptions, including this ReadMe, and in brainstorming the scope of the project.

### Literature Review

The literature review queries Semantic Scholar and arXiv for relevant papers. Thank you to arXiv for use of its open access interoperability.

```
python -m src.main --query <query> [--max-papers N] [--sort FIELD:DIR] [--year RANGE]
```

| Argument | Description | Default |
|---|---|---|
| `--query` | Search query string | required |
| `--max-papers` | Maximum number of papers to return **per source** (Semantic Scholar and arXiv); total may be up to 2× this value | `10` |
| `--sort` | Sort order with `:asc` or `:desc` direction. Semantic Scholar fields: `paperId`, `publicationDate`, `citationCount`. arXiv fields: `relevance`, `lastUpdatedDate`, `submittedDate`. Each sort field is routed to its respective source only. | none |
| `--year` | Year filter: exact (`2023`), range (`2020:2023`), or open-ended (`2023-`) | `2023-` |

### Unit Tests

Run unit tests with ```python -m pytest --cov=src --cov-report=term-missing```