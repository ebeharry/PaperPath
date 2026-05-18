# PaperPath
A CS153 final project that provides an agentic pipeline that receives a research project description and retrieves and clusters relevant literature, drafts a positioned paper skeleton, and auto-formats it for the best-matched conference given a specified deadline timeline.

This repo is co-authored with Claude Code, which is used to plan and implement modular components of the code. Generative AI is also used in refining text and descriptions, including this ReadMe, and in brainstorming the scope of the project.

### Literature Review

```
python -m src.main --query <query> [--max-papers N] [--sort FIELD:DIR] [--year RANGE]
```

| Argument | Description | Default |
|---|---|---|
| `--query` | Search query string | required |
| `--max-papers` | Maximum number of papers to return | `10` |
| `--sort` | Sort order: `paperId`, `publicationDate`, or `citationCount`, with `:asc` or `:desc` appended to specifcy the direction (ascending or descending) | none |
| `--year` | Year filter: exact (`2023`), range (`2020:2023`), or open-ended (`2023-`) | `2023-` | none

CLI Command:
`python -m src.main --query "your query here" --max-papers <max_papers> --year "year range" --sort "sort_option:dir"`