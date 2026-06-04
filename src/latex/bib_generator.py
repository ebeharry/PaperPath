from __future__ import annotations

import re

from src.data_classes import Paper


def _extract_last_name(author: str) -> str:
    author = author.strip()
    if "," in author:
        return author.split(",")[0].strip()
    parts = author.split()
    return parts[-1] if parts else "unknown"


def _make_cite_key(paper: Paper, used: set[str]) -> str:
    last = re.sub(r"[^a-z]", "", _extract_last_name(paper.authors[0]).lower()) if paper.authors else "unknown"
    year = str(paper.year) if paper.year else "0000"
    base = f"{last}{year}"
    key = base
    suffix = ord("a")
    while key in used:
        key = f"{base}{chr(suffix)}"
        suffix += 1
    used.add(key)
    return key


def _escape_bibtex(text: str) -> str:
    return text.replace("{", r"\{").replace("}", r"\}").replace("&", r"\&").replace("%", r"\%")


def generate_bib_entries(
    papers: list[Paper],
    paper_ids: set[str],
) -> tuple[str, dict[str, str]]:
    """Return (bibtex_str, {paper_id: cite_key}) for papers in paper_ids."""
    id_to_paper = {p.paper_id: p for p in papers}
    used_keys: set[str] = set()
    id_to_key: dict[str, str] = {}
    entries: list[str] = []

    for pid in paper_ids:
        paper = id_to_paper.get(pid)
        if paper is None:
            continue
        key = _make_cite_key(paper, used_keys)
        id_to_key[pid] = key

        author_str = " and ".join(paper.authors) if paper.authors else "Unknown"
        title = _escape_bibtex(paper.title)
        lines = [
            f"@misc{{{key},",
            f"  author = {{{_escape_bibtex(author_str)}}},",
            f"  title  = {{{title}}},",
        ]
        if paper.year:
            lines.append(f"  year   = {{{paper.year}}},")
        if paper.url:
            lines.append(f"  howpublished = {{\\url{{{paper.url}}}}},")
        lines.append("}")
        entries.append("\n".join(lines))

    return "\n\n".join(entries), id_to_key
