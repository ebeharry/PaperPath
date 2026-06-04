from __future__ import annotations

import re

from src.data_classes import Paper, RelatedWorkSubsection


def _build_name_year_map(
    subsection: RelatedWorkSubsection,
    papers: list[Paper],
    id_to_key: dict[str, str],
) -> dict[tuple[str, str], str]:
    """Map (lowercase_last_name, year_str) → cite_key for cited papers in a subsection."""
    id_to_paper = {p.paper_id: p for p in papers}
    mapping: dict[tuple[str, str], str] = {}
    for pid in subsection.cited_paper_ids:
        paper = id_to_paper.get(pid)
        key = id_to_key.get(pid)
        if paper is None or key is None:
            continue
        if not paper.authors:
            continue
        first_author = paper.authors[0].strip()
        # handle "Last, First" and "First Last"
        if "," in first_author:
            last = first_author.split(",")[0].strip()
        else:
            parts = first_author.split()
            last = parts[-1] if parts else first_author
        year = str(paper.year) if paper.year else ""
        mapping[(last.lower(), year)] = key
    return mapping


def _replace_citations(text: str, name_year_map: dict[tuple[str, str], str]) -> str:
    """Replace [Name et al., YYYY] and [Name, YYYY] with \\cite{key}."""

    def _sub(m: re.Match) -> str:
        name_part = m.group(1).strip()
        year_part = m.group(2).strip()
        # strip "et al." suffix
        name_clean = re.sub(r"\s+et\s+al\.?$", "", name_part, flags=re.IGNORECASE).strip()
        key = name_year_map.get((name_clean.lower(), year_part))
        if key:
            return rf"\cite{{{key}}}"
        return m.group(0)

    return re.sub(r"\[([^\[\],]+(?:et al\.)?)[,\s]+(\d{4})\]", _sub, text)


def _build_related_work(
    subsections: list[RelatedWorkSubsection],
    papers: list[Paper],
    id_to_key: dict[str, str],
) -> str:
    parts = [r"\section{Related Work}", ""]
    for sub in subsections:
        name_year_map = _build_name_year_map(sub, papers, id_to_key)
        paragraph = _replace_citations(sub.paragraph, name_year_map)
        parts.append(rf"\subsection{{{sub.theme}}}")
        parts.append("")
        parts.append(paragraph)
        parts.append("")
    return "\n".join(parts)


def _ensure_cite_package(tex: str) -> str:
    if r"\usepackage{cite}" in tex or r"\usepackage[" in tex and "cite" in tex:
        return tex
    # insert after \documentclass line
    return re.sub(
        r"(\\documentclass(?:\[[^\]]*\])?\{[^}]+\})",
        lambda m: m.group(1) + "\n\\usepackage{cite}\n\\usepackage{url}",
        tex,
        count=1,
    )


def populate_template(
    template_tex: str,
    abstract_text: str,
    subsections: list[RelatedWorkSubsection],
    papers: list[Paper],
    id_to_key: dict[str, str],
) -> str:
    tex = template_tex

    # 1. Replace or insert abstract
    abstract_block = f"\\begin{{abstract}}\n{abstract_text}\n\\end{{abstract}}"
    if re.search(r"\\begin\{abstract\}", tex):
        tex = re.sub(
            r"\\begin\{abstract\}.*?\\end\{abstract\}",
            lambda _: abstract_block,
            tex,
            flags=re.DOTALL,
        )
    else:
        tex = tex.replace(r"\begin{document}", r"\begin{document}" + "\n\n" + abstract_block, 1)

    # 2. Build related work section with converted citations
    related_work = _build_related_work(subsections, papers, id_to_key)

    # 3. Replace existing \section{Related Work} if present, else insert after Introduction
    existing_rw = re.search(r"\\section\{Related Work\}", tex, re.IGNORECASE)
    if existing_rw:
        next_section = re.search(r"(\\section\{|\\end\{document\})", tex[existing_rw.end():])
        if next_section:
            replace_end = existing_rw.end() + next_section.start()
            tex = tex[:existing_rw.start()] + related_work + "\n\n" + tex[replace_end:]
        else:
            tex = tex[:existing_rw.start()] + related_work + "\n"
    else:
        intro_match = re.search(r"(\\section\{Introduction\})", tex, re.IGNORECASE)
        if intro_match:
            next_section = re.search(r"(\\section\{|\\end\{document\})", tex[intro_match.end():])
            if next_section:
                insert_pos = intro_match.end() + next_section.start()
                tex = tex[:insert_pos] + "\n\n" + related_work + "\n\n" + tex[insert_pos:]
            else:
                tex = tex.replace(r"\end{document}", related_work + "\n\n" + r"\end{document}", 1)
        else:
            tex = tex.replace(r"\end{document}", related_work + "\n\n" + r"\end{document}", 1)

    # 4. Add bibliography before \end{document} if not already present
    if r"\bibliography{" not in tex:
        tex = tex.replace(
            r"\end{document}",
            "\\bibliography{references}\n\\bibliographystyle{plain}\n\n\\end{document}",
            1,
        )

    # 5. Ensure cite package in preamble
    tex = _ensure_cite_package(tex)

    return tex
