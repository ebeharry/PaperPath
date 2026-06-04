from __future__ import annotations

import io
import zipfile
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.data_classes import ConferenceMatch

_TIMEOUT = 15
_LINK_KEYWORDS = {"latex", "style", "template", "author", "kit", "submission", "format", ".zip", ".sty", ".cls"}
_SUBPAGES = ["/cfp", "/call-for-papers", "/authors", "/submission", "/paper-submission", "/author-kit"]
_MAIN_TEX_NAMES = {"sample", "template", "main", "paper", "example"}

_FALLBACK_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{cite}
\usepackage{hyperref}
\usepackage{url}

\title{Paper Title}
\author{Author Name}
\date{}

\begin{document}

\maketitle

\begin{abstract}
Abstract goes here.
\end{abstract}

\section{Introduction}

Introduction goes here.

\section{Related Work}

Related work goes here.

\section{Conclusion}

Conclusion goes here.

\bibliography{references}
\bibliographystyle{plain}

\end{document}
"""


def _get(url: str) -> requests.Response | None:
    try:
        r = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r
    except requests.RequestException:
        return None


def _is_template_link(href: str, text: str) -> bool:
    combined = (href + " " + text).lower()
    return any(kw in combined for kw in _LINK_KEYWORDS)


def _candidate_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href: str = tag["href"]
        text: str = tag.get_text(strip=True)
        if _is_template_link(href, text):
            links.append(urljoin(base_url, href))
    return links


def _extract_zip(data: bytes) -> tuple[str | None, dict[str, bytes]]:
    style_files: dict[str, bytes] = {}
    tex_candidates: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                basename = name.split("/")[-1].lower()
                _, _, ext = basename.rpartition(".")
                content = zf.read(name)
                if ext in {"sty", "cls", "bst"}:
                    style_files[basename] = content
                elif ext == "tex" and b"\\documentclass" in content:
                    tex_candidates.append((basename, content))
    except zipfile.BadZipFile:
        return None, {}

    if not tex_candidates:
        return None, style_files

    # prefer files whose stem matches common names
    for basename, content in tex_candidates:
        stem = basename.rpartition(".")[0]
        if any(n in stem for n in _MAIN_TEX_NAMES):
            return content.decode("utf-8", errors="replace"), style_files

    # fall back to the shortest file (most likely a minimal sample)
    tex_candidates.sort(key=lambda x: len(x[1]))
    return tex_candidates[0][1].decode("utf-8", errors="replace"), style_files


def fetch_latex_template(conference: ConferenceMatch) -> tuple[str, dict[str, bytes]]:
    """Return (main_tex_content, {filename: bytes}) for conference style files.

    Falls back to a generic template if nothing is found.
    """
    if not conference.link:
        return _FALLBACK_TEMPLATE, {}

    base = conference.link
    parsed = urlparse(base)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    urls_to_check = [base] + [urljoin(origin, sp) for sp in _SUBPAGES]
    candidate_links: list[str] = []

    for url in urls_to_check:
        resp = _get(url)
        if resp is None:
            continue
        ct = resp.headers.get("Content-Type", "")
        if "zip" in ct or url.endswith(".zip"):
            tex, styles = _extract_zip(resp.content)
            if tex:
                return tex, styles
        if "html" in ct or "text" in ct:
            candidate_links.extend(_candidate_links(resp.text, url))

    # deduplicate preserving order
    seen: set[str] = set()
    unique_links = [l for l in candidate_links if not (l in seen or seen.add(l))]  # type: ignore[func-returns-value]

    zip_links = [l for l in unique_links if ".zip" in l.lower()]
    other_links = [l for l in unique_links if ".zip" not in l.lower()]

    for link in zip_links + other_links:
        resp = _get(link)
        if resp is None:
            continue
        ct = resp.headers.get("Content-Type", "")
        if "zip" in ct or link.endswith(".zip"):
            tex, styles = _extract_zip(resp.content)
            if tex:
                return tex, styles
        if link.endswith(".tex") and b"\\documentclass" in resp.content:
            return resp.text, {}

    return _FALLBACK_TEMPLATE, {}
