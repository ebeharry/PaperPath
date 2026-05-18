import os
import time
import xml.etree.ElementTree as ET

import requests

from src.literature_review.data_classes import Paper

_SEARCH_BASE_URL = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
_SEARCH_FIELDS = "paperId,title,abstract,authors,year,url"
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_TIME_DELAY = 3.0 # Semantic Scholar has a 1 request/second limit, while arXiv has 1 request/3 seconds limist.

_ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
_ARXIV_BATCH_SIZE = 2000
_ARXIV_NS = "http://www.w3.org/2005/Atom"
_ARXIV_OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"

# -------------- GENERAL PURPOSE QUERY CODE -------------- 

def _request_with_retry(url: str, params: dict, headers: dict | None = None) -> requests.Response:
    """
    Query the url with the given params.
    Retry is available for network errors and codes 429 and 500.
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, headers=headers or {}, timeout=30)
        except requests.RequestException:
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(_BACKOFF_BASE ** attempt)
            continue

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            if attempt == _MAX_RETRIES:
                response.raise_for_status()
            retry_after = float(response.headers.get("Retry-After", _BACKOFF_BASE ** attempt))
            time.sleep(retry_after)
            continue

        if response.status_code >= 500:
            if attempt == _MAX_RETRIES:
                response.raise_for_status()
            time.sleep(_BACKOFF_BASE ** attempt)
            continue

        response.raise_for_status()

    raise RuntimeError("unreachable")  # pragma: no cover

# -------------- SEMANTIC SCHOLAR SPECIFIC CODE -------------- 

def _get_semantic_scholar_headers() -> dict:
    """
    Return auth headers if SEMANTIC_SCHOLAR_API_KEY is set, else empty dict.
    An API key is not required to query Semantic Scholar.
    """
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    return {"x-api-key": api_key} if api_key else {}

def _parse_semantic_scholar_paper(data: dict) -> Paper:
    """
    Map a single Semantic Scholar API result dict to a Paper object.
    """
    return Paper(
        paper_id=data["paperId"],
        title=data.get("title") or "",
        abstract=data.get("abstract") or "",
        authors=[a["name"] for a in data.get("authors") or []],
        year=data.get("year"),
        url=data.get("url"),
        source="semantic_scholar",
    )

def search_semantic_scholar(query: str, limit: int = 10, sort: str | None = None, year: str | None = None) -> list[Paper]:
    """
    Search Semantic Scholar and return up to limit Paper objects.
    """
    params: dict = {"query": query, "fields": _SEARCH_FIELDS}
    if sort:
        params["sort"] = sort
    if year:
        params["year"] = year
    papers: list[Paper] = []
    token: str | None = None

    while len(papers) < limit:
        if token:
            params["token"] = token

        body = _request_with_retry(_SEARCH_BASE_URL, params, headers=_get_semantic_scholar_headers()).json()

        for item in body.get("data", []):
            papers.append(_parse_semantic_scholar_paper(item))
            if len(papers) >= limit:
                break

        token = body.get("token")
        if not token:
            break
        time.sleep(_TIME_DELAY)

    return papers

# -------------- ARXIV SPECIFIC CODE --------------

def _build_arxiv_query(query: str, year: str | None) -> str:
    base = f"all:{query}"
    if not year:
        return base
    if ":" in year:
        start, end = year.split(":", 1)
        date_range = f"[{start}01010000 TO {end}12312359]"
    elif year.endswith("-"):
        start = year[:-1]
        date_range = f"[{start}01010000 TO 999912312359]"
    else:
        date_range = f"[{year}01010000 TO {year}12312359]"
    return f"{base} AND submittedDate:{date_range}"

def _build_arxiv_sort_params(sort: str | None) -> dict:
    if not sort:
        return {}
    sort_by, _, direction = sort.partition(":")
    sort_order = "descending" if direction == "desc" else "ascending"
    return {"sortBy": sort_by, "sortOrder": sort_order}

def _parse_arxiv_entry(entry: ET.Element) -> Paper:
    ns = _ARXIV_NS
    raw_id = (entry.findtext(f"{{{ns}}}id") or "").strip()
    paper_id = raw_id.rstrip("/").split("/")[-1] if raw_id else ""
    title = (entry.findtext(f"{{{ns}}}title") or "").strip()
    abstract = (entry.findtext(f"{{{ns}}}summary") or "").strip()
    authors = [
        (el.findtext(f"{{{ns}}}name") or "").strip()
        for el in entry.findall(f"{{{ns}}}author")
    ]
    published = entry.findtext(f"{{{ns}}}published") or ""
    year: int | None = None
    if published:
        try:
            year = int(published[:4])
        except ValueError:
            year = None
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        url=raw_id if raw_id else None,
        source="arxiv",
    )

def search_arxiv(query: str, limit: int = 10, sort: str | None = None, year: str | None = None) -> list[Paper]:
    """
    Search arXiv and return up to limit Paper objects.
    """
    search_query = _build_arxiv_query(query, year)
    sort_params = _build_arxiv_sort_params(sort)
    papers: list[Paper] = []
    start = 0

    while len(papers) < limit:
        batch = min(_ARXIV_BATCH_SIZE, limit - len(papers))
        params: dict = {"search_query": search_query, "start": start, "max_results": batch, **sort_params}

        response = _request_with_retry(_ARXIV_BASE_URL, params)
        root = ET.fromstring(response.text)

        total = int(root.findtext(f"{{{_ARXIV_OPENSEARCH_NS}}}totalResults") or 0)
        entries = root.findall(f"{{{_ARXIV_NS}}}entry")

        if not entries:
            break

        for entry in entries:
            papers.append(_parse_arxiv_entry(entry))
            if len(papers) >= limit:
                break

        start += len(entries)
        if start >= total:
            break

        time.sleep(_TIME_DELAY)

    return papers
