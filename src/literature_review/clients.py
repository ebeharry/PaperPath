import os
import time

import requests

from src.literature_review.data_classes import Paper

_SEARCH_BASE_URL = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
_SEARCH_FIELDS = "paperId,title,abstract,authors,year,url"
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_TIME_DELAY = 1 # Without API, rates are limited to 1/sec

# -------------- GENERAL PURPOSE QUERY CODE -------------- 

def _request_with_retry(url: str, params: dict) -> dict:
    """
    Query the url with the given params.
    Retry is available for network errors and codes 429 and 500.
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, headers=_get_semantic_scholar_headers(), timeout=30)
        except requests.RequestException:
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(_BACKOFF_BASE ** attempt)
            continue

        if response.status_code == 200:
            return response.json()

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

    raise RuntimeError("unreachable")

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

def search_semantic_scholar(query: str, limit: int = 20) -> list[Paper]:
    """
    Search Semantic Scholar and return up to limit Paper objects.
    """
    params: dict = {"query": query, "fields": _SEARCH_FIELDS}
    papers: list[Paper] = []
    token: str | None = None

    while len(papers) < limit:
        if token:
            params["token"] = token

        body = _request_with_retry(_SEARCH_BASE_URL, params)

        for item in body.get("data", []):
            papers.append(_parse_semantic_scholar_paper(item))
            if len(papers) >= limit:
                break

        token = body.get("token")
        if not token:
            break
        time.delay(_TIME_DELAY)

    return papers

# -------------- ARXIV SPECIFIC CODE -------------- 

def search_arxiv(query: str, limit: int = 20) -> list[Paper]:
    """
    Search arXiv and return up to limit Paper objects.
    """
    raise NotImplementedError
