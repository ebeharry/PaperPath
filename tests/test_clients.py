import pytest
import requests
from unittest.mock import MagicMock, patch, call

from src.literature_review.clients import (
    _get_semantic_scholar_headers,
    _parse_semantic_scholar_paper,
    _request_with_retry,
    search_semantic_scholar,
)
from src.literature_review.data_classes import Paper


# ---------------------------------------------------------------------------
# _get_semantic_scholar_headers
# ---------------------------------------------------------------------------

def test_headers_with_api_key(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key-123")
    assert _get_semantic_scholar_headers() == {"x-api-key": "test-key-123"}


def test_headers_without_api_key(monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    assert _get_semantic_scholar_headers() == {}


# ---------------------------------------------------------------------------
# _parse_semantic_scholar_paper
# ---------------------------------------------------------------------------

_FULL_DATA = {
    "paperId": "abc123",
    "title": "Attention Is All You Need",
    "abstract": "We propose the Transformer.",
    "authors": [{"name": "Vaswani"}, {"name": "Shazeer"}],
    "year": 2017,
    "url": "https://arxiv.org/abs/1706.03762",
}


def test_parse_full_paper():
    paper = _parse_semantic_scholar_paper(_FULL_DATA)
    assert paper == Paper(
        paper_id="abc123",
        title="Attention Is All You Need",
        abstract="We propose the Transformer.",
        authors=["Vaswani", "Shazeer"],
        year=2017,
        url="https://arxiv.org/abs/1706.03762",
        source="semantic_scholar",
    )


def test_parse_missing_title():
    data = {**_FULL_DATA, "title": None}
    assert _parse_semantic_scholar_paper(data).title == ""


def test_parse_missing_abstract():
    data = {**_FULL_DATA, "abstract": None}
    assert _parse_semantic_scholar_paper(data).abstract == ""


def test_parse_missing_authors():
    data = {**_FULL_DATA, "authors": None}
    assert _parse_semantic_scholar_paper(data).authors == []


def test_parse_missing_year():
    data = {k: v for k, v in _FULL_DATA.items() if k != "year"}
    assert _parse_semantic_scholar_paper(data).year is None


def test_parse_missing_url():
    data = {k: v for k, v in _FULL_DATA.items() if k != "url"}
    assert _parse_semantic_scholar_paper(data).url is None


def test_parse_source_always_semantic_scholar():
    assert _parse_semantic_scholar_paper(_FULL_DATA).source == "semantic_scholar"


# ---------------------------------------------------------------------------
# _request_with_retry
# ---------------------------------------------------------------------------

def _mock_response(status_code: int, json_data: dict = None, headers: dict = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


@patch("src.literature_review.clients.requests.get")
def test_request_returns_json_on_200(mock_get):
    mock_get.return_value = _mock_response(200, {"data": [{"paperId": "x"}]})
    result = _request_with_retry("http://example.com", {})
    assert result == {"data": [{"paperId": "x"}]}


@patch("src.literature_review.clients.time.sleep")
@patch("src.literature_review.clients.requests.get")
def test_request_retries_on_429_and_succeeds(mock_get, mock_sleep):
    mock_get.side_effect = [
        _mock_response(429, headers={"Retry-After": "0.1"}),
        _mock_response(200, {"data": []}),
    ]
    result = _request_with_retry("http://example.com", {})
    assert result == {"data": []}
    assert mock_sleep.called


@patch("src.literature_review.clients.time.sleep")
@patch("src.literature_review.clients.requests.get")
def test_request_raises_after_max_retries_on_429(mock_get, mock_sleep):
    mock_get.return_value = _mock_response(429, headers={"Retry-After": "0.1"})
    with pytest.raises(requests.HTTPError):
        _request_with_retry("http://example.com", {})


@patch("src.literature_review.clients.time.sleep")
@patch("src.literature_review.clients.requests.get")
def test_request_retries_on_500_and_raises(mock_get, mock_sleep):
    mock_get.return_value = _mock_response(500)
    with pytest.raises(requests.HTTPError):
        _request_with_retry("http://example.com", {})


@patch("src.literature_review.clients.time.sleep")
@patch("src.literature_review.clients.requests.get")
def test_request_retries_on_network_error_and_raises(mock_get, mock_sleep):
    mock_get.side_effect = requests.ConnectionError("unreachable")
    with pytest.raises(requests.ConnectionError):
        _request_with_retry("http://example.com", {})


@patch("src.literature_review.clients.requests.get")
def test_request_raises_immediately_on_4xx(mock_get):
    mock_get.return_value = _mock_response(404)
    with pytest.raises(requests.HTTPError):
        _request_with_retry("http://example.com", {})
    assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# search_semantic_scholar
# ---------------------------------------------------------------------------

def _ss_item(paper_id: str) -> dict:
    return {
        "paperId": paper_id,
        "title": f"Paper {paper_id}",
        "abstract": "",
        "authors": [],
        "year": 2020,
        "url": None,
    }


@patch("src.literature_review.clients._request_with_retry")
def test_search_returns_papers_up_to_limit(mock_req):
    items = [_ss_item(str(i)) for i in range(5)]
    mock_req.return_value = {"data": items, "token": None}
    papers = search_semantic_scholar("transformers", limit=3)
    assert len(papers) == 3


@patch("src.literature_review.clients._request_with_retry")
def test_search_follows_pagination_token(mock_req):
    page1 = {"data": [_ss_item("1"), _ss_item("2")], "token": "tok-next"}
    page2 = {"data": [_ss_item("3"), _ss_item("4")], "token": None}
    mock_req.side_effect = [page1, page2]
    papers = search_semantic_scholar("transformers", limit=4)
    assert len(papers) == 4
    assert mock_req.call_count == 2


@patch("src.literature_review.clients._request_with_retry")
def test_search_stops_when_no_token(mock_req):
    mock_req.return_value = {"data": [_ss_item("1")], "token": None}
    papers = search_semantic_scholar("transformers", limit=10)
    assert len(papers) == 1


@patch("src.literature_review.clients._request_with_retry")
def test_search_returns_empty_list_on_no_data(mock_req):
    mock_req.return_value = {"data": [], "token": None}
    papers = search_semantic_scholar("transformers")
    assert papers == []


@patch("src.literature_review.clients._request_with_retry")
def test_search_includes_sort_param(mock_req):
    mock_req.return_value = {"data": [], "token": None}
    search_semantic_scholar("transformers", sort="citationCount:desc")
    params = mock_req.call_args[0][1]
    assert params["sort"] == "citationCount:desc"


@patch("src.literature_review.clients._request_with_retry")
def test_search_excludes_sort_param_when_none(mock_req):
    mock_req.return_value = {"data": [], "token": None}
    search_semantic_scholar("transformers")
    params = mock_req.call_args[0][1]
    assert "sort" not in params


@patch("src.literature_review.clients._request_with_retry")
def test_search_includes_publication_date_or_year_param(mock_req):
    mock_req.return_value = {"data": [], "token": None}
    search_semantic_scholar("transformers", publication_date_or_year="2020:2023")
    params = mock_req.call_args[0][1]
    assert params["publicationDateOrYear"] == "2020:2023"


@patch("src.literature_review.clients._request_with_retry")
def test_search_excludes_publication_date_or_year_param_when_none(mock_req):
    mock_req.return_value = {"data": [], "token": None}
    search_semantic_scholar("transformers")
    params = mock_req.call_args[0][1]
    assert "publicationDateOrYear" not in params
