import xml.etree.ElementTree as ET

import pytest
import requests
from unittest.mock import MagicMock, patch, call

from src.literature_review.clients import (
    _get_semantic_scholar_headers,
    _parse_semantic_scholar_paper,
    _parse_arxiv_entry,
    _build_arxiv_query,
    _build_arxiv_sort_params,
    _request_with_retry,
    search_semantic_scholar,
    search_arxiv,
)
from src.data_classes import Paper


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
    assert result.json() == {"data": [{"paperId": "x"}]}


@patch("src.literature_review.clients.time.sleep")
@patch("src.literature_review.clients.requests.get")
def test_request_retries_on_429_and_succeeds(mock_get, mock_sleep):
    mock_get.side_effect = [
        _mock_response(429, headers={"Retry-After": "0.1"}),
        _mock_response(200, {"data": []}),
    ]
    result = _request_with_retry("http://example.com", {})
    assert result.json() == {"data": []}
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

def _ss_response(data: dict) -> MagicMock:
    m = MagicMock()
    m.json.return_value = data
    return m


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
    mock_req.return_value = _ss_response({"data": items, "token": None})
    papers = search_semantic_scholar("transformers", limit=3)
    assert len(papers) == 3


@patch("src.literature_review.clients._request_with_retry")
def test_search_follows_pagination_token(mock_req):
    page1 = {"data": [_ss_item("1"), _ss_item("2")], "token": "tok-next"}
    page2 = {"data": [_ss_item("3"), _ss_item("4")], "token": None}
    mock_req.side_effect = [_ss_response(page1), _ss_response(page2)]
    papers = search_semantic_scholar("transformers", limit=4)
    assert len(papers) == 4
    assert mock_req.call_count == 2


@patch("src.literature_review.clients._request_with_retry")
def test_search_stops_when_no_token(mock_req):
    mock_req.return_value = _ss_response({"data": [_ss_item("1")], "token": None})
    papers = search_semantic_scholar("transformers", limit=10)
    assert len(papers) == 1


@patch("src.literature_review.clients._request_with_retry")
def test_search_returns_empty_list_on_no_data(mock_req):
    mock_req.return_value = _ss_response({"data": [], "token": None})
    papers = search_semantic_scholar("transformers")
    assert papers == []


@patch("src.literature_review.clients._request_with_retry")
def test_search_includes_sort_param(mock_req):
    mock_req.return_value = _ss_response({"data": [], "token": None})
    search_semantic_scholar("transformers", sort="citationCount:desc")
    params = mock_req.call_args[0][1]
    assert params["sort"] == "citationCount:desc"


@patch("src.literature_review.clients._request_with_retry")
def test_search_excludes_sort_param_when_none(mock_req):
    mock_req.return_value = _ss_response({"data": [], "token": None})
    search_semantic_scholar("transformers")
    params = mock_req.call_args[0][1]
    assert "sort" not in params


@patch("src.literature_review.clients._request_with_retry")
def test_search_includes_year_param(mock_req):
    mock_req.return_value = _ss_response({"data": [], "token": None})
    search_semantic_scholar("transformers", year="2020:2023")
    params = mock_req.call_args[0][1]
    assert params["year"] == "2020-2023"


@patch("src.literature_review.clients._request_with_retry")
def test_search_excludes_year_param_when_none(mock_req):
    mock_req.return_value = _ss_response({"data": [], "token": None})
    search_semantic_scholar("transformers")
    params = mock_req.call_args[0][1]
    assert "year" not in params


# ---------------------------------------------------------------------------
# _build_arxiv_query
# ---------------------------------------------------------------------------

def test_build_arxiv_query_no_year():
    result = _build_arxiv_query("transformers", None)
    assert result == "all:transformers"


def test_build_arxiv_query_range():
    result = _build_arxiv_query("transformers", "2020:2023")
    assert "submittedDate:[202001010000 TO 202312312359]" in result


def test_build_arxiv_query_from_year():
    result = _build_arxiv_query("transformers", "2023-")
    assert "submittedDate:[202301010000 TO 999912312359]" in result


def test_build_arxiv_query_single_year():
    result = _build_arxiv_query("transformers", "2021")
    assert "submittedDate:[202101010000 TO 202112312359]" in result


# ---------------------------------------------------------------------------
# _build_arxiv_sort_params
# ---------------------------------------------------------------------------

def test_build_arxiv_sort_none():
    assert _build_arxiv_sort_params(None) == {}


def test_build_arxiv_sort_submitted_desc():
    assert _build_arxiv_sort_params("submittedDate:desc") == {
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }


def test_build_arxiv_sort_relevance_desc():
    assert _build_arxiv_sort_params("relevance:desc") == {
        "sortBy": "relevance",
        "sortOrder": "descending",
    }


def test_build_arxiv_sort_lastupdated_asc():
    assert _build_arxiv_sort_params("lastUpdatedDate:asc") == {
        "sortBy": "lastUpdatedDate",
        "sortOrder": "ascending",
    }


# ---------------------------------------------------------------------------
# _parse_arxiv_entry
# ---------------------------------------------------------------------------

_ARXIV_NS = "http://www.w3.org/2005/Atom"


def _arxiv_entry_xml(
    paper_id="2301.04567v1",
    title="Test Title",
    summary="Test Abstract",
    authors=("Alice", "Bob"),
    published="2023-01-12T00:00:00Z",
) -> ET.Element:
    ns = _ARXIV_NS
    author_els = "".join(
        f"<author xmlns='{ns}'><name>{a}</name></author>" for a in authors
    )
    return ET.fromstring(
        f"<entry xmlns='{ns}'>"
        f"<id>http://arxiv.org/abs/{paper_id}</id>"
        f"<title>{title}</title>"
        f"<summary>{summary}</summary>"
        f"{author_els}"
        f"<published>{published}</published>"
        f"</entry>"
    )


def test_parse_arxiv_full_entry():
    paper = _parse_arxiv_entry(_arxiv_entry_xml())
    assert paper == Paper(
        paper_id="2301.04567v1",
        title="Test Title",
        abstract="Test Abstract",
        authors=["Alice", "Bob"],
        year=2023,
        url="http://arxiv.org/abs/2301.04567v1",
        source="arxiv",
    )


def test_parse_arxiv_paper_id_extracted_from_url():
    paper = _parse_arxiv_entry(_arxiv_entry_xml(paper_id="2301.04567v1"))
    assert paper.paper_id == "2301.04567v1"


def test_parse_arxiv_url_is_full_id_field():
    paper = _parse_arxiv_entry(_arxiv_entry_xml(paper_id="2301.04567v1"))
    assert paper.url == "http://arxiv.org/abs/2301.04567v1"


def test_parse_arxiv_missing_title():
    entry = _arxiv_entry_xml(title="")
    assert _parse_arxiv_entry(entry).title == ""


def test_parse_arxiv_missing_summary():
    entry = _arxiv_entry_xml(summary="")
    assert _parse_arxiv_entry(entry).abstract == ""


def test_parse_arxiv_no_authors():
    entry = _arxiv_entry_xml(authors=())
    assert _parse_arxiv_entry(entry).authors == []


def test_parse_arxiv_missing_published():
    ns = _ARXIV_NS
    entry = ET.fromstring(
        f"<entry xmlns='{ns}'>"
        f"<id>http://arxiv.org/abs/2301.04567v1</id>"
        f"<title>T</title><summary>S</summary>"
        f"</entry>"
    )
    assert _parse_arxiv_entry(entry).year is None


def test_parse_arxiv_malformed_published():
    entry = _arxiv_entry_xml(published="not-a-date")
    assert _parse_arxiv_entry(entry).year is None


def test_parse_arxiv_source_always_arxiv():
    assert _parse_arxiv_entry(_arxiv_entry_xml()).source == "arxiv"


# ---------------------------------------------------------------------------
# search_arxiv
# ---------------------------------------------------------------------------

def _arxiv_entry_str(suffix="2301.00001v1", title="Paper", year="2023") -> str:
    ns = _ARXIV_NS
    return (
        f"<entry xmlns='{ns}'>"
        f"<id>http://arxiv.org/abs/{suffix}</id>"
        f"<title>{title}</title><summary>Abs.</summary>"
        f"<author><name>A</name></author>"
        f"<published>{year}-01-01T00:00:00Z</published>"
        f"</entry>"
    )


def _arxiv_feed(entries: list[str], total: int) -> MagicMock:
    ns = _ARXIV_NS
    ns_os = "http://a9.com/-/spec/opensearch/1.1/"
    body = "\n".join(entries)
    xml = (
        f'<?xml version="1.0"?>'
        f'<feed xmlns="{ns}" xmlns:opensearch="{ns_os}">'
        f"<opensearch:totalResults>{total}</opensearch:totalResults>"
        f"{body}"
        f"</feed>"
    )
    m = MagicMock()
    m.text = xml
    return m


@patch("src.literature_review.clients._request_with_retry")
def test_search_arxiv_returns_papers_up_to_limit(mock_req):
    entries = [_arxiv_entry_str(suffix=f"230{i}.00001v1") for i in range(5)]
    mock_req.return_value = _arxiv_feed(entries, total=5)
    papers = search_arxiv("transformers", limit=3)
    assert len(papers) == 3


@patch("src.literature_review.clients._request_with_retry")
def test_search_arxiv_stops_when_no_entries(mock_req):
    mock_req.return_value = _arxiv_feed([], total=0)
    papers = search_arxiv("transformers")
    assert papers == []


@patch("src.literature_review.clients.time.sleep")
@patch("src.literature_review.clients._request_with_retry")
def test_search_arxiv_paginates_when_more_results_exist(mock_req, mock_sleep):
    page1 = [_arxiv_entry_str(suffix=f"2301.0000{i}v1") for i in range(2)]
    page2 = [_arxiv_entry_str(suffix=f"2302.0000{i}v1") for i in range(2)]
    mock_req.side_effect = [
        _arxiv_feed(page1, total=4),
        _arxiv_feed(page2, total=4),
    ]
    papers = search_arxiv("transformers", limit=4)
    assert len(papers) == 4
    assert mock_req.call_count == 2
    assert mock_sleep.call_count == 1


@patch("src.literature_review.clients._request_with_retry")
def test_search_arxiv_stops_at_total_results(mock_req):
    entries = [_arxiv_entry_str(suffix=f"2301.0000{i}v1") for i in range(2)]
    mock_req.return_value = _arxiv_feed(entries, total=2)
    papers = search_arxiv("transformers", limit=10)
    assert len(papers) == 2
    assert mock_req.call_count == 1


@patch("src.literature_review.clients._request_with_retry")
def test_search_arxiv_includes_sort_params(mock_req):
    mock_req.return_value = _arxiv_feed([], total=0)
    search_arxiv("transformers", sort="submittedDate:desc")
    params = mock_req.call_args[0][1]
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"


@patch("src.literature_review.clients._request_with_retry")
def test_search_arxiv_excludes_sort_when_none(mock_req):
    mock_req.return_value = _arxiv_feed([], total=0)
    search_arxiv("transformers")
    params = mock_req.call_args[0][1]
    assert "sortBy" not in params
    assert "sortOrder" not in params


@patch("src.literature_review.clients._request_with_retry")
def test_search_arxiv_includes_year_in_query(mock_req):
    mock_req.return_value = _arxiv_feed([], total=0)
    search_arxiv("transformers", year="2020:2023")
    params = mock_req.call_args[0][1]
    assert "submittedDate" in params["search_query"]


@patch("src.literature_review.clients._request_with_retry")
def test_search_arxiv_excludes_year_when_none(mock_req):
    mock_req.return_value = _arxiv_feed([], total=0)
    search_arxiv("transformers")
    params = mock_req.call_args[0][1]
    assert "submittedDate" not in params["search_query"]


@patch("src.literature_review.clients._request_with_retry")
def test_search_arxiv_source_is_arxiv(mock_req):
    entries = [_arxiv_entry_str()]
    mock_req.return_value = _arxiv_feed(entries, total=1)
    papers = search_arxiv("transformers", limit=1)
    assert papers[0].source == "arxiv"
