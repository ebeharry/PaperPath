from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.conference.fetcher import (
    _parse_deadline,
    _slugify,
    fetch_conferences,
    filter_future_conferences,
)
from src.data_classes import RawConferenceEntry

_CSV_HEADERS = [
    "name", "full_name", "year", "category", "abstract_deadline",
    "paper_deadline", "deadline_timezone", "conf_start", "conf_end",
    "location", "homepage", "status", "notes",
]


def _raw_entry(**kwargs) -> dict:
    base = {
        "name": "CONF",
        "full_name": "Conference on Something",
        "year": "2030",
        "category": "",
        "abstract_deadline": "",
        "paper_deadline": "2030-01-01T23:59:59",
        "deadline_timezone": "UTC",
        "conf_start": "",
        "conf_end": "",
        "location": "",
        "homepage": "",
        "status": "open",
        "notes": "",
    }
    return {**base, **kwargs}


def _mock_response(*entries: dict) -> MagicMock:
    m = MagicMock()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_HEADERS)
    writer.writeheader()
    for e in entries:
        writer.writerow(e)
    m.text = buf.getvalue()
    return m


# --- fetch_conferences ---

@patch("src.conference.fetcher._request_with_retry")
def test_fetch_conferences_returns_entries(mock_req):
    mock_req.return_value = _mock_response(_raw_entry())
    result = fetch_conferences()
    assert len(result) == 1
    assert result[0].name == "CONF"


@patch("src.conference.fetcher._request_with_retry")
def test_fetch_conferences_extra_keys_accepted(mock_req):
    entry = _raw_entry()
    mock_req.return_value = _mock_response(entry)
    result = fetch_conferences()
    assert len(result) == 1


@patch("src.conference.fetcher._request_with_retry")
def test_fetch_conferences_malformed_entry_skipped(mock_req):
    good = _raw_entry()
    bad = _raw_entry(name="")  # empty name is skipped before validation
    mock_req.return_value = _mock_response(good, bad)
    result = fetch_conferences()
    assert len(result) == 1
    assert result[0].name == "CONF"


@patch("src.conference.fetcher._request_with_retry")
def test_fetch_conferences_pydantic_validation_failure_skipped(mock_req):
    mock_req.return_value = _mock_response(_raw_entry(), _raw_entry())
    from src.data_classes import RawConferenceEntry as _RCE
    original = _RCE.model_validate
    call_count = 0
    def fail_on_second(row, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return original({})  # empty dict → missing required fields → ValidationError
        return original(row, **kw)
    with patch("src.conference.fetcher.RawConferenceEntry.model_validate", side_effect=fail_on_second):
        result = fetch_conferences()
    assert len(result) == 1


@patch("src.conference.fetcher._request_with_retry")
def test_fetch_conferences_empty_csv(mock_req):
    m = MagicMock()
    m.text = ""
    mock_req.return_value = m
    result = fetch_conferences()
    assert result == []


@patch("src.conference.fetcher._request_with_retry")
def test_fetch_conferences_empty_paper_deadline_skipped(mock_req):
    no_deadline = _raw_entry(paper_deadline="")
    mock_req.return_value = _mock_response(_raw_entry(), no_deadline)
    result = fetch_conferences()
    assert len(result) == 1


@patch("src.conference.fetcher._request_with_retry")
def test_fetch_conferences_optional_fields_default_to_none(mock_req):
    mock_req.return_value = _mock_response(_raw_entry())
    result = fetch_conferences()
    assert result[0].abstract_deadline is None
    assert result[0].category is None
    assert result[0].homepage is None


@patch("src.conference.fetcher._request_with_retry")
def test_fetch_conferences_empty_deadline_timezone_defaults_to_none(mock_req):
    mock_req.return_value = _mock_response(_raw_entry(deadline_timezone=""))
    result = fetch_conferences()
    assert result[0].deadline_timezone is None


# --- _slugify ---

def test_slugify_lowercases():
    assert _slugify("NEURIPS") == "neurips"


def test_slugify_replaces_spaces_with_dashes():
    assert _slugify("Neural IPS") == "neural-ips"


def test_slugify_replaces_special_chars():
    assert _slugify("CONF@2024!") == "conf-2024"


def test_slugify_collapses_consecutive_separators():
    assert _slugify("a  b--c") == "a-b-c"


def test_slugify_strips_leading_trailing_dashes():
    assert _slugify("---test---") == "test"


def test_slugify_preserves_alphanumeric():
    assert _slugify("conf2024ml") == "conf2024ml"


def test_slugify_empty_string():
    assert _slugify("") == ""


def test_slugify_only_special_chars():
    assert _slugify("---") == ""


# --- filter_future_conferences ---

def _entry(paper_deadline: str, deadline_timezone: str = "UTC") -> RawConferenceEntry:
    return RawConferenceEntry(
        id="x", name="X", full_name="X Conference",
        paper_deadline=paper_deadline, deadline_timezone=deadline_timezone,
    )


def test_filter_future_keeps_future():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    entries = [_entry("2026-01-01 00:00:00")]
    assert len(filter_future_conferences(entries, now=now)) == 1


def test_filter_future_excludes_past():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    entries = [_entry("2024-01-01 00:00:00")]
    assert filter_future_conferences(entries, now=now) == []


def test_filter_future_excludes_exact_now():
    now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    entries = [_entry("2025-06-01 12:00:00")]
    assert filter_future_conferences(entries, now=now) == []


def test_filter_future_excludes_tbd():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    entries = [_entry("TBD")]
    assert filter_future_conferences(entries, now=now) == []


def test_filter_future_mixed():
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    entries = [
        _entry("2024-01-01 00:00:00"),
        _entry("2026-01-01 00:00:00"),
        _entry("TBD"),
    ]
    result = filter_future_conferences(entries, now=now)
    assert len(result) == 1
    assert result[0].paper_deadline == "2026-01-01 00:00:00"


# --- _parse_deadline ---

def test_parse_deadline_valid_utc():
    dt = _parse_deadline("2030-06-15 23:59:59", "UTC")
    assert dt is not None
    assert dt.year == 2030
    assert dt.tzinfo is not None


def test_parse_deadline_iso8601_t_separator():
    dt = _parse_deadline("2030-06-15T23:59:59", "UTC")
    assert dt is not None
    assert dt.year == 2030
    assert dt.month == 6


def test_parse_deadline_iana_timezone():
    dt = _parse_deadline("2030-09-01 23:59:59", "America/Los_Angeles")
    assert dt is not None
    assert dt.utcoffset() is not None


def test_parse_deadline_utc_negative_offset():
    dt = _parse_deadline("2030-01-01 23:59:59", "UTC-12")
    assert dt is not None
    assert dt.utcoffset() == timedelta(hours=-12)


def test_parse_deadline_utc_positive_offset():
    dt = _parse_deadline("2030-01-01 23:59:59", "UTC+5")
    assert dt is not None
    assert dt.utcoffset() == timedelta(hours=5)


def test_parse_deadline_aoe():
    dt = _parse_deadline("2030-01-01 23:59:59", "AoE")
    assert dt is not None
    assert dt.utcoffset() == timedelta(hours=-12)


def test_parse_deadline_tbd_returns_none():
    assert _parse_deadline("TBD", "UTC") is None


def test_parse_deadline_empty_string_returns_none():
    assert _parse_deadline("", "UTC") is None


def test_parse_deadline_unparseable_returns_none():
    assert _parse_deadline("not-a-date", "UTC") is None


def test_parse_deadline_none_timezone_falls_back_to_utc():
    dt = _parse_deadline("2030-01-01 00:00:00", None)
    assert dt is not None
    assert dt.tzinfo == timezone.utc


def test_parse_deadline_unknown_timezone_falls_back_to_utc():
    dt = _parse_deadline("2030-01-01 00:00:00", "Galaxy/Zorg")
    assert dt is not None
    assert dt.tzinfo == timezone.utc
