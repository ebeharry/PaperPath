from __future__ import annotations

import csv
import io
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ValidationError

from src.data_classes import RawConferenceEntry
from src.literature_review.clients import _request_with_retry

_DEADLINES_URL = "https://raw.githubusercontent.com/khairulislam/ML-conferences/main/docs/data/conferences.csv"

_AOE_OFFSET = timezone(timedelta(hours=-12))

_OPTIONAL_FIELDS = frozenset({"abstract_deadline", "deadline_timezone", "homepage", "category", "full_name"})


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def fetch_conferences(url: str = _DEADLINES_URL) -> list[RawConferenceEntry]:
    response = _request_with_retry(url, params={})
    reader = csv.DictReader(io.StringIO(response.text))
    entries: list[RawConferenceEntry] = []
    for row in reader:
        if not row.get("paper_deadline", "").strip() or not row.get("name", "").strip():
            continue
        year = row.get("year", "").strip() or "0"
        row["id"] = f"{_slugify(row['name'])}-{year}"
        for key in _OPTIONAL_FIELDS:
            if not row.get(key, "").strip():
                row[key] = None
        try:
            entries.append(RawConferenceEntry.model_validate(row))
        except ValidationError:
            continue
    return entries


def _parse_deadline(deadline_str: str, timezone_str: str | None) -> datetime | None:
    if not deadline_str or deadline_str.strip().upper() == "TBD":
        return None
    s = deadline_str.strip()
    dt = None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        return None
    tz = _resolve_timezone(timezone_str)
    return dt.replace(tzinfo=tz)


def _resolve_timezone(timezone_str: str | None) -> timezone | ZoneInfo:
    if not timezone_str:
        return timezone.utc

    tz_str = timezone_str.strip()

    if tz_str.upper() == "AOE":
        return _AOE_OFFSET

    try:
        return ZoneInfo(tz_str)
    except (ZoneInfoNotFoundError, KeyError):
        pass

    match = re.fullmatch(r"UTC([+-])(\d{1,2})(?::(\d{2}))?", tz_str, re.IGNORECASE)
    if match:
        sign = 1 if match.group(1) == "+" else -1
        hours = int(match.group(2))
        minutes = int(match.group(3)) if match.group(3) else 0
        return timezone(timedelta(hours=sign * hours, minutes=sign * minutes))

    return timezone.utc


def filter_future_conferences(
    entries: list[RawConferenceEntry],
    now: datetime | None = None,
) -> list[RawConferenceEntry]:
    if now is None:
        now = datetime.now(timezone.utc)
    result: list[RawConferenceEntry] = []
    for entry in entries:
        dt = _parse_deadline(entry.paper_deadline, entry.deadline_timezone)
        if dt is not None and dt > now:
            result.append(entry)
    return result
