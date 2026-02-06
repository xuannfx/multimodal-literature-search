#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError


USER_AGENT = "multimodal-literature-search/1.0 (+https://openai.com; codex-skill)"


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _http_get(url: str, timeout_s: int = 30, extra_headers: dict[str, str] | None = None) -> bytes:
    ssl_context: ssl.SSLContext | None = None
    try:
        import certifi  # type: ignore

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ssl_context = None

    headers: dict[str, str] = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/xml,application/atom+xml,*/*",
    }
    if extra_headers:
        headers.update(extra_headers)

    last_err: Exception | None = None
    for attempt in range(1, 4):
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s, context=ssl_context) as resp:
                return resp.read()
        except HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                retry_after = e.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else 1.5 * attempt
                time.sleep(min(10.0, sleep_s))
                continue
            raise
        except URLError as e:
            last_err = e
            time.sleep(0.8 * attempt)
            continue
    if last_err:
        raise last_err
    raise RuntimeError("HTTP request failed without exception")


def _http_post_json(
    url: str,
    payload: Any,
    timeout_s: int = 45,
    extra_headers: dict[str, str] | None = None,
) -> Any:
    ssl_context: ssl.SSLContext | None = None
    try:
        import certifi  # type: ignore

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ssl_context = None

    headers: dict[str, str] = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,*/*",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    body = json.dumps(payload).encode("utf-8")

    last_err: Exception | None = None
    for attempt in range(1, 5):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s, context=ssl_context) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                retry_after = e.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else 1.5 * attempt
                time.sleep(min(15.0, sleep_s))
                continue
            raise
        except URLError as e:
            last_err = e
            time.sleep(0.8 * attempt)
            continue
    if last_err:
        raise last_err
    raise RuntimeError("HTTP request failed without exception")


def _http_get_json(
    url: str, timeout_s: int = 30, extra_headers: dict[str, str] | None = None
) -> Any:
    body = _http_get(url, timeout_s=timeout_s, extra_headers=extra_headers)
    return json.loads(body.decode("utf-8"))


def _normalize_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip().lower()
    title = re.sub(r"[^a-z0-9]+", "", title)
    return title


def _canonical_arxiv_id(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    s = str(arxiv_id).strip()
    if not s:
        return None
    return re.sub(r"v\d+$", "", s, flags=re.IGNORECASE) or None


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            parsed = dt.datetime.strptime(value, fmt)
            if fmt == "%Y":
                return dt.date(parsed.year, 1, 1)
            if fmt == "%Y-%m":
                return dt.date(parsed.year, parsed.month, 1)
            return parsed.date()
        except ValueError:
            continue
    return None


def _in_range(date_value: dt.date | None, since: dt.date, until: dt.date) -> bool:
    if date_value is None:
        return True
    return since <= date_value <= until


def _pick_first(d: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return None


@dataclass
class Paper:
    title: str
    authors: list[str]
    venue: str | None
    year: int | None
    publication_date: str | None
    abstract: str | None
    url: str | None
    pdf_url: str | None
    doi: str | None
    arxiv_id: str | None
    citation_count: int | None
    source: str

    def dedupe_key(self) -> tuple[str, str]:
        if self.doi:
            return ("doi", self.doi.lower())
        if self.arxiv_id:
            return ("arxiv", (_canonical_arxiv_id(self.arxiv_id) or self.arxiv_id).lower())
        return ("title", _normalize_title(self.title))

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "venue": self.venue,
            "year": self.year,
            "publicationDate": self.publication_date,
            "abstract": self.abstract,
            "url": self.url,
            "pdfUrl": self.pdf_url,
            "doi": self.doi,
            "arxivId": self.arxiv_id,
            "citationCount": self.citation_count,
            "source": self.source,
        }


def fetch_arxiv(
    *, query: str, mode: str, max_results: int, raw_query: bool = False, page_size: int = 200
) -> list[Paper]:
    """
    arXiv API query helper.

    By default, treat `query` as a phrase and wrap it as all:"...".
    When `raw_query=True`, pass `query` through verbatim as `search_query`,
    so callers can use arXiv advanced query syntax (field qualifiers, boolean ops).
    """
    if raw_query:
        search_query = query
    elif mode == "author":
        search_query = f'au:"{query}"'
    else:
        search_query = f'all:"{query}"'

    papers: list[Paper] = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    page_size = max(1, min(int(page_size), 2000))
    start = 0
    while start < max_results:
        batch = min(page_size, max_results - start)
        params = {
            "search_query": search_query,
            "start": str(start),
            "max_results": str(batch),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        xml_bytes = _http_get(url)

        root = ET.fromstring(xml_bytes)
        entries = root.findall("atom:entry", ns)
        if not entries:
            break

        for entry in entries:
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            title = re.sub(r"\s+", " ", title)
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            summary = re.sub(r"\s+", " ", summary)

            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
            published_date = _parse_date(published[:10])  # 2025-01-02T...
            publication_date = published_date.isoformat() if published_date else None

            authors: list[str] = []
            for a in entry.findall("atom:author", ns):
                name = (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
                if name:
                    authors.append(name)

            entry_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else None

            url_abs: str | None = None
            pdf_url: str | None = None
            for link in entry.findall("atom:link", ns):
                href = link.attrib.get("href")
                rel = link.attrib.get("rel")
                link_type = link.attrib.get("type")
                title_attr = link.attrib.get("title")
                if rel == "alternate" and href:
                    url_abs = href
                if (title_attr == "pdf" or link_type == "application/pdf") and href:
                    pdf_url = href

            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    venue=None,
                    year=published_date.year if published_date else None,
                    publication_date=publication_date,
                    abstract=summary or None,
                    url=url_abs or entry_id or None,
                    pdf_url=pdf_url,
                    doi=None,
                    arxiv_id=arxiv_id,
                    citation_count=None,
                    source="arxiv",
                )
            )

        start += len(entries)
        if len(entries) < batch:
            break
        time.sleep(0.2)

    return papers


def _ss_fields() -> str:
    # Keep fields minimal but useful for report + citation.
    return ",".join(
        [
            "title",
            "authors",
            "venue",
            "year",
            "publicationDate",
            "abstract",
            "url",
            "externalIds",
            "openAccessPdf",
            "citationCount",
        ]
    )


def fetch_semantic_scholar_keyword(
    *, query: str, max_results: int, api_key: str | None = None
) -> list[Paper]:
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    page_limit = min(100, max_results)
    offset = 0
    out: list[Paper] = []
    extra_headers = {"x-api-key": api_key} if api_key else None
    while len(out) < max_results:
        params = {
            "query": query,
            "limit": str(page_limit),
            "offset": str(offset),
            "fields": _ss_fields(),
        }
        url = base + "?" + urllib.parse.urlencode(params)
        payload = _http_get_json(url, extra_headers=extra_headers)
        data = payload.get("data") or []
        if not data:
            break
        for item in data:
            external_ids = item.get("externalIds") or {}
            doi = _pick_first(external_ids, ["DOI", "doi"])
            arxiv_id = _pick_first(external_ids, ["ArXiv", "arXiv", "arxiv"])
            pdf_url = None
            open_access = item.get("openAccessPdf") or {}
            if isinstance(open_access, dict):
                pdf_url = open_access.get("url")
            authors = [a.get("name") for a in (item.get("authors") or []) if a.get("name")]
            out.append(
                Paper(
                    title=item.get("title") or "",
                    authors=authors,
                    venue=item.get("venue"),
                    year=item.get("year"),
                    publication_date=item.get("publicationDate"),
                    abstract=item.get("abstract"),
                    url=item.get("url"),
                    pdf_url=pdf_url,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    citation_count=item.get("citationCount"),
                    source="semanticscholar",
                )
            )
            if len(out) >= max_results:
                break
        offset += len(data)
        if offset >= (payload.get("total") or offset):
            break
    return out


def fetch_semantic_scholar_author(
    *, author: str, max_results: int, api_key: str | None = None
) -> list[Paper]:
    # 1) resolve authorId
    base = "https://api.semanticscholar.org/graph/v1/author/search"
    params = {
        "query": author,
        "limit": "5",
        "fields": "name,affiliations,paperCount,citationCount,aliases",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    extra_headers = {"x-api-key": api_key} if api_key else None
    payload = _http_get_json(url, extra_headers=extra_headers)
    candidates = payload.get("data") or []
    if not candidates:
        return []
    chosen = candidates[0]
    author_id = chosen.get("authorId")
    if not author_id:
        return []

    # 2) fetch papers
    papers_url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"
    page_limit = min(100, max_results)
    offset = 0
    out: list[Paper] = []
    while len(out) < max_results:
        params2 = {
            "limit": str(page_limit),
            "offset": str(offset),
            "fields": _ss_fields(),
        }
        url2 = papers_url + "?" + urllib.parse.urlencode(params2)
        payload2 = _http_get_json(url2, extra_headers=extra_headers)
        data = payload2.get("data") or []
        if not data:
            break
        for row in data:
            # API sometimes returns {"paper": {...}}.
            item = row.get("paper") if isinstance(row, dict) and "paper" in row else row
            if not isinstance(item, dict):
                continue
            external_ids = item.get("externalIds") or {}
            doi = _pick_first(external_ids, ["DOI", "doi"])
            arxiv_id = _pick_first(external_ids, ["ArXiv", "arXiv", "arxiv"])
            pdf_url = None
            open_access = item.get("openAccessPdf") or {}
            if isinstance(open_access, dict):
                pdf_url = open_access.get("url")
            authors = [a.get("name") for a in (item.get("authors") or []) if a.get("name")]
            out.append(
                Paper(
                    title=item.get("title") or "",
                    authors=authors,
                    venue=item.get("venue"),
                    year=item.get("year"),
                    publication_date=item.get("publicationDate"),
                    abstract=item.get("abstract"),
                    url=item.get("url"),
                    pdf_url=pdf_url,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    citation_count=item.get("citationCount"),
                    source="semanticscholar",
                )
            )
            if len(out) >= max_results:
                break
        offset += len(data)
        if offset >= (payload2.get("total") or offset):
            break
    return out


def _merge_prefer_richer(existing: Paper, incoming: Paper) -> Paper:
    # Prefer entries with publication_date, abstract, venue, pdf_url.
    def score(p: Paper) -> int:
        return sum(
            1
            for v in (
                p.publication_date,
                p.abstract,
                p.venue,
                p.pdf_url,
                p.doi,
                p.arxiv_id,
            )
            if v
        )

    return incoming if score(incoming) > score(existing) else existing


def _enrich_with_semantic_scholar_batch(
    papers: list[Paper],
    *,
    api_key: str | None,
    batch_size: int = 100,
    sleep_s: float = 0.2,
    max_batches: int | None = None,
    verbose: bool = False,
) -> int:
    extra_headers = {"x-api-key": api_key} if api_key else None
    ids: list[str] = []
    idxs: list[int] = []
    seen: set[str] = set()

    for i, p in enumerate(papers):
        arxiv_id = _canonical_arxiv_id(p.arxiv_id)
        paper_id: str | None = None
        if arxiv_id:
            paper_id = f"ARXIV:{arxiv_id}"
        elif p.doi:
            paper_id = f"DOI:{p.doi.strip()}"

        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        ids.append(paper_id)
        idxs.append(i)

    if not ids:
        return 0

    batch_size = max(1, min(int(batch_size), 500))
    url = "https://api.semanticscholar.org/graph/v1/paper/batch?" + urllib.parse.urlencode(
        {"fields": _ss_fields()}
    )

    calls = 0
    total_batches = (len(ids) + batch_size - 1) // batch_size
    for b in range(total_batches):
        if max_batches is not None and b >= max_batches:
            break
        start = b * batch_size
        chunk = ids[start : start + batch_size]
        chunk_idxs = idxs[start : start + batch_size]
        try:
            payload = _http_post_json(url, {"ids": chunk}, extra_headers=extra_headers)
            calls += 1
        except Exception as e:
            calls += 1
            if verbose:
                _stderr(f"[WARN] S2 enrichment failed (batch {b+1}/{total_batches}): {e}")
            time.sleep(max(0.0, float(sleep_s)))
            continue

        if not isinstance(payload, list):
            if verbose:
                _stderr(f"[WARN] Unexpected S2 payload type: {type(payload)}")
            time.sleep(max(0.0, float(sleep_s)))
            continue

        for paper_idx, item in zip(chunk_idxs, payload):
            if not isinstance(item, dict):
                continue
            p = papers[paper_idx]

            if p.venue is None and item.get("venue"):
                p.venue = item.get("venue")
            if p.year is None and item.get("year"):
                try:
                    p.year = int(item.get("year"))
                except Exception:
                    pass
            if p.publication_date is None and item.get("publicationDate"):
                p.publication_date = item.get("publicationDate")
            if p.abstract is None and item.get("abstract"):
                p.abstract = item.get("abstract")
            if p.url is None and item.get("url"):
                p.url = item.get("url")
            if p.citation_count is None and item.get("citationCount") is not None:
                try:
                    p.citation_count = int(item.get("citationCount"))
                except Exception:
                    pass

            if not p.authors:
                authors = [a.get("name") for a in (item.get("authors") or []) if isinstance(a, dict) and a.get("name")]
                if authors:
                    p.authors = authors

            external_ids = item.get("externalIds") or {}
            if p.doi is None:
                doi = _pick_first(external_ids, ["DOI", "doi"])
                if isinstance(doi, str) and doi.strip():
                    p.doi = doi.strip()
            if p.arxiv_id is None:
                arx = _pick_first(external_ids, ["ArXiv", "arXiv", "arxiv"])
                if isinstance(arx, str) and arx.strip():
                    p.arxiv_id = arx.strip()

            open_access = item.get("openAccessPdf") or {}
            if p.pdf_url is None and isinstance(open_access, dict) and open_access.get("url"):
                p.pdf_url = open_access.get("url")

        time.sleep(max(0.0, float(sleep_s)))

    return calls


def _to_markdown_table(papers: list[Paper]) -> str:
    headers = ["#", "标题", "发表时间", "引用", "来源", "Venue", "链接"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for i, p in enumerate(papers, 1):
        pub = p.publication_date or (str(p.year) if p.year else "")
        venue = p.venue or ""
        url = p.url or ""
        cites = "" if p.citation_count is None else str(p.citation_count)
        title = p.title.replace("|", "\\|")
        lines.append(f"| {i} | {title} | {pub} | {cites} | {p.source} | {venue} | {url} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and dedupe recent multimodal papers.")
    q = parser.add_mutually_exclusive_group(required=True)
    q.add_argument("--query", help="Topic keywords (keyword search).")
    q.add_argument("--author", help="Author or team name (author search when supported).")
    parser.add_argument(
        "--sources",
        default="arxiv,semanticscholar",
        help="Comma-separated sources. Supported: arxiv, semanticscholar.",
    )
    parser.add_argument(
        "--arxiv-raw",
        action="store_true",
        help='If set, pass --query/--author directly to arXiv search_query (advanced syntax).',
    )
    parser.add_argument(
        "--arxiv-page-size",
        type=int,
        default=200,
        help="arXiv paging batch size (default: 200).",
    )
    parser.add_argument("--since", help="Start date (YYYY-MM-DD). Overrides --months if set.")
    parser.add_argument("--until", help="End date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--months", type=int, default=18, help="Look back N months (default: 18).")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Final number of papers after dedupe (default: 5, for top-5 reporting).",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Fetch up to N papers per source before dedupe/filter (default: auto, usually 80 or limit*3).",
    )
    parser.add_argument(
        "--sort",
        choices=["date", "citations", "citations_per_year"],
        default="date",
        help="Sort strategy for final output (default: date).",
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=0,
        help="Minimum citationCount (default: 0). Requires --enrich-s2 to be meaningful for arXiv-only queries.",
    )
    parser.add_argument(
        "--venue-contains",
        help='Comma-separated substrings; keep only papers whose venue contains any of them (case-insensitive).',
    )
    parser.add_argument(
        "--exclude-venue-contains",
        help='Comma-separated substrings; drop papers whose venue contains any of them (case-insensitive).',
    )
    parser.add_argument(
        "--require-venue",
        action="store_true",
        help="Drop papers without venue (after optional enrichment).",
    )
    parser.add_argument(
        "--enrich-s2",
        action="store_true",
        help="Enrich with Semantic Scholar /paper/batch to fill citationCount/venue/DOI when possible.",
    )
    parser.add_argument("--s2-batch-size", type=int, default=100, help="S2 batch size (default: 100).")
    parser.add_argument("--s2-sleep", type=float, default=0.2, help="Sleep seconds between S2 batches (default: 0.2).")
    parser.add_argument(
        "--s2-max-batches",
        type=int,
        help="Optional cap for S2 enrichment batches (useful to avoid rate limits).",
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--out-md", help="Optional output Markdown table path.")
    parser.add_argument(
        "--s2-api-key",
        default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
        help="Optional Semantic Scholar API key (env: SEMANTIC_SCHOLAR_API_KEY).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    today = dt.date.today()
    until = _parse_date(args.until) or today
    if args.since:
        since = _parse_date(args.since)
        if since is None:
            raise SystemExit(f"Invalid --since: {args.since}")
    else:
        since = until - dt.timedelta(days=int(args.months * 30.5))

    sources = [s.strip().lower() for s in (args.sources or "").split(",") if s.strip()]
    mode = "author" if args.author else "keyword"
    query = args.author or args.query or ""

    # Auto-tune max_per_source if not set
    if args.max_per_source is None:
        # If limit is small (e.g. 5), default 80 is fine.
        # If limit is large (e.g. 100), we need more candidates.
        # We assume ~20% survival rate after date/relevance filtering as a heuristic.
        args.max_per_source = max(80, args.limit * 5)
    
    if args.verbose or args.limit > 50:
        _stderr(f"[INFO] Configuration: limit={args.limit}, max_per_source={args.max_per_source}, sources={sources}")

    all_results: list[Paper] = []
    for src in sources:
        try:
            if src == "arxiv":
                all_results.extend(
                    fetch_arxiv(
                        query=query,
                        mode=mode,
                        max_results=args.max_per_source,
                        raw_query=bool(args.arxiv_raw),
                        page_size=int(args.arxiv_page_size),
                    )
                )
            elif src in ("semanticscholar", "semantic-scholar", "s2"):
                if mode == "author":
                    all_results.extend(
                        fetch_semantic_scholar_author(
                            author=query, max_results=args.max_per_source, api_key=args.s2_api_key
                        )
                    )
                else:
                    all_results.extend(
                        fetch_semantic_scholar_keyword(
                            query=query, max_results=args.max_per_source, api_key=args.s2_api_key
                        )
                    )
            else:
                _stderr(f"[WARN] Unsupported source ignored: {src}")
        except Exception as e:
            _stderr(f"[WARN] Failed fetching {src}: {e}")

    if args.verbose:
        _stderr(f"[INFO] fetched raw: {len(all_results)}")

    # Filter by date range (keep unknown dates).
    filtered: list[Paper] = []
    for p in all_results:
        d = _parse_date(p.publication_date) or (_parse_date(str(p.year)) if p.year else None)
        if _in_range(d, since, until):
            filtered.append(p)
    if args.verbose:
        _stderr(f"[INFO] after date filter: {len(filtered)} (since={since} until={until})")

    # Dedupe.
    by_key: dict[tuple[str, str], Paper] = {}
    for p in filtered:
        key = p.dedupe_key()
        if key in by_key:
            by_key[key] = _merge_prefer_richer(by_key[key], p)
        else:
            by_key[key] = p
    deduped = list(by_key.values())

    needs_enrich = bool(args.enrich_s2) or any(
        [
            args.sort != "date",
            int(args.min_citations) > 0,
            bool(args.venue_contains),
            bool(args.exclude_venue_contains),
            bool(args.require_venue),
        ]
    )
    enrichment_calls = 0
    if needs_enrich:
        enrichment_calls = _enrich_with_semantic_scholar_batch(
            deduped,
            api_key=args.s2_api_key,
            batch_size=int(args.s2_batch_size),
            sleep_s=float(args.s2_sleep),
            max_batches=int(args.s2_max_batches) if args.s2_max_batches is not None else None,
            verbose=bool(args.verbose),
        )

    venue_terms = [s.strip().lower() for s in (args.venue_contains or "").split(",") if s.strip()]
    exclude_venue_terms = [s.strip().lower() for s in (args.exclude_venue_contains or "").split(",") if s.strip()]

    filtered2: list[Paper] = []
    for p in deduped:
        venue = (p.venue or "").lower()
        if venue_terms and not any(t in venue for t in venue_terms):
            continue
        if exclude_venue_terms and any(t in venue for t in exclude_venue_terms):
            continue
        if args.require_venue and not (p.venue or "").strip():
            continue
        if int(args.min_citations) > 0 and int(p.citation_count or 0) < int(args.min_citations):
            continue
        filtered2.append(p)

    # Sort.
    def paper_date(p: Paper) -> dt.date | None:
        return _parse_date(p.publication_date) or (_parse_date(str(p.year)) if p.year else None)

    def citations_per_year(p: Paper) -> float:
        d = paper_date(p)
        if not d:
            return float(int(p.citation_count or 0))
        days = max(1, (dt.date.today() - d).days)
        return float(int(p.citation_count or 0)) / (days / 365.25)

    def sort_key(p: Paper) -> tuple[float, int]:
        d = paper_date(p)
        ts = int(dt.datetime.combine(d, dt.time.min).timestamp()) if d else 0
        cites = int(p.citation_count or 0)
        if args.sort == "date":
            return (float(ts), cites)
        if args.sort == "citations_per_year":
            return (citations_per_year(p), ts)
        return (float(cites), ts)

    filtered2.sort(key=sort_key, reverse=True)
    final = filtered2[: max(0, int(args.limit))]

    out_path = Path(args.out)
    _ensure_parent(out_path)
    payload = {
        "meta": {
            "retrievedAt": dt.datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "query": args.query,
            "author": args.author,
            "sources": sources,
            "since": since.isoformat(),
            "until": until.isoformat(),
            "rawCount": len(all_results),
            "filteredCount": len(filtered),
            "dedupedCount": len(deduped),
            "finalCount": len(final),
            "sort": args.sort,
            "minCitations": int(args.min_citations),
            "venueContains": venue_terms,
            "excludeVenueContains": exclude_venue_terms,
            "requireVenue": bool(args.require_venue),
            "enrichS2": bool(args.enrich_s2),
            "s2EnrichmentCalls": enrichment_calls,
        },
        "papers": [p.to_dict() for p in final],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.out_md:
        md_path = Path(args.out_md)
        _ensure_parent(md_path)
        md_path.write_text(_to_markdown_table(final), encoding="utf-8")

    if args.verbose:
        _stderr(f"[OK] wrote: {out_path}")
        if args.out_md:
            _stderr(f"[OK] wrote: {args.out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
