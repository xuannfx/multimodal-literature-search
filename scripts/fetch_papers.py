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


def _http_get_json(
    url: str, timeout_s: int = 30, extra_headers: dict[str, str] | None = None
) -> Any:
    body = _http_get(url, timeout_s=timeout_s, extra_headers=extra_headers)
    return json.loads(body.decode("utf-8"))


def _normalize_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip().lower()
    title = re.sub(r"[^a-z0-9]+", "", title)
    return title


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
            return ("arxiv", self.arxiv_id.lower())
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


def fetch_arxiv(*, query: str, mode: str, max_results: int) -> list[Paper]:
    if mode == "author":
        search_query = f'au:"{query}"'
    else:
        search_query = f'all:"{query}"'

    params = {
        "search_query": search_query,
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    xml_bytes = _http_get(url)

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_bytes)
    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ns):
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


def _to_markdown_table(papers: list[Paper]) -> str:
    headers = ["#", "标题", "发表时间", "来源", "Venue", "链接"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for i, p in enumerate(papers, 1):
        pub = p.publication_date or (str(p.year) if p.year else "")
        venue = p.venue or ""
        url = p.url or ""
        title = p.title.replace("|", "\\|")
        lines.append(f"| {i} | {title} | {pub} | {p.source} | {venue} | {url} |")
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
        default=80,
        help="Fetch up to N papers per source before dedupe/filter.",
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

    all_results: list[Paper] = []
    for src in sources:
        try:
            if src == "arxiv":
                all_results.extend(
                    fetch_arxiv(query=query, mode=mode, max_results=args.max_per_source)
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

    # Sort by date desc, then citation_count desc.
    def sort_key(p: Paper) -> tuple[int, int]:
        d = _parse_date(p.publication_date) or (_parse_date(str(p.year)) if p.year else None)
        ts = int(dt.datetime.combine(d, dt.time.min).timestamp()) if d else 0
        citations = int(p.citation_count or 0)
        return (ts, citations)

    deduped.sort(key=sort_key, reverse=True)
    final = deduped[: max(0, int(args.limit))]

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
