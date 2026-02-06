#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe(s: Any) -> str:
    return "" if s is None else str(s)


def _parse_date(value: Any) -> dt.date | None:
    if value in (None, ""):
        return None
    s = str(value).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            parsed = dt.datetime.strptime(s, fmt)
            if fmt == "%Y":
                return dt.date(parsed.year, 1, 1)
            if fmt == "%Y-%m":
                return dt.date(parsed.year, parsed.month, 1)
            return parsed.date()
        except ValueError:
            continue
    return None


def _citations_per_year(p: dict[str, Any], *, as_of: dt.date) -> str:
    citations = p.get("citationCount")
    if citations in (None, ""):
        return ""
    try:
        citations_i = int(citations)
    except Exception:
        return ""

    d = _parse_date(p.get("publicationDate"))
    if d is None:
        # If we only know the year (or date is missing), keep blank to avoid
        # misleading "per-year" numbers.
        return ""

    days = max(1, (as_of - d).days)
    value = citations_i / (days / 365.25)
    return f"{value:.2f}"


def _paper_type(p: dict[str, Any]) -> str:
    venue = str(p.get("venue") or "").strip()
    if not venue or "arxiv" in venue.lower():
        return "preprint"
    return "published"


def _md_escape_table(text: str) -> str:
    # Escape for Markdown table cells.
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("]", "\\]")

def _paper_ref(i: int, p: dict[str, Any]) -> str:
    authors = p.get("authors") or []
    authors_s = ", ".join(authors[:8]) + (" et al." if len(authors) > 8 else "")
    title = _safe(p.get("title")).strip()
    venue = _safe(p.get("venue")).strip()
    pub = _safe(p.get("publicationDate")).strip() or _safe(p.get("year")).strip()
    url = _safe(p.get("url")).strip()
    doi = _safe(p.get("doi")).strip()
    arxiv_id = _safe(p.get("arxivId")).strip()
    extras = []
    if venue:
        extras.append(venue)
    if pub:
        extras.append(pub)
    if doi:
        extras.append(f"DOI: {doi}")
    if arxiv_id:
        extras.append(f"arXiv: {arxiv_id}")
    extra_s = "；".join(extras)
    suffix = f"（{extra_s}）" if extra_s else ""
    return f"[{i}] {authors_s}. {title}. {url}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a structured multimodal literature review report.")
    parser.add_argument("--papers", required=True, help="Input JSON from fetch_papers.py.")
    parser.add_argument("--out", default="artifacts/report.md", help="Output Markdown path.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Use only top N papers from the input (default: 5).",
    )
    parser.add_argument(
        "--title",
        help="Optional report title. Default: auto from query/author + time range.",
    )
    args = parser.parse_args()

    papers_path = Path(args.papers)
    payload = json.loads(papers_path.read_text(encoding="utf-8"))
    meta = payload.get("meta") or {}
    papers_all = payload.get("papers") or []
    papers = papers_all[: max(0, int(args.top_n))]

    query = meta.get("query")
    author = meta.get("author")
    topic = author or query or "（未命名主题）"
    since = meta.get("since") or ""
    until = meta.get("until") or ""
    retrieved_at = meta.get("retrievedAt") or dt.datetime.now().isoformat(timespec="seconds")
    sources = ", ".join(meta.get("sources") or [])
    as_of = _parse_date(meta.get("until")) or _parse_date(str(retrieved_at)[:10]) or dt.date.today()

    title = args.title or f"{topic} 的多模态研究最新进展（{since}–{until}）"
    tldr = "（由模型生成）用一句话给出本次调研的结论：趋势概括 + 关键瓶颈 + 下一步最优先行动。"

    # Reference table (single block)
    table_lines = [
        "| # | 标题 | 作者 | Venue | 发表时间 | 引用 | 引用/年 | 类型 | DOI | arXivId | 来源 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, p in enumerate(papers, 1):
        raw_title = _safe(p.get("title")).strip()
        url = _safe(p.get("url")).strip()
        title_cell = _md_escape_table(raw_title)
        if url:
            title_cell = f"[{title_cell}]({url})"

        authors = p.get("authors") or []
        authors_s = ", ".join([str(a) for a in authors[:5] if a]) + (" et al." if len(authors) > 5 else "")
        authors_cell = _md_escape_table(authors_s)

        venue_cell = _md_escape_table(_safe(p.get("venue")).strip())
        date_cell = _md_escape_table(_safe(p.get("publicationDate")).strip() or _safe(p.get("year")).strip())
        citations_cell = _md_escape_table(_safe(p.get("citationCount")).strip())
        cpy_cell = _citations_per_year(p, as_of=as_of)
        type_cell = _paper_type(p)
        doi_cell = _md_escape_table(_safe(p.get("doi")).strip())
        arxiv_cell = _md_escape_table(_safe(p.get("arxivId")).strip())
        source_cell = _md_escape_table(_safe(p.get("source")).strip())

        table_lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    title_cell,
                    authors_cell,
                    venue_cell,
                    date_cell,
                    citations_cell,
                    cpy_cell,
                    type_cell,
                    doi_cell,
                    arxiv_cell,
                    source_cell,
                ]
            )
            + " |"
        )

    out = f"""# {title}

> **TL;DR（行动建议）**：{tldr}

## 研究背景和意义

- （用 1–3 段概括：为什么重要、外部驱动、你的检索覆盖边界）

**检索信息**
- 检索日期：{retrieved_at}
- 时间范围：{since}–{until}
- 检索源：{sources}
- 检索对象：{topic}
- 返回/筛选篇数：raw={meta.get('rawCount')} / filtered={meta.get('filteredCount')} / deduped={meta.get('dedupedCount')} / final={meta.get('finalCount')}
- 用于报告篇数：{len(papers)}

## 研究对象

- （明确任务/模态组合/应用场景与边界）

## 解决的问题

- （列 3–6 条问题：对齐/推理/数据/评测/安全等，并给出指标或典型失败案例）

## 进展（方法）

（按方法谱系组织 3–5 个小节；每节包含：核心思路 + 代表论文（按时间）+ 优势/代价 + 未解决点）

## 结论与洞察

- （趋势共识 2–5 条）
- （短板与风险 2–5 条）
- （未来 3–6 个月可操作建议 2–6 条）

### 参考文献表格（含影响力指标）

{chr(10).join(table_lines)}
"""

    out_path = Path(args.out)
    _ensure_parent(out_path)
    out_path.write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
