---
name: multimodal-literature-search
description: 检索多模态（视觉-语言/音频-语言/视频等）方向的最新论文；根据用户给定的主题关键词或作者/团队名称与检索源（如 arXiv、Semantic Scholar、PubMed、OpenReview、CVF/ACL）收集、去重、筛选论文，并按固定综述结构输出报告，同时提供每篇论文的引用信息与发表时间。
---

# 多模态最新文献检索与综述

## 快速开始

1. 收集用户输入：`主题关键词/作者（团队）` + `检索源`（必要），其余走默认。
2. 用 `scripts/fetch_papers.py` 拉取并去重，得到 `papers.json`。
3. 基于 `papers.json` 的标题/摘要做筛选与归类；按 `references/review-structure.md` 生成综述报告，并附论文清单（含发表时间与链接/引用）。

## 输入与默认值

- **必填输入**
  - `query`：检索主题关键词 *或* 作者/团队名称（二选一即可）
  - `sources`：检索源列表（例如：`arxiv, semanticscholar`；也可写 `PubMed/OpenReview/CVF/ACL` 等）
- **可选输入（未提供则使用默认）**
  - `time_range`：默认最近 **18 个月**（用户说“最新/最近/2025 年以来”等时，优先转为明确日期区间并在报告写清检索日期）
  - `limit`：默认 **5** 篇（只用 top5 生成报告；如需更全面，再提高 limit）
  - `language`：默认中文
  - `focus`：可选聚焦（例如：`VLM 对齐` / `视频理解` / `多模态 Agent` / `3D+LLM` / `医疗多模态`）

## 工作流（检索 → 筛选 → 写作）

### 1) 明确检索意图

- 判断是 **关键词检索** 还是 **作者/团队检索**：
  - 关键词：补充同义词/缩写（例如 `vision-language` / `VLM` / `multimodal` / `image-text` / `video-language` 等）
  - 作者/团队：尽量使用更精确的名称（英文全称优先）；如同名较多，需要在结果中核对机构/合作者
- 若用户给的 `sources` 含脚本暂不支持的站点（如 `OpenReview`、`CVF`、`ACL Anthology`），按 `references/source-playbook.md` 的查询模板改用网页检索/站内检索补齐。

### 2) 拉取论文（可复用脚本）

优先用脚本得到可审计的结构化结果，再进入写作：

```bash
python3 skills/multimodal-literature-search/scripts/fetch_papers.py \
  --query "vision-language alignment" \
  --sources arxiv,semanticscholar \
  --months 18 \
  --limit 5 \
  --out artifacts/papers.json \
  --out-md artifacts/papers.md
```

作者/团队检索示例：

```bash
python3 skills/multimodal-literature-search/scripts/fetch_papers.py \
  --author "Google DeepMind" \
  --sources semanticscholar \
  --months 18 \
  --limit 5 \
  --out artifacts/papers.json \
  --out-md artifacts/papers.md
```

### 3) 去重、筛选与归类

- **去重规则（最低要求）**：优先按 `DOI` / `arXiv ID` 去重；无则用规范化标题去重。
- **筛选规则（建议）**：只保留“明确涉及 ≥2 种模态建模/对齐/联合训练/跨模态推理/跨模态评测”的工作；把纯单模态论文移到“相关但非多模态核心”。
- **归类维度（写作要用）**：模态组合（图文/视频文/音频文/3D+文…）、训练范式（对比学习/生成式/指令微调/RLHF）、对齐机制（adapter/投影/融合/检索增强）、数据与评测（基准/指标/安全与偏见）。

### 4) 写综述报告（必须按固定结构）

按 `references/review-structure.md` 输出报告，结构固定为：

1. 标题
2. 研究背景和意义
3. 研究对象
4. 解决的问题
5. 进展（方法）
6. 结论与洞察

## 输出规范（必须满足）

- 报告中必须包含“检索信息”一段：检索日期、时间范围、检索源、关键词/作者字符串、返回/筛选篇数。
- 必须附“论文清单”，每条至少包含：标题、作者、发表时间（或 arXiv 提交时间）、来源（arXiv/期刊/会议）、链接。
- 若用户强调“最新”，排序优先按发表/提交时间倒序；必要时说明“预印本可能后续被接收/版本更新”。
- 引用格式：可用编号 `[1]`… 并在末尾给出参考文献列表；同一条目给出 `URL`（如有 `DOI` 更好）。

## 资源

### scripts/

- `scripts/fetch_papers.py`：从 `arXiv`/`Semantic Scholar` 拉取论文、归一化字段、去重、按时间排序并导出 JSON/Markdown。
- 如遇到 `Semantic Scholar` 频控，可设置环境变量 `SEMANTIC_SCHOLAR_API_KEY` 或传参 `--s2-api-key`（有 key 时）。
- `scripts/make_report.py`：从 `papers.json` 生成综述 Markdown 框架（含论文表格与参考文献条目占位），便于继续补全内容。

生成报告骨架示例：

```bash
python3 skills/multimodal-literature-search/scripts/make_report.py \
  --papers artifacts/papers.json \
  --out artifacts/report.md \
  --top-n 5
```

### references/

- `references/review-structure.md`：综述结构的写作提示与检查清单（按你给定的 6 段结构）。
- `references/source-playbook.md`：不同检索源的查询模板、常见坑与补救策略。
