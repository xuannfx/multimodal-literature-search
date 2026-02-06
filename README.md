# multimodal-literature-search

多模态（视觉-语言 / 音频-语言 / 视频等）方向的**最新论文检索、去重、筛选**与**结构化综述**生成 Skill（仅作为 Codex Skill 使用）。

## 使用方式

在 Codex 中直接提出需求即可，推荐一次性给全这些信息：

- 检索主题关键词 *或* 作者/团队名称（二选一）
- 检索源（如 `arXiv`, `Semantic Scholar`；也可写 `OpenReview/CVF/ACL/PubMed` 等）
- 可选：时间范围（默认近 18 个月）、返回篇数（默认 5）、输出语言（默认中文）、聚焦方向（可选）

完整交互说明、默认值与产出规范见 `SKILL.md`。

## 能输出什么

- 按固定 6 段结构生成综述（标题 / 背景与意义 / 研究对象 / 解决的问题 / 进展（方法） / 结论与洞察）
- 报告内包含：TL;DR（1 句行动建议）、检索信息（日期/范围/来源/计数）、单表参考文献（含发表时间与链接/引用）

固定结构与检查清单见 `references/review-structure.md`。

## 检索源支持

- 工具侧（结构化拉取）当前支持：`arXiv`、`Semantic Scholar`
- 当用户指定 `OpenReview / CVF / ACL Anthology / PubMed` 等来源时，按 `references/source-playbook.md` 的查询模板改用网页/站内检索进行补齐，并在报告“检索信息”中写清实际使用的来源与方法

## 配置（可选）

Semantic Scholar 如遇频控，建议配置 API Key 环境变量：`SEMANTIC_SCHOLAR_API_KEY`。

## 目录

- `SKILL.md`：Skill 的完整工作流与输出规范（推荐先读）
- `references/review-structure.md`：固定综述结构模板
- `references/source-playbook.md`：不同检索源查询模板与策略

## 备注与限制

- “最新”默认解释为近 18 个月；若用户使用相对时间（如“2025 年以来/最近三个月”），会在报告中转为明确日期区间并写明检索日期
- 预印本可能后续被接收/更新版本；报告会提示该不确定性
