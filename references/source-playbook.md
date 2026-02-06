# 检索源 Playbook（模板 + 常见坑）

## 总原则（先定范围）

- 优先把“最新”落到**明确时间范围**（例如最近 18 个月，或 `2025-01-01` 至今），并在报告写明检索日期。
- 同一个主题至少用 2 个来源交叉验证（例如 `arXiv + Semantic Scholar`）。
- 对“多模态”相关性做二次筛选：很多论文标题带“multimodal”但实际是数据融合/多视角等，需看摘要/方法。

## arXiv

适用：预印本、最新进展最及时；但 venue/接收信息不全。

建议查询写法：
- 关键词：`multimodal` + 子方向词（如 `vision-language` / `video-language` / `audio-language` / `multimodal agent`）
- 作者：用英文姓名（必要时加中间名缩写）

常见坑与补救：
- 同名作者多：结合机构/合作者/论文主题核对
- 版本更新：以 `published`/提交时间为主，必要时注明“后续版本可能更新”

## Semantic Scholar

适用：聚合多个来源、可拿到 citationCount/venue/DOI 等；适合做去重与补全元数据。

建议：
- 关键词检索后，按 `publicationDate/year` 排序，再人工筛选
- 作者/团队检索：先确认 authorId（同名会很多）；团队名可当关键词处理

常见坑与补救：
- publicationDate 可能缺失：降级用 year；或从 arXiv/publisher 页面补齐
- 结果噪声：用“必含词/排除词”迭代 query

## PubMed（脚本未内置时的策略）

适用：生物医学多模态（影像+报告、医学语音、EHR+文本）。

建议搜索：
- 站内：`(multimodal OR vision-language) AND (radiology OR pathology OR clinical)` + 时间过滤
- 或网页：`site:pubmed.ncbi.nlm.nih.gov <关键词> 2025..2026`

## OpenReview（脚本未内置时的策略）

适用：ICLR/NeurIPS/ICML 等的评审稿、poster/oral；最新但链接结构复杂。

建议：
- 站内 search：`<关键词>` + conference year
- 网页：`site:openreview.net <关键词> (ICLR OR NeurIPS OR ICML) 2025`

## CVF / ACL Anthology（脚本未内置时的策略）

适用：CVPR/ICCV/ECCV 与 ACL/EMNLP/NAACL 的正式出版记录。

建议：
- CVF：`site:openaccess.thecvf.com <关键词> 2025`（或直接在 CVF 会议信息页搜索）
- ACL：`site:aclanthology.org <关键词> 2025`（注意同义词：`vision-language` / `multimodal` / `grounding` 等）

## IEEE Xplore / ACM DL / Springer（可选）

适用：工程与应用型工作；但检索权限/付费墙可能影响可见性。

策略：
- 优先用 Semantic Scholar 补齐 DOI，再跳转 publisher 页面确认发表时间与 venue

