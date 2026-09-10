# results/ 结果文件口径说明

**引用结论只看 `evaluation/EVALUATION_AUDIT.md`，本目录只放原始产物。**

| 文件 | 口径 | 是否可引用 |
|---|---|---|
| `ab_retrieval.json`、`ab_retrieval_details.csv` | 当前口径：100 题（80 可答）+ 45 chunk 语料，真实 bge-small-zh 向量，bigram 相关判定 | ✅ 与审计报告一致 |
| `evaluation_summary_v2_*.json`、`evaluation_details_v2_*.csv` | **历史口径**：5 chunk 语料 + 字符包含度判定。随机基线 Hit@3 高达 60%，指标已饱和，仅作「Prompt 优化前后对比」的过程留档 | ❌ 不作为检索质量证据 |

历史文件保留的原因：它们是「Prompt 优化前后」那次对比实验的原始记录，
换成新口径后该对比无法重跑（当时用的是旧语料），删掉就等于丢掉那次实验的痕迹。
