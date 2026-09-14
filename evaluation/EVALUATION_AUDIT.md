# RAG 评测体系审计与最终实验报告

本文记录 `knowledge-rag-chat` 项目的检索评测设计、策略对比、
统计显著性检验以及最终端到端 RAG 评测。

> 当前正式结论以 2026-09 最终 Benchmark 为准。
> 历史实验仅用于记录评测体系的迭代过程，不作为当前项目结论。

---

## 1. 最终评测配置

### 数据集

正式评测集：

```text
evaluation/eval_dataset_v3.jsonl