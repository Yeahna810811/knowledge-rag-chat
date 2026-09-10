# distractors/ — 检索评测干扰语料

本目录存放**故意与本项目无关**的技术文档，用于检索评测（见 `../EVALUATION_AUDIT.md`）。

## 为什么需要干扰语料

项目自身语料只有 8 篇（切分后约 10 chunk）。若只用真实语料评测，
检索器即使随机猜测也能拿到很高的 Hit@3，指标完全失真（饱和）。
加入本目录的无关文档后，语料扩大到 59 chunk（随机基线 Hit@3 ≈ 5.1%），
才能真实测出检索器「区分相关/无关文档」的能力——包括在
语义相近但事实不同的干扰项面前（如 FAISS vs Milvus、8000 vs 8080 端口）是否混淆。

## 约定

- 本目录文档描述的均为**假想方案**，参数与本项目无关，**不得作为本项目的事实依据**；
- 评测标注（`eval_dataset_*.jsonl`）的 evidence 全部来自 `../corpus/`，
  `../validate_distractors.py` 会校验干扰块与 evidence 无污染（包含度 < 0.85）；
- 修改本目录后需重新运行 `../build_extended_corpus.py` 重新生成 `../extended_noise/`。
