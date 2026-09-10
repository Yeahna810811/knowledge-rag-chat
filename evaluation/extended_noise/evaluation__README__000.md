# RAG Benchmark

`knowledge-rag-chat` 的检索与回答质量基准测试，由 `evaluate_rag.py` 驱动。
基准运行时把语料写入**独立的临时 FAISS 索引**（`tempfile.TemporaryDirectory`），不会触碰应用正式索引 `frontend/local_rag/data/faiss_index/`。

## 评估的指标

对应 `evaluate_rag.py` 的产出（详见 `summary` 结构）：

| 维度 | 指标 | 判定逻辑（代码位置） |
|---|---|---|
| 检索 | `hit@K` / `mrr` | `first_relevant_rank()`：对可答题，遍历检索结果找第一条相关文档的排名 |
| 回答 | `basic_correctness` | `basic_answer_correct()`：规范化后 reference_answer 是否包含在生成答案中；否则提取答案中的数字/英文/CJK 关键词，取前 3 个逐个比对 |