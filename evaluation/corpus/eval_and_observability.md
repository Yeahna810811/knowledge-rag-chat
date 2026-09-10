# 评测与可观测性说明（Benchmark v3）

检索评测脚本为 evaluation/evaluate_rag.py。

评测数据集为 evaluation/eval_dataset.jsonl，默认包含 50 条题目。

数据集中包含 10 条知识库未覆盖的不可答题，用于检验 Safe Refusal 拒答能力。

检索侧指标为 Hit@K 与 MRR。

生成质量可选用 Ragas 评测 Faithfulness 与 Answer Relevancy。

多 Agent 链路为 document_parse_agent、retrieval_agent、generation_agent。

问答接口返回的 agent_trace 字段记录了本次回答实际调用了哪些 Agent。

知识库中没有证据时，系统会提示联系人工客服，而不是凭空编造答案。

端到端评测还会统计检索延迟与端到端延迟。
