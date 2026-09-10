答。每行是一个 JSON 对象，必填字段（`load_dataset()` 会校验缺失并报错）：

```json
{"id": "q001", "question": "...", "reference_answer": "...", "answerable": true,
 "evidence": "标准证据原文", "source_contains": "rag_architecture.md"}
```

| 字段 | 必填 | 用途 |
|---|---|---|
| `id` / `question` / `answerable` | 是 | 题号、问题、是否可回答 |
| `reference_answer` | 是 | 参考答案，用于 `basic_answer_correct` 和 judge 打分 |
| `evidence` | 否 | 标准证据原文，命中判定的主要依据 |
| `source_contains` | 否 | 兜底：无 evidence 时按文件名匹配来源 |