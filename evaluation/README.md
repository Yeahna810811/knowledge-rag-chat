# RAG Evaluation

本目录包含 `knowledge-rag-chat` 的 Retrieval Benchmark、统计显著性检验、Runtime Benchmark 和 End-to-End RAG 自动评测。

正式项目结论以：

```text
evaluation/EVALUATION_AUDIT.md
```

和：

```text
evaluation/results/retrieval_final/
evaluation/results/rag_final/
```

为准。

---

## 1. 正式评测集

正式数据集：

```text
evaluation/eval_dataset_v3.jsonl
```

规模：

```text
100 questions
├── 80 answerable
└── 20 unanswerable
```

字段：

```json
{
  "id": "q001",
  "question": "...",
  "reference_answer": "...",
  "answerable": true,
  "evidence": "...",
  "source_contains": "..."
}
```

字段用途：

| Field | Usage |
| --- | --- |
| `id` | 问题 ID |
| `question` | 用户问题 |
| `reference_answer` | 参考答案 |
| `answerable` | 是否属于知识库可答范围 |
| `evidence` | Ground Truth 证据 |
| `source_contains` | 目标来源文件 |

20 道不可答题主要用于 Safe Refusal 评测。

---

## 2. Corpus

基础语料：

```text
evaluation/corpus/
```

当前包含 8 个 Markdown 文件。

Hard Negative：

```text
evaluation/extended_noise/
```

由：

```bash
python evaluation/build_extended_corpus.py
```

生成。

正式实验运行时：

```text
44 corpus files
→ 46 chunks
```

切分参数：

```text
chunk_size = 500
chunk_overlap = 50
```

Hard Negative 来自项目自身文档，用于增加和正确答案具有相似技术词汇、但实际无关的干扰内容。

> `extended_noise` 是派生 Benchmark 语料。项目文档发生变化后重新生成它，具体 chunk 和最终指标可能出现小范围漂移。

---

## 3. Retrieval 指标

正式 Retrieval Benchmark 使用：

```text
Hit@1
Hit@3
Hit@5
MRR
```

相关性判定使用：

```text
bigram containment
```

而不是早期的字符级 containment。

原因是字符级匹配在中文长文本中容易产生假阳性。

---

## 4. Retrieval A/B Benchmark

脚本：

```text
evaluation/evaluate_retrieval_ab.py
```

正式命令：

```bash
python evaluation/evaluate_retrieval_ab.py \
  --dataset evaluation/eval_dataset_v3.jsonl \
  --dense localbge \
  --include-noise \
  --metric bigram \
  --ks 1 3 5 \
  --chunk-size 500 \
  --chunk-overlap 50 \
  --output-dir evaluation/results/retrieval_final
```

正式结果：

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense / BGE | 0.625 | 0.775 | 0.838 | 0.7060 |
| **BM25** | **0.825** | **0.950** | **0.975** | **0.8869** |
| Hybrid 1:1 | 0.700 | 0.887 | 0.925 | 0.7927 |
| Hybrid + MMR | 0.700 | 0.875 | 0.938 | 0.7860 |

正式 Retrieval 策略：

```text
BM25
```

---

## 5. 统计显著性检验

脚本：

```text
evaluation/significance_test.py
```

运行：

```bash
python evaluation/significance_test.py \
  --dataset evaluation/eval_dataset_v3.jsonl \
  --rounds 10000
```

Dense 相对 BM25：

```text
ΔMRR = -0.1808
95% CI = [-0.2848, -0.0781]

p(bootstrap) < 0.001
p(sign) = 0.003
```

因此当前 Benchmark 对：

```text
BM25 > Dense
```

提供了统计支持。

不同权重的 Hybrid 当前均未超过 BM25。

---

## 6. Runtime Benchmark

脚本：

```text
evaluation/benchmark_runtime.py
```

运行：

```bash
python evaluation/benchmark_runtime.py --queries 46
```

当前本地实验：

| Metric | BM25 | Dense |
| --- | ---: | ---: |
| Model Load | 0 ms | 188.73 ms |
| Ingest | 4.17 ms | 5359.55 ms |
| Query Avg | 0.06 ms | 22.88 ms |
| BM25 Reload | 6.52 ms | - |

注意：

Dense Runtime 使用本地 NumPy BGE 推理实现，因此绝对耗时不能等同于生产 Torch / ONNX / GPU 性能。

这里用于比较不同 Retrieval 架构在当前本地环境下的成本量级，而不是宣称：

```text
“把 Dense 从 5359 ms 优化成了 BM25 的 4 ms”
```

BM25 和 Dense 是两种不同方案。

---

## 7. End-to-End RAG Benchmark

脚本：

```text
evaluation/evaluate_rag.py
```

支持：

```text
--retriever dense
--retriever bm25
```

正式 BM25 评测：

```bash
python evaluation/evaluate_rag.py \
  --retriever bm25 \
  --judge \
  --output-dir evaluation/results/rag_final
```

当前默认数据集：

```text
evaluation/eval_dataset_v3.jsonl
```

默认语料：

```text
evaluation/corpus/
+
evaluation/extended_noise/
```

---

## 8. End-to-End 指标

Retrieval：

```text
Hit@1 = 0.8125
Hit@3 = 0.9500
Hit@5 = 0.9750
MRR   = 0.8806
```

Rule-Based：

```text
basic_correctness = 0.8125
refusal_accuracy  = 1.0000
```

LLM-as-a-Judge：

```text
Correctness  = 0.970
Faithfulness = 0.993
Relevance    = 0.969

Safe Refusal = 1.000
```

其中 Safe Refusal 仅统计：

```text
20 个 unanswerable questions
```

Judge 覆盖：

```text
judged_questions = 100
judged_unanswerable = 20
```

---

## 9. Latency

正式 BM25-RAG：

### Retrieval

```text
avg = 0.82 ms
P50 = 0.66 ms
P95 = 1.13 ms
```

### End-to-End

```text
avg = 3485.42 ms
P50 = 3289.74 ms
P95 = 5328.89 ms
```

---

## 10. LLM-as-a-Judge

开启：

```bash
--judge
```

后会真实调用配置的大模型。

默认 Judge Model：

```text
CHAT_MODEL
```

也可以显式指定：

```bash
--judge-model MODEL_NAME
```

Judge 输入包括：

```text
Question
Reference Answer
Retrieved Context
Generated Answer
```

输出：

```text
correctness
faithfulness
relevance
safe_refusal
```

LLM Judge 属于自动离线评测，不等同于人工专家评分。

如果 Generation Model 和 Judge Model 相同，需要考虑 self-evaluation bias。

---

## 11. Retrieval A/B 与 RAG Benchmark 的区别

Retrieval A/B：

```text
evaluation/evaluate_retrieval_ab.py
```

用于：

```text
比较 Dense / BM25 / Hybrid
↓
决定 Retrieval Strategy
```

正式 BM25：

```text
Hit@1 = 0.825
MRR   = 0.8869
```

End-to-End：

```text
evaluation/evaluate_rag.py
```

用于：

```text
Retrieval
+
Generation
+
Answer Evaluation
+
Latency
```

最终：

```text
Hit@1 = 0.8125
MRR   = 0.8806
```

两套数字属于不同实验链路，不混合引用。

---

## 12. Results

正式 Retrieval：

```text
evaluation/results/retrieval_final/
├── ab_retrieval.json
├── ab_retrieval_details.csv
└── significance_test.txt
```

正式 End-to-End：

```text
evaluation/results/rag_final/
├── evaluation_details.csv
└── evaluation_summary.json
```

历史实验：

```text
evaluation/results/archive/prompt_ablation/
```

---

## 13. 其他评测脚本

Ragas：

```bash
pip install -r evaluation/requirements-eval.txt
python evaluation/evaluate_with_ragas.py --limit 10
```

口语化问题数据集：

```text
evaluation/eval_dataset_oral_v3.jsonl
```

Query Rewrite 实验：

```text
evaluation/evaluate_query_rewrite.py
```

这些实验用于补充分析，不替代当前正式 Retrieval Benchmark。

---

## 14. 测试

```bash
python tests/test_retrieval.py
python tests/test_knowledge_retriever.py
```

此前结果：

```text
26 passed
38 passed
```

---

## 15. 注意事项

1. `--judge` 会真实调用大模型并产生 Token 消耗。
2. Hard Negative 来源于项目文档，因此文档更新后重新生成可能造成指标轻微漂移。
3. 当前 Dense 使用 `bge-small-zh-v1.5`，不能把当前结论外推为“BM25 永远优于 Dense”。
4. 当前正式结论统一参考 `EVALUATION_AUDIT.md`。
5. 历史 Prompt 结果不再作为当前 Retrieval 质量证据。