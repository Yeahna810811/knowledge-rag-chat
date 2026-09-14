# Evaluation Results

本目录保存 `knowledge-rag-chat` 的正式评测结果与历史实验记录。

当前正式结论统一参考：

```text
evaluation/EVALUATION_AUDIT.md
```

---

## retrieval_final

正式 Retrieval A/B Benchmark：

```text
evaluation/results/retrieval_final/
├── ab_retrieval.json
├── ab_retrieval_details.csv
└── significance_test.txt
```

配置：

```text
100 questions
80 answerable
20 unanswerable

44 corpus files
46 chunks

chunk_size = 500
chunk_overlap = 50

Dense = BAAI/bge-small-zh-v1.5
Sparse = BM25
Metric = bigram
```

结果：

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.625 | 0.775 | 0.838 | 0.7060 |
| **BM25** | **0.825** | **0.950** | **0.975** | **0.8869** |
| Hybrid 1:1 | 0.700 | 0.887 | 0.925 | 0.7927 |
| Hybrid + MMR | 0.700 | 0.875 | 0.938 | 0.7860 |

Dense vs BM25：

```text
BM25 - Dense Hit@1 = +20 percentage points

BM25 - Dense ΔMRR = +0.1808
Bootstrap p < 0.001
Sign Test p = 0.003
```

---

## rag_final

最终 BM25-RAG + LLM-as-a-Judge：

```text
evaluation/results/rag_final/
├── evaluation_details.csv
└── evaluation_summary.json
```

Retrieval：

```text
Hit@1 = 0.8125
Hit@3 = 0.9500
Hit@5 = 0.9750
MRR   = 0.8806
```

LLM-as-a-Judge：

```text
Correctness  = 0.970
Faithfulness = 0.993
Relevance    = 0.969
Safe Refusal = 1.000
```

Safe Refusal 仅统计 20 道知识库外不可答题。

Latency：

```text
Retrieval Avg = 0.82 ms
Retrieval P95 = 1.13 ms

End-to-End Avg = 3.49 s
End-to-End P95 = 5.33 s
```

---

## archive

```text
evaluation/results/archive/prompt_ablation/
```

保存早期 Prompt 优化前后的实验结果。

这些文件：

```text
evaluation_details_v2_before_prompt.csv
evaluation_details_v2_after_prompt.csv
evaluation_summary_v2_before_prompt.json
evaluation_summary_v2_after_prompt.json
```

属于历史实验。

由于当时使用的语料规模、相关性判定和 Benchmark 口径与当前版本不同，因此：

```text
不作为当前 Retrieval 质量证据
```

只用于保留项目实验迭代过程。