# Stage 5 最终回归报告

目的只有一个：

> 确认 Stage 1~4 的工程化改造（异步化、SSE、MySQL/Redis、可靠性与可观测性）
> **没有破坏原有的 RAG 检索质量**。

不是重新调参，不是把数字做得更好看。数据集、语料、指标口径、切分参数
全部沿用原样，只重跑。

---

## 1. 重跑了什么

| 项目 | 命令 | 状态 |
| --- | --- | --- |
| Retrieval A/B（100 题） | `python evaluation/evaluate_retrieval_ab.py --dataset evaluation/eval_dataset_v3.jsonl --dense localbge --include-noise --metric bigram --ks 1 3 5 --chunk-size 500 --chunk-overlap 50 --output-dir evaluation/results/retrieval_stage5` | ✅ 已跑 |
| RAG retrieval-only（100 题） | `python evaluation/evaluate_rag.py --dataset evaluation/eval_dataset_v3.jsonl --retriever bm25 --retrieval-only --ks 1 3 5 --output-dir evaluation/results/rag_stage5_retrieval_only` | ✅ 已跑 |
| RAG retrieval-only（修复后复跑） | 同上，`--output-dir evaluation/results/rag_stage5_after_fix` | ✅ 已跑（见 §5.3） |
| LLM Judge（全量 100 题） | `python evaluation/evaluate_rag.py --judge`（全量） | ⛔ **NOT RUN**（成本，见第 4 节） |
| LLM Judge（子集 20 题） | `python evaluation/evaluate_rag.py --retriever bm25 --judge --limit 20 --output-dir evaluation/results/rag_stage5_judge_subset` | ✅ 已跑（辅助，不可与历史值直接比较） |

环境：Apple M1 Pro / 16 GB，Python 3.9.13（项目 `.venv`）。
语料 44 篇 → 46 chunks（与原实验一致），切分 500/50 未改。

---

## 2. Retrieval A/B 回归（100 题）

| Strategy | 指标 | Previous | Stage 5 | Delta | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| dense | Hit@1 | 0.6250 | 0.6250 | +0.0000 | PASS |
| dense | Hit@3 | 0.7750 | 0.7750 | +0.0000 | PASS |
| dense | Hit@5 | 0.8375 | 0.8375 | +0.0000 | PASS |
| dense | MRR | 0.7060 | 0.7060 | +0.0000 | PASS |
| **bm25** | Hit@1 | 0.8250 | 0.8250 | +0.0000 | PASS |
| **bm25** | Hit@3 | 0.9500 | 0.9500 | +0.0000 | PASS |
| **bm25** | Hit@5 | 0.9750 | 0.9750 | +0.0000 | PASS |
| **bm25** | MRR | 0.8869 | 0.8869 | +0.0000 | PASS |
| hybrid(1:1) | Hit@1 | 0.7000 | 0.7000 | +0.0000 | PASS |
| hybrid(1:1) | Hit@3 | 0.8875 | 0.8875 | +0.0000 | PASS |
| hybrid(1:1) | Hit@5 | 0.9250 | 0.9250 | +0.0000 | PASS |
| hybrid(1:1) | MRR | 0.7927 | 0.7927 | +0.0000 | PASS |
| hybrid+mmr | Hit@1 | 0.7000 | 0.7000 | +0.0000 | PASS |
| hybrid+mmr | Hit@3 | 0.8750 | 0.8750 | +0.0000 | PASS |
| hybrid+mmr | Hit@5 | 0.9375 | 0.9375 | +0.0000 | PASS |
| hybrid+mmr | MRR | 0.7860 | 0.7860 | +0.0000 | PASS |

随机基线同样一致：`Hit@1 0.0217 / Hit@3 0.0652 / Hit@5 0.1087`。

**16 项指标全部完全一致（delta = 0.0000）。** 这不是"轻微波动"，
是完全复现——检索链路在这套确定性输入下不受 Stage 1~4 改动影响。

---

## 3. RAG retrieval-only 回归（100 题）

| 指标 | Previous (rag_final) | Stage 5 | Delta | 判定 |
| --- | ---: | ---: | ---: | --- |
| Hit@1 | 0.8125 | 0.8125 | +0.0000 | PASS |
| Hit@3 | 0.9500 | 0.9500 | +0.0000 | PASS |
| Hit@5 | 0.9750 | 0.9750 | +0.0000 | PASS |
| MRR | 0.8806 | 0.8806 | +0.0000 | PASS |

检索延迟（Stage 5 实测，仅 retrieval-only 路径）：
`avg 0.16 ms / P50 0.16 ms / P95 0.23 ms`。
历史 rag_final 记录的是 `avg 0.82 ms / P95 1.13 ms`——
两者都远小于任何端到端开销，差异属于机器负载与采样噪声，
**不构成检索质量或性能的回归**。

---

## 4. LLM Judge

### 4.1 全量 100 题：NOT RUN

原因（不是"跑不了"，是成本）：

* 当前 `.env` 里 `CHAT_MODEL=qwen3.8-max-0902`，单次生成实测约 **26 秒**（压测 P50 5.1~7.7 s，评测场景更长）。
* 全量重跑 = 100 次生成 + 100 次 Judge 调用，按当前模型约 **1 小时以上**，且会消耗显著额度。
* 更关键的是：即使跑完也**不能和历史值直接比较**——历史结果用的是另一个模型
  （历史 E2E avg 3.49 s，当前模型 4.66 s 量级），换了生成器就不是同一件事。

因此按"宁可不写，也不编造"的原则：

```text
Full 100-question LLM Judge: NOT RUN in Stage 5
```

历史结果完整保留在 `evaluation/results/rag_final/evaluation_summary.json`，
未做任何修改：

```text
Correctness  = 0.970
Faithfulness = 0.993
Relevance    = 0.969
Safe Refusal = 1.000   (20 道不可答题)
judged_questions = 100
```

### 4.2 子集 20 题（辅助证据）

只跑了数据集前 20 题，且**这 20 题全部是可答题**（不可答题集中在后段），
所以 Safe Refusal 没有覆盖：

| 指标 | Stage 5 子集 (n=20, 全可答) | 历史全量 (n=100) | 可比性 |
| --- | ---: | ---: | --- |
| Correctness | 1.000 | 0.970 | ❌ 不可比 |
| Faithfulness | 1.000 | 0.993 | ❌ 不可比 |
| Relevance | 1.000 | 0.969 | ❌ 不可比 |
| Safe Refusal | n/a（0 道不可答题） | 1.000 | ❌ 未覆盖 |
| basic_correctness | 0.850 | 0.8125 | ❌ 不可比 |
| E2E avg | 4664 ms | 3485 ms | ❌ 模型不同 |

**必须诚实说明的两点：**

1. 这 20 题的 Judge 打分**全部是 1.0，没有任何区分度**（取值集合就是 `{1.0}`）。
   在 n=20 且零方差的情况下，这个数字只能证明"链路能跑通、Judge 能返回结构化结果"，
   **不能用来宣称质量提升**。
2. 它和历史的 0.97 不构成"提升"——子集不同、模型不同、样本量差 5 倍。

---

## 5. 压测发现的两个真实缺陷 —— 已修复并复测

### 5.1 流式路径重复检索（已修复）

`frontend/local_rag/core/agents/orchestrator.py` 的 `astream()`：
检索被执行了两次——第一次的返回值被丢弃，第二次才包在错误处理里。

```python
# 修复前
retrieval = await anyio.to_thread.run_sync(
    lambda: self.retrieval_agent.run(question=question)   # 第一次，返回值被丢弃
)
try:
    retrieval = await anyio.to_thread.run_sync(
        lambda: self.retrieval_agent.run(question=question)   # 第二次
    )
except ClassifiedError:
    raise

# 修复后：只保留 try 里这一次
```

而**非流式 `ask()` 只检索一次**——所以这是流式路径独有的回归。

**修复方式**：`orchestrator.py` 删除 3 行（第一次调用）。

**证据 1 · 确定性的计数**（不依赖延迟，不受噪声影响）：

进程内探针 `evaluation/performance/probe_retrieval_count.py`：

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| 3 个全新问题各问 1 次 | `miss +3`**且** `hit +3` | `miss +3`, `hit 0`, `write +3` |
| 同一问题重复问 2 次 | `hit +4` | `hit +2`, `miss 0` |
| 非流式 `ask()` 对照 | `miss +1`, `hit 0` | `miss +1`, `hit 0`（未变） |

**证据 2 · 单元回归测试**：新增
`tests/test_orchestrator_retrieval_once.py`，**22 项断言全部通过**：
rag 模式检索恰好 1 次、chat 模式 0 次、事件顺序 `sources→trace→delta`、
异常仍归类为 `RETRIEVAL_ERROR` 且失败路径也只检索 1 次、
`_cache_hit_since()` 在 hit/miss/无检索/拿不到指标四种情况下分别返回
`True` / `False` / `None` / `None`。已接入 CI（`backend-smoke` job）。

### 5.2 `timing.cache_hit` 归因错误（随 5.1 一并修复）

同一个根因：因为一次请求有两次缓存查询，只要其中一次命中，
`_cache_hit_since()` 就返回 `True`，**首次查询也会被标成命中**。

修复 5.1 后，每次请求只会 +1 一个计数器（要么 hit 要么 miss），
归因自动恢复正确，**无需单独改代码**。已由 5.1 的单元用例 `[G]` 组钉死。

### 5.3 修复后的回归复跑

改了编排代码就必须复跑质量回归。100 题 retrieval-only 重跑
（`evaluation/results/rag_stage5_after_fix/`）：

| 指标 | 修复前 | 修复后 | Delta |
| --- | ---: | ---: | ---: |
| hit@1 | 0.8125 | 0.8125 | 0.0000 |
| hit@3 | 0.9500 | 0.9500 | 0.0000 |
| hit@5 | 0.9750 | 0.9750 | 0.0000 |
| mrr | 0.8806 | 0.8806 | 0.0000 |

questions 100 / chunks 46 不变。**零回归**——符合预期，
因为被删掉的那次检索的返回值本来就没人用。

### 5.4 修复带来了多少性能收益（交错 A/B 实测）

见 `evaluation/performance/README.md` §5.4、主 README §8.5。

方法：修复版（8010）与未修复版（8011）两个服务**交替**各跑 5 轮、
n=100/档，两侧都用**全新对等的** `PERF_WORK_DIR`，
并以 `ask` 模式作天然对照组（修复没动同步路径）。

**先证明两个端口跑的确实不是同一份代码**——压测 app 新增 `/_metrics`
端点，发一个全新问题看检索缓存计数器：

| 端口 | 计数器 delta | 含义 |
| --- | --- | --- |
| 8011 基线 | `miss +1` 且 `hit +1` | 一次请求检索两次（未修复） |
| 8010 修复 | `miss +1`、`hit +0` | 只检索一次 |

结果：

| mode | 并发 | P50 delta | TTFT P50 delta |
| --- | ---: | ---: | ---: |
| ask（对照） | 1 / 5 / 10 / 20 | +4.4% / −8.8% / −3.9% / −2.5% | – |
| stream | 1 / 5 / 10 / 20 | −5.0% / −3.6% / **+2.8%** / −2.5% | **−9.2% / −7.1% / −5.4% / −9.0%** |
| 检索耗时（miss/hit，c=1） | | **−27.8% / −39.2%**（5/5 轮全负） | – |
| 检索耗时（miss/hit，c=10） | | **−24.2% / −42.7%**（5/5 轮全负） | – |

**结论要说得很克制**：

* 能站住的：**检索耗时 −24%~−43%**（4 组全部 5/5 轮为负）、
  **TTFT −5%~−9%**（四档方向一致）。
* **不能写「端到端延迟降低 X%」**：对照组 ask 符号不一致
  （+4.4% ~ −8.8%），实验组 stream 端到端同样符号不一致
  （c=10 甚至是 +2.8%），幅度全部落在本机噪声内
  （基线 5 轮峰谷差 19~27%）。

### 5.5 修正了两个错误结论（都是测量方法问题）

**（一）缓存的端到端收益。** 修正了三次：

| 版本 | 方法 | c=10 端到端收益 |
| --- | --- | --- |
| v1 | n=20 单轮 | −17% |
| v2 | n=100 × 5 轮交错，但两服务状态不对等 | −4.7% |
| v3 | n=100 × 5 轮交错，两服务全新对等目录 | **−7.8%** |

检索那组（−42%~−45%）三轮都稳定；端到端一直在 5%~8% 之间晃。
所以最终只写「个位数百分点」，不写精确值。

**（二）修复的端到端收益。** v2 曾测出「stream P50 随并发单调改善
−4.3% → −14.7%」，看起来非常漂亮，**但它是假的**。

根因：修复版服务用的是跑了几十轮的 `/tmp/rag-perf-work`
（SQLite 已积累几千条会话历史），基线用全新目录——等于让修复版负重跑。
本该中性的 ask 对照组因此出现 +2.5%~+9.5% 的系统性偏差，
全部叠加成了"修复的收益"。

修正后单调性消失，端到端 delta 落到噪声内。
教训：**A/B 的两个对象必须只差那一个变量**；这类偏差长得特别像真信号
（单调、有趋势、看起来合理），只有靠对照组才能识破。
详见 `evaluation/performance/README.md` §5.6。

---

## 6. 判定总结

| 检查项 | 结果 |
| --- | --- |
| Retrieval A/B（100 题，148 个数值单元全量比对） | ✅ PASS（全部逐位相同） |
| RAG retrieval-only（100 题，4 项指标） | ✅ PASS（delta = 0.0000） |
| RAG retrieval-only（修复后复跑） | ✅ PASS（delta = 0.0000） |
| 随机基线一致 | ✅ PASS |
| LLM Judge 全量重跑 | ⛔ NOT RUN（成本 + 模型不可比），历史结果保留 |
| LLM Judge 子集 | ⚠️ 仅作链路验证，不可用于质量结论 |
| 流式重复检索缺陷 | ✅ 已修复（计数探针 + 22 项单测双重验证） |
| `cache_hit` 归因错误 | ✅ 随上一项一并修复 |
| 修复后质量回归 | ✅ 零回归 |
| 修复后性能收益 | ✅ 已实测：检索耗时 −24%~−43%（5/5 轮全负）、TTFT −5%~−9%；端到端在噪声内，不作数值声称 |
| 全部测试 | ✅ 34 + 57 passed；自带 runner 158 + 44 + 22 项，0 失败 |

**结论：Stage 1~4 的工程化改造没有造成 RAG 检索质量回归。**
压测中发现的两个真实缺陷已修复、已复测、已回归，并用计数而非延迟
作为确定性证据钉死，防止复发。
