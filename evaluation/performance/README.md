# Stage 5 · Performance Benchmark

这一套工具用来回答一个具体问题：

> 在 1 / 5 / 10 / 20 并发下，**我自己的应用层**表现如何？

它不测上游大模型本身。

---

## 1. 两类 benchmark 必须分开看

| | Application-level | Real-LLM |
| --- | --- | --- |
| LLM | `FakeLLM`（进程内确定性 token 流，零网络、零额度） | Qwen / DashScope 真实调用 |
| 测的是 | FastAPI 路由、限流、Redis 检索缓存、BM25 检索、SQLite 持久化、SSE 分帧、并发调度 | 用户真实体感：TTFT、整轮生成耗时、上游抖动 |
| 结果文件 | `performance_summary.csv` / `cache_*_summary.csv` | `real_summary.csv` |
| 能不能写进简历的 QPS | **不能**当作真实线上 QPS | 才是真实 QPS |

**FakeLLM 的 QPS 不是真实 LLM 的 QPS。** 两者相差两个数量级（见下），
混着说就是造假。

---

## 2. 目录

```text
evaluation/performance/
├── fake_app.py              # FakeLLM 压测专用 app（真实栈 + 假 LLM）
├── run_load_test.py         # 压测执行器（CLI 可配置）
├── make_charts.py           # 从 CSV 出图
├── README.md
└── results/
    ├── performance_summary.csv        # fake: ask + stream × 1/5/10/20
    ├── performance_requests.csv       # fake: 每次请求原始测量
    ├── performance_report.md          # 脚本自动生成的报告
    ├── cache_miss_summary.csv / cache_hit_summary.csv
    ├── real_summary.csv / real_requests.csv      # 真实 LLM
    └── *.png
```

---

## 3. 怎么跑

### 3.1 Application-level（默认，不需要 API key）

```bash
# 终端 1：起 FakeLLM 应用
python -m uvicorn evaluation.performance.fake_app:app --host 127.0.0.1 --port 8010

# 终端 2：压测
python evaluation/performance/run_load_test.py \
  --base-url http://127.0.0.1:8010 \
  --llm-type fake \
  --mode ask stream \
  --concurrency 1 5 10 20 \
  --requests-per-level 20 \
  --warmup 5
```

cache hit / miss 对比：

```bash
python evaluation/performance/run_load_test.py \
  --base-url http://127.0.0.1:8010 --llm-type fake \
  --mode stream --concurrency 1 10 --requests-per-level 20 \
  --cache-state miss \
  --summary-filename cache_miss_summary.csv \
  --requests-filename cache_miss_requests.csv

python evaluation/performance/run_load_test.py \
  --base-url http://127.0.0.1:8010 --llm-type fake \
  --mode stream --concurrency 1 10 --requests-per-level 20 \
  --cache-state hit \
  --summary-filename cache_hit_summary.csv \
  --requests-filename cache_hit_requests.csv
```

### 3.2 Real-LLM（需要有效 `DASHSCOPE_API_KEY`）

```bash
python run.py     # 或 uvicorn frontend.local_rag.app:app --port 8000

python evaluation/performance/run_load_test.py \
  --base-url http://127.0.0.1:8000 \
  --llm-type real \
  --mode ask stream \
  --concurrency 1 5 \
  --requests-per-level 10 \
  --summary-filename real_summary.csv \
  --requests-filename real_requests.csv
```

### 3.3 出图

```bash
python evaluation/performance/make_charts.py
```

### 3.4 交错 A/B + 汇总（改代码前后对比必用）

单轮数字不可信（见 §5.5）。要比较两个版本，起两个服务**交替**打：

```bash
# 8010 = 修复版，8011 = 基线版（同一份代码回退改动）
for i in 1 2 3 4 5; do
  python evaluation/performance/run_load_test.py --base-url http://127.0.0.1:8010 \
    --llm-type fake --mode ask stream --concurrency 1 5 10 20 \
    --requests-per-level 100 --warmup 10 --cache-state mixed
  cp evaluation/performance/results/performance_summary.csv \
     evaluation/performance/results/ab_fix_vs_baseline/fixed_rep$i.csv

  python evaluation/performance/run_load_test.py --base-url http://127.0.0.1:8011 \
    --llm-type fake --mode ask stream --concurrency 1 5 10 20 \
    --requests-per-level 100 --warmup 10 --cache-state mixed
  cp evaluation/performance/results/performance_summary.csv \
     evaluation/performance/results/ab_fix_vs_baseline/baseline_rep$i.csv
done

python evaluation/performance/aggregate_ab.py   # -> results/ab_summary.csv
```

配套的确定性验证（不靠延迟、靠计数）：

```bash
python tests/test_orchestrator_retrieval_once.py      # 22 项，CI 里跑
PERF_WORK_DIR=/tmp/rag-perf-work \
  python evaluation/performance/probe_retrieval_count.py
```

### 3.5 全部 CLI 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--base-url` | `http://127.0.0.1:8000` | 被测服务地址 |
| `--mode` | `ask stream` | 测哪些接口 |
| `--llm-type` | `fake` | **结果标签**，脚本不会伪造，需自己如实指定 |
| `--concurrency` | `1 5 10 20` | 并发档位 |
| `--requests-per-level` | `20` | 每档正式样本数（warm-up 不计） |
| `--warmup` | `5` | 每档预热请求数，不计入统计 |
| `--question-pool-size` | `10` | 问题池大小 |
| `--session-pool-size` | `50` | session 池大小（限流按 session 计） |
| `--cache-state` | `mixed` | `mixed`/`hit` 用固定问题池；`miss` 每请求唯一问题 |
| `--output-dir` | `evaluation/performance/results` | 输出目录 |
| `--summary-filename` 等 | — | 换名可避免覆盖上次结果 |

`fake_app.py` 的环境变量：`PERF_WORK_DIR`、`PERF_REDIS_URL`、`PERF_CORPUS`、
`PERF_FIRST_TOKEN_DELAY`、`PERF_PIECE_DELAY`、`PERF_CHUNKS`、
`PERF_RATE_LIMIT`、`PERF_CACHE_ENABLED`。

---

## 4. 指标怎么算的

* **延迟**：客户端视角 `time.perf_counter()`，从发起请求到响应/流结束，
  包含 FastAPI 路由、限流、Redis、检索、写库、SSE 分帧。
* **TTFT**：客户端视角，发起请求 → 收到**第一个 `delta` 事件**。
  `meta` / `sources` / `trace` / 心跳注释帧都不算。
* **百分位**：线性插值法（与 `numpy.percentile` 默认一致），
  空样本返回空，**不会**用均值顶替。样本量写在同一行的
  `sample_size_latency` / `sample_size_ttft`。
* **throughput**：`正式请求数 / 该档总耗时`，单位是 req/s。
* **error_rate**：非 2xx、流未收到 `done`、或流中出现 `error` 事件都算失败。
* **`cache_hit`**：取自 SSE `done` 事件的 `timing.cache_hit`。
  `/api/ask` 不回传该字段，所以 ask 档位是 `unknown`。

### 曾经存在的口径缺陷（已修复）

`timing.cache_hit` 一度**不可靠**：`AgentOrchestrator.astream()` 里
检索被执行了两次，第一次 miss（写入缓存）、第二次立刻命中刚写进去的那条，
于是 `cache_hit` 只要两次里有 1 次命中就报 `true`，**首次查询也被标成命中**。

根因已修（删掉重复的那次检索调用，`orchestrator.py` 3 行删除）。
修复后每次请求只会 +1 一个计数器，`cache_hit` 恢复正确。
由 `tests/test_orchestrator_retrieval_once.py`（22 项）和
`probe_retrieval_count.py`（进程内计数探针）双重钉死，防止复发。

---

## 5. 本次结果（Stage 5）

测试机：Apple M1 Pro / 8 核 / 16 GB，macOS 26.2 (Darwin 25.2.0)。
Python 3.9.13（项目 `.venv`）。Redis 为真机 `127.0.0.1:6379`，
数据库为 SQLite 临时文件。压测客户端与服务端在同一台机器上。

所有数字都是 **n=100/档、warm-up 10（不计入）、5 轮交错测量的中位数**，
来自 `results/ab_summary.csv`；30 个原始文件在 `results/ab_fix_vs_baseline/`。
为什么不直接用单轮数字见 §5.5——**单轮噪声大到会出现"命中缓存比未命中还慢"
这种违背因果的结果**。

### 5.1 Application-level（FakeLLM，修复后）

| mode | c | rps | mean | P50 | P95 | P99 | TTFT P50 | TTFT P95 | TTFT P99 | err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ask | 1 | 36.45 | 27.34 | 27.23 | 33.30 | 38.37 | – | – | – | 0% |
| ask | 5 | 189.13 | 25.54 | 23.41 | 37.89 | 54.82 | – | – | – | 0% |
| ask | 10 | 235.66 | 37.48 | 29.83 | 88.62 | 163.68 | – | – | – | 0% |
| ask | 20 | 191.78 | 73.69 | 43.24 | 270.04 | 414.06 | – | – | – | 0% |
| stream | 1 | 14.13 | 70.76 | 68.63 | 81.40 | 104.27 | 29.12 | 39.42 | 44.80 | 0% |
| stream | 5 | 79.21 | 62.38 | 58.91 | 79.68 | 97.94 | 26.04 | 37.41 | 39.61 | 0% |
| stream | 10 | 142.00 | 67.51 | 64.82 | 102.30 | 148.80 | 28.40 | 52.48 | 55.80 | 0% |
| stream | 20 | 178.18 | 97.24 | 77.58 | 177.77 | 333.92 | 34.12 | 79.13 | 81.05 | 0% |

（单位 ms。）

注意 c=20 的长尾：ask P99 414ms、stream P99 334ms，
而 P50 只有 43 / 78 ms。**n=20 的旧数据看不到这个尾巴**，
因为它只有 20 个样本，P99 基本等于最大值且极不稳定。
要拿这组数字谈容量，必须盯着 P95/P99 而不是 P50。

其中 FakeLLM 自身注入的模拟耗时为
`first_token_delay 10ms + 6 × piece_delay 5ms ≈ 40ms`，
也就是说 **stream 的 ~70ms 里有约 40ms 是假的**，
真实应用层开销大约在 25–30ms 量级。

### 5.2 Real-LLM（qwen3.8-max-0902，n=10/档，warm-up 1）

| mode | c | rps | mean | P50 | P95 | P99 | TTFT P50 | TTFT P95 | err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ask | 1 | 0.10 | 10098.99 | 7672.74 | 23687.86 | 29672.5 | – | – | 0% |
| ask | 5 | 0.41 | 8781.29 | 7104.86 | 17415.61 | 18834.4 | – | – | 0% |
| stream | 1 | 0.15 | 6497.91 | 5120.04 | 12819.59 | 14109.6 | 4247.94 | 8204.22 | 0% |
| stream | 5 | 0.54 | 7575.39 | 5580.33 | 15055.61 | 17017.0 | 5031.65 | 10189.36 | 0% |

样本量只有 10，**P95/P99 统计意义很弱**（n=10 时 P95 基本就是第二大值），
只能用来看量级和上游抖动范围，不能当 SLA。

> 这组是**修复前**测的，未重测。理由：本次修复省掉的是约 2–3 ms 的本地检索，
> 而真实 LLM 的端到端是 5000 ms 量级——占比 <0.1%，
> 远低于 n=10 的噪声（±20% 以上），重测只会得到噪声。
> 如果将来换成低延迟模型（<500ms），这组数字必须重跑。

### 5.3 Cache hit vs miss（stream，FakeLLM，修复后）

| 并发 | 状态 | 检索 mean | 检索 P50 | E2E mean | E2E P50 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | miss | 6.74 | 6.00 | 73.57 | 72.91 |
| 1 | hit | 3.93 | 3.40 | 70.14 | 69.54 |
| 10 | miss | 6.46 | 5.70 | 65.64 | 62.24 |
| 10 | hit | 3.56 | 3.30 | 60.53 | 55.85 |

结论必须分开说：

* **检索延迟改善明显**：c=1 时 6.74 → 3.93 ms（−41.7%）；
  c=10 时 6.46 → 3.56 ms（−44.9%）。
* **端到端改善很小**：c=1 时 −4.7%，c=10 时 −7.8%。
  因为端到端里生成占大头，**缓存不是万能药**，
  不要拿检索的改善幅度去暗示端到端的改善幅度。

> 修正一处旧结论：最早 n=20 单轮测出「c=10 端到端 −17%」，是噪声；
> 第二轮（服务状态不对等）测出 −4.7%；
> 第三轮（两服务全新对等目录）为 **−7.8%**。
> 检索的 −42%~−45% 是稳定的，端到端的数字一直在 5%~8% 之间晃——
> **只写"个位数百分点"，不要写精确值。**

### 5.4 修复 vs 基线（去掉重复检索的收益）

方法：起两个服务——`8010` 修复版、`8011` 未修复版（同一份代码
`git checkout` 回退 3 行），**交替各跑 1 轮、共 5 轮**，n=100/档，
两侧都用**全新的** `PERF_WORK_DIR`（见 §5.6 的教训）。

**先证明两个端口跑的确实是不同代码**——靠计数，不靠"我应该是先启动的"：

```text
GET /_metrics → 发一个全新问题 → 再 GET /_metrics

8011 基线: cache_miss +1 且 cache_hit +1   ← 一次请求检索两次
8010 修复: cache_miss +1 且 cache_hit +0   ← 只检索一次
```

| mode | c | 基线 P50 | 修复后 P50 | delta | 基线 TTFT P50 | 修复后 TTFT P50 | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ask | 1 | 26.07 | 27.23 | +4.4% | – | – | – |
| ask | 5 | 25.66 | 23.41 | −8.8% | – | – | – |
| ask | 10 | 31.05 | 29.83 | −3.9% | – | – | – |
| ask | 20 | 44.37 | 43.24 | −2.5% | – | – | – |
| stream | 1 | 72.23 | 68.63 | −5.0% | 32.07 | 29.12 | **−9.2%** |
| stream | 5 | 61.12 | 58.91 | −3.6% | 28.02 | 26.04 | **−7.1%** |
| stream | 10 | 63.07 | 64.82 | +2.8% | 30.03 | 28.40 | **−5.4%** |
| stream | 20 | 79.55 | 77.58 | −2.5% | 37.51 | 34.12 | **−9.0%** |

被直接影响的**检索耗时**才是信号最强的指标：

| 并发 | 状态 | 基线 | 修复后 | delta | 逐轮符号 |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | miss | 9.34 | 6.74 | −27.8% | 5/5 为负 |
| 1 | hit | 6.46 | 3.93 | −39.2% | 5/5 为负 |
| 10 | miss | 8.52 | 6.46 | −24.2% | 5/5 为负 |
| 10 | hit | 6.21 | 3.56 | −42.7% | 5/5 为负 |

**怎么读这张表**——`ask` 是天然对照组（修复只动 `astream()`，
同步路径本来就只检索一次）：

* ask 四档 delta 为 +4.4% / −8.8% / −3.9% / −2.5%，**符号不一致**；
  基线自身 5 轮峰谷差就有 19~27%。这就是噪声底线。
* 端到端 P50：stream 四档 −5.0% / −3.6% / **+2.8%** / −2.5%，
  **同样符号不一致、幅度都在噪声内**。
* 只有 **TTFT（−5.4%~−9.2%）和检索耗时（−24%~−43%）** 是稳定的。

### 结论（能写进简历的，和不能写的）

能写：

* 修复确实去掉了一次冗余检索——**检索耗时 −24%~−43%，5/5 轮全负**。
* **TTFT −5%~−9%**，四档方向一致。

不能写：

* **不要写"端到端延迟降低 X%"**。这一项在本机噪声内（±20% 以上），
  实测 stream P50 delta 在 −5.0% ~ +2.8% 之间摆动，
  连符号都不稳定。说"略有改善、但在测量误差内"才是诚实的。

### 5.5 测量噪声（为什么不能用单轮数字）

| 样本量 | ask P50 峰谷差（5 轮） | 检索 mean 峰谷差（5 轮） |
| ---: | --- | --- |
| n=20 | 19% ~ 35% | 20% ~ 97% |
| n=100 | 19% ~ 27% | — |

本机（Apple M1 Pro）基线 load average 常驻 ~4，压测客户端与服务端同机，
所以微基准天然抖。n=20 时甚至出现过「c=10 cache hit 端到端 166ms」
这种离群跑到中位数两倍的情况。

**注意：加大样本量并不能解决。** n=20 → n=100 把峰谷差从 35% 降到 27%，
改善有限——因为主要噪声来源是**机器负载漂移**，不是样本量。
能做的是「交错测量 + 多轮取中位数 + 只信方向一致的指标」。

规则：**任何单轮数字都不得直接写进 README**。
本目录的做法是「n=100 + 5 轮交错 + 取中位数」，
原始 30 个 CSV 全留在 `results/ab_fix_vs_baseline/`，
`aggregate_ab.py` 可一键复算出这张表。

### 5.6 一次真实的方法论错误（留作教训）

第一版 A/B 测出「stream P50 随并发单调改善 −4.3% → −14.7%」，
看起来非常漂亮，**但它是假的**。

根因：修复版服务用的是跑了几十轮的 `/tmp/rag-perf-work`
（SQLite 里已积累几千条会话历史），基线服务用的是全新目录。
等于让修复版**负重**跑——本该中性的 ask 对照组因此出现 +2.5%~+9.5% 的系统性偏差，
全部叠加成了"修复的收益"。

修正：两侧都换成**全新对等的** `PERF_WORK_DIR`，重测后单调性消失，
端到端 delta 落到噪声内。

教训：**A/B 的两个对象必须只差那一个变量**。
工作目录、数据库历史、进程启动时长——任何一个不同都会变成系统性偏差。
而且这类偏差长得特别像真信号（单调、有趋势、看起来合理），
只有靠**对照组**（这里是未被修改的 `ask` 路径）才能识破。

---

## 6. 图表

全部由 `make_charts.py` 从 CSV 生成，无任何手工补点：

```text
latency_percentiles_ask.png        latency_percentiles_stream.png
throughput_vs_concurrency.png      error_rate_vs_concurrency.png
ttft_percentiles_stream.png
cache_hit_vs_miss_retrieval.png    cache_hit_vs_miss_e2e.png
```
