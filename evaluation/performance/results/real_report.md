# Performance Benchmark

- Base URL: `http://127.0.0.1:8000`
- LLM type: **real**
- Cache state: `mixed`
- Concurrency levels: [1, 5]
- Requests per level: 10（warm-up 1 不计入）
- Question pool size: 10
- Session pool size: 50（限流按 session 计）

> 百分位为线性插值法（同 numpy.percentile 默认），样本量见表内 `sample_size`。TTFT 为客户端视角：发起请求到收到第一个 delta 事件。

## Summary

| mode | concurrency | requests | success | error_rate | throughput_rps | mean_ms | p50_ms | p95_ms | p99_ms | ttft_p50 | ttft_p95 | ttft_p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ask | 1 | 10 | 10 | 0.0 | 0.1 | 10098.99 | 7672.74 | 23687.86 | 29672.48 | - | - | - |
| ask | 5 | 10 | 10 | 0.0 | 0.41 | 8781.29 | 7104.86 | 17415.61 | 18834.38 | - | - | - |
| stream | 1 | 10 | 10 | 0.0 | 0.15 | 6497.91 | 5120.04 | 12819.59 | 14109.59 | 4247.94 | 8204.22 | 9324.07 |
| stream | 5 | 10 | 10 | 0.0 | 0.54 | 7575.39 | 5580.33 | 15055.61 | 17017.0 | 5031.65 | 10189.36 | 12052.87 |

## Notes

- `ask` @ c=1: done_rate=1.0 duplicate_delta=0 cache hit=0 / miss=0 / unknown=10 retrieval mean=3.67ms p50=3.85ms (n=10)
- `ask` @ c=5: done_rate=1.0 duplicate_delta=0 cache hit=0 / miss=0 / unknown=10 retrieval mean=4.52ms p50=3.7ms (n=10)
- `stream` @ c=1: done_rate=1.0 duplicate_delta=1 cache hit=10 / miss=0 / unknown=0 retrieval mean=6.09ms p50=5.7ms (n=10)
- `stream` @ c=5: done_rate=1.0 duplicate_delta=2 cache hit=10 / miss=0 / unknown=0 retrieval mean=8.49ms p50=8.45ms (n=10)
