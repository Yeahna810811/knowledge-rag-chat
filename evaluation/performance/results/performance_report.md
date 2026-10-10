# Performance Benchmark

- Base URL: `http://127.0.0.1:8010`
- LLM type: **fake**
- Cache state: `hit`
- Concurrency levels: [1, 10]
- Requests per level: 100（warm-up 5 不计入）
- Question pool size: 10
- Session pool size: 50（限流按 session 计）

> 百分位为线性插值法（同 numpy.percentile 默认），样本量见表内 `sample_size`。TTFT 为客户端视角：发起请求到收到第一个 delta 事件。

## Summary

| mode | concurrency | requests | success | error_rate | throughput_rps | mean_ms | p50_ms | p95_ms | p99_ms | ttft_p50 | ttft_p95 | ttft_p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stream | 1 | 100 | 100 | 0.0 | 13.53 | 73.87 | 74.05 | 79.17 | 85.55 | 30.87 | 34.61 | 36.03 |
| stream | 10 | 100 | 100 | 0.0 | 151.74 | 63.87 | 56.12 | 113.1 | 136.67 | 25.17 | 52.01 | 55.08 |

## Notes

- `stream` @ c=1: done_rate=1.0 duplicate_delta=0 cache hit=100 / miss=0 / unknown=0 retrieval mean=3.74ms p50=3.4ms (n=100)
- `stream` @ c=10: done_rate=1.0 duplicate_delta=0 cache hit=100 / miss=0 / unknown=0 retrieval mean=3.26ms p50=2.75ms (n=100)
