# Performance Benchmark

- Base URL: `http://127.0.0.1:8010`
- LLM type: **fake**
- Cache state: `hit`
- Concurrency levels: [1, 10]
- Requests per level: 20（warm-up 5 不计入）
- Question pool size: 10
- Session pool size: 50（限流按 session 计）

> 百分位为线性插值法（同 numpy.percentile 默认），样本量见表内 `sample_size`。TTFT 为客户端视角：发起请求到收到第一个 delta 事件。

## Summary

| mode | concurrency | requests | success | error_rate | throughput_rps | mean_ms | p50_ms | p95_ms | p99_ms | ttft_p50 | ttft_p95 | ttft_p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stream | 1 | 20 | 20 | 0.0 | 17.48 | 57.19 | 57.37 | 59.87 | 60.43 | 26.23 | 29.15 | 30.07 |
| stream | 10 | 20 | 20 | 0.0 | 122.33 | 71.89 | 66.69 | 98.19 | 110.08 | 37.9 | 42.99 | 43.1 |

## Notes

- `stream` @ c=1: done_rate=1.0 duplicate_delta=0 cache hit=20 / miss=0 / unknown=0 retrieval mean=5.37ms p50=4.8ms (n=20)
- `stream` @ c=10: done_rate=1.0 duplicate_delta=0 cache hit=20 / miss=0 / unknown=0 retrieval mean=7.07ms p50=6.95ms (n=20)
