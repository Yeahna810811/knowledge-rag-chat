# Performance Benchmark

- Base URL: `http://127.0.0.1:8010`
- LLM type: **fake**
- Cache state: `miss`
- Concurrency levels: [1, 10]
- Requests per level: 20（warm-up 5 不计入）
- Question pool size: 10
- Session pool size: 50（限流按 session 计）

> 百分位为线性插值法（同 numpy.percentile 默认），样本量见表内 `sample_size`。TTFT 为客户端视角：发起请求到收到第一个 delta 事件。

## Summary

| mode | concurrency | requests | success | error_rate | throughput_rps | mean_ms | p50_ms | p95_ms | p99_ms | ttft_p50 | ttft_p95 | ttft_p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stream | 1 | 20 | 20 | 0.0 | 17.03 | 58.69 | 58.96 | 65.78 | 67.36 | 27.2 | 34.12 | 34.43 |
| stream | 10 | 20 | 20 | 0.0 | 96.58 | 86.36 | 84.17 | 131.02 | 144.6 | 45.54 | 55.16 | 55.52 |

## Notes

- `stream` @ c=1: done_rate=1.0 duplicate_delta=0 cache hit=20 / miss=0 / unknown=0 retrieval mean=6.42ms p50=5.95ms (n=20)
- `stream` @ c=10: done_rate=1.0 duplicate_delta=0 cache hit=20 / miss=0 / unknown=0 retrieval mean=12.2ms p50=12.5ms (n=20)
