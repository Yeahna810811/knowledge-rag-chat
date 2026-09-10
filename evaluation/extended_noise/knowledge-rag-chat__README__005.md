| CI/CD Webhook（可选 `X-Webhook-Secret`） |

问答示例：

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"这份文档主要讲了什么？","session_id":"demo","mode":"rag"}'
```

Webhook 示例：

```bash
curl -X POST http://127.0.0.1:8000/api/webhook \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: your_secret" \
  -d '{"event":"health.check","payload":{}}'
```

## Docker

```bash
docker compose up --build
```

## Benchmark