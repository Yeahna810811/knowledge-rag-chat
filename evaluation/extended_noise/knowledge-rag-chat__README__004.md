000
- Swagger：http://127.0.0.1:8000/docs

## API

| Method | Endpoint | 说明 |
| --- | --- | --- |
| POST | `/api/upload` | 上传并入库文档 |
| POST | `/api/ask` | 双模式问答（`mode`: `rag` \| `chat`） |
| GET | `/api/status` | 知识库 / Agent / LangSmith 状态 |
| GET | `/api/history` | 查询会话历史 |
| DELETE | `/api/reset` | 清空 FAISS 知识库 |
| POST | `/api/clear_history` | 清空指定会话历史 |
| POST | `/api/webhook` | CI/CD Webhook（可选 `X-Webhook-Secret`） |

问答示例：