Prompt -> Qwen -> Answer + Sources
```

## 支持能力

- 文档：TXT / Markdown / PDF / DOCX / CSV
- Embedding：`all-MiniLM-L6-v2`（默认，本地运行）
- Vector Store：FAISS（本地持久化）
- LLM：`qwen-plus`（默认，通过 DashScope OpenAI-compatible API）
- 会话记忆：基于 `session_id` 的内存历史，最多保留 10 轮

## 目录