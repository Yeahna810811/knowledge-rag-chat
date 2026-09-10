I）
- 会话记忆：基于 `session_id` 的内存历史，最多保留 10 轮

## 目录

```text
local_rag/
├── app.py
├── cli.py
├── index.html
├── requirements.txt
├── .env.example
├── api/routes.py
├── config/settings.py
├── core/
│   ├── document_loader.py
│   ├── document_processor.py
│   ├── embedding_service.py
│   ├── text_splitter.py
│   ├── vector_store.py
│   └── rag_chain.py
├── services/knowledge_service.py
├── utils/file_utils.py
└── data/
```

## 配置

复制环境变量模板：