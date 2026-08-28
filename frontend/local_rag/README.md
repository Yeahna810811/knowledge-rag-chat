# Local RAG Module

该目录实现项目的核心 RAG Web/API 模块，技术栈为 **FastAPI + LangChain + FAISS + Sentence-Transformers + Qwen/DashScope**。

## 数据链路

```text
Upload -> Document Loader -> Text Splitter -> Local Embedding -> FAISS
Question + History -> Top-K Retrieval -> Prompt -> Qwen -> Answer + Sources
```

## 支持能力

- 文档：TXT / Markdown / PDF / DOCX / CSV
- Embedding：`all-MiniLM-L6-v2`（默认，本地运行）
- Vector Store：FAISS（本地持久化）
- LLM：`qwen-plus`（默认，通过 DashScope OpenAI-compatible API）
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

```bash
cp frontend/local_rag/.env.example frontend/local_rag/.env
```

至少填写：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

默认配置：

```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHAT_MODEL=qwen-plus
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=4
```

## 从项目根目录运行

```bash
pip install -r frontend/local_rag/requirements.txt
python run.py
```

Web：`http://127.0.0.1:8000`  
Docs：`http://127.0.0.1:8000/docs`

## CLI

```bash
python -m frontend.local_rag.cli upload frontend/local_rag/data/sample.txt
python -m frontend.local_rag.cli ask "LangChain 在项目中的作用是什么？"
python -m frontend.local_rag.cli status
python -m frontend.local_rag.cli reset
```
