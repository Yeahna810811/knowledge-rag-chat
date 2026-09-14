# Local RAG Module

`frontend/local_rag` 是 `knowledge-rag-chat` 的核心 RAG 后端模块。

主要技术：

```text
FastAPI
LangChain
BM25
BGE
FAISS
Qwen / DashScope
```

当前应用默认 Retrieval：

```text
BM25
```

Dense 与 Hybrid 作为可配置模式保留。

---

## 数据链路

文档入库：

```text
Upload
↓
Document Loader
↓
DocumentParseAgent
↓
Recursive Text Splitter
↓
KnowledgeRetriever
├── BM25 / LexicalStore
└── BGE + FAISS
```

问答：

```text
Question
+
Session History
↓
RetrievalAgent
↓
Top-K Sources
↓
GenerationAgent
↓
Qwen
↓
Answer + Sources
```

---

## 支持能力

- TXT / Markdown / PDF / DOCX / CSV 文档解析
- `RecursiveCharacterTextSplitter`
- BM25 稀疏检索
- BM25 JSON 持久化
- `BAAI/bge-small-zh-v1.5` Embedding
- FAISS Dense Retrieval
- Dense / BM25 / Hybrid 三种 Retrieval 模式
- RRF Ranking Fusion
- MMR 去冗余
- PRF Query Rewrite
- Dense Fallback
- RAG / Chat 双模式
- Qwen / DashScope Generation
- Sources 来源溯源
- `session_id` 多会话管理
- 最多保留 10 轮会话历史
- LangSmith 可选追踪

---

## 默认配置

```env
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

CHAT_MODEL=qwen-plus
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=4

RETRIEVAL_MODE=bm25
RETRIEVAL_DENSE_FALLBACK=true
RETRIEVAL_LAZY_DENSE=true

RETRIEVAL_QUERY_REWRITE=prf
QUERY_REWRITE_WEIGHT=0.3
QUERY_REWRITE_TERMS=6
QUERY_REWRITE_FEEDBACK_DOCS=3
```

---

## Retrieval Mode

### bm25

当前默认。

```text
Query
↓
Query Rewrite（可选）
↓
BM25
↓
Top-K
```

如果启用 Dense Fallback 且 Dense Store 已加载，可在稀疏检索无法提供结果时进行降级处理。

### dense

```text
Query
↓
BGE Embedding
↓
FAISS
↓
Top-K
```

用于语义 Retrieval 和 A/B 对照。

### hybrid

```text
BM25
+
Dense
↓
RRF
↓
Top-K
```

当前完整保留用于实验，但正式 Benchmark 中没有超过 BM25。

---

## 目录

```text
local_rag/
├── app.py
├── cli.py
├── requirements.txt
├── .env.example
│
├── api/
│   └── routes.py
│
├── config/
│   └── settings.py
│
├── core/
│   ├── agents/
│   ├── retrieval/
│   ├── document_loader.py
│   ├── document_processor.py
│   ├── embedding_service.py
│   ├── text_splitter.py
│   ├── vector_store.py
│   └── rag_chain.py
│
├── services/
│   └── knowledge_service.py
│
├── utils/
│
└── data/
    ├── uploads/
    ├── bm25_index/
    └── faiss_index/
```

运行时索引不会提交到 Git。

---

## 配置

复制：

```bash
cp frontend/local_rag/.env.example frontend/local_rag/.env
```

配置：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

不要把 `.env` 上传到 GitHub。

---

## 启动

项目根目录：

```bash
pip install -r frontend/local_rag/requirements.txt
python run.py
```

Web：

```text
http://127.0.0.1:8000
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

---

## CLI

上传：

```bash
python -m frontend.local_rag.cli upload frontend/local_rag/data/sample.txt
```

问答：

```bash
python -m frontend.local_rag.cli ask "LangChain 在项目中的作用是什么？"
```

状态：

```bash
python -m frontend.local_rag.cli status
```

清空：

```bash
python -m frontend.local_rag.cli reset
```

---

## Retrieval Benchmark

当前正式 Retrieval Benchmark：

```text
BM25 Hit@1 = 82.5%
BM25 Hit@3 = 95.0%
BM25 MRR   = 0.8869

Dense Hit@1 = 62.5%
Dense MRR   = 0.7060
```

当前项目因此选择 BM25 作为默认 Retrieval。

详细实验：

```text
evaluation/EVALUATION_AUDIT.md
```

---

## Tests

```bash
python tests/test_retrieval.py
python tests/test_knowledge_retriever.py
```

此前测试：

```text
26 passed
38 passed
```