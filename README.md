# 基于 LangChain 的智能知识库 RAG 问答平台

一个面向本地/私有文档的检索增强生成（RAG）应用。项目使用 **FastAPI + LangChain + Sentence-Transformers + FAISS + Qwen/DashScope**，支持多格式文档入库、Top-K 语义检索、多轮会话记忆、知识增强问答和 Web 交互。

> 当前项目是 RAG 应用，不包含 Agent 循环或工具规划逻辑；Embedding 在本地运行，LLM 默认通过阿里云百炼 DashScope 的 OpenAI-compatible API 调用 `qwen-plus`。

## 功能

- 支持 `.txt`、`.md`、`.pdf`、`.docx`、`.csv` 文档上传与解析
- 使用 `RecursiveCharacterTextSplitter` 完成可配置 Chunk 切分与重叠
- 使用 `BAAI/bge-small-zh-v1.5` / Sentence-Transformers 生成本地语义向量
- 基于 FAISS 实现向量索引持久化、增量入库与 Top-K 相似度检索
- 将检索上下文、历史对话与当前问题组合后交给 Qwen 生成回答
- 返回检索来源片段，便于查看回答依据
- 基于 `session_id` 维护多会话历史，默认最多保留 10 轮
- 提供文档上传、问答、状态查询、知识库清空、会话历史清理 API
- 提供 Web 页面和 CLI 两种使用方式

## 系统流程

```text
文档上传
   ↓
多格式 Document Loader
   ↓
RecursiveCharacterTextSplitter
   ↓
Sentence-Transformers Embedding
   ↓
FAISS 持久化向量索引

用户问题 + session_id
   ↓
FAISS Top-K 语义检索
   ↓
检索片段 + 历史对话 + 当前问题
   ↓
Qwen / DashScope
   ↓
回答 + Sources
```

## 技术栈

| 模块 | 技术 |
| --- | --- |
| Web API | FastAPI, Uvicorn |
| RAG 编排 | LangChain |
| Embedding | HuggingFaceEmbeddings / Sentence-Transformers |
| 向量检索 | FAISS |
| 文本切分 | RecursiveCharacterTextSplitter |
| LLM | Qwen (`qwen-plus`) via DashScope |
| 文档解析 | PyPDF, Docx2txt, Unstructured, CSVLoader |
| 前端 | HTML / CSS / JavaScript |

## 项目结构

```text
knowledge-rag-chat/
├── run.py                         # Web 服务启动入口
├── evaluate.py                    # 基础检索相关性 / 响应耗时检查脚本
├── evaluation/                    # RAG Benchmark 评测模块
│   ├── evaluate_rag.py
│   ├── eval_dataset.jsonl
│   ├── corpus/
│   └── results/
├── README.md
├── .gitignore
└── frontend/
    ├── __init__.py
    └── local_rag/
        ├── app.py                 # FastAPI App
        ├── cli.py                 # CLI 入口
        ├── index.html             # Web 聊天页面
        ├── requirements.txt
        ├── .env.example
        ├── api/
        │   └── routes.py          # REST API
        ├── config/
        │   └── settings.py        # 环境变量 / 配置
        ├── core/
        │   ├── document_loader.py
        │   ├── document_processor.py
        │   ├── embedding_service.py
        │   ├── text_splitter.py
        │   ├── vector_store.py
        │   └── rag_chain.py
        ├── services/
        │   └── knowledge_service.py
        ├── utils/
        │   └── file_utils.py
        └── data/
            ├── uploads/           # 运行时生成，不提交 Git
            └── faiss_index/       # 运行时生成，不提交 Git
```

## 快速开始

### 1. 创建环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r frontend/local_rag/requirements.txt
```

Windows PowerShell 激活方式：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. 配置 DashScope

```bash
cp frontend/local_rag/.env.example frontend/local_rag/.env
```

编辑 `frontend/local_rag/.env`：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
CHAT_MODEL=qwen-plus
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> `.env` 已被 `.gitignore` 排除，请勿提交真实 API Key。

### 3. 启动

在项目根目录执行：

```bash
python run.py
```

浏览器访问：

```text
http://127.0.0.1:8000
```

Swagger API 文档：

```text
http://127.0.0.1:8000/docs
```

首次运行 Embedding 时会从 Hugging Face 下载模型。

## API

| Method | Endpoint | 说明 |
| --- | --- | --- |
| POST | `/api/upload` | 上传并入库文档 |
| POST | `/api/ask` | RAG / 普通聊天问答 |
| GET | `/api/status` | 查看知识库和模型状态 |
| DELETE | `/api/reset` | 清空 FAISS 知识库 |
| POST | `/api/clear_history` | 清空指定会话历史 |

问答请求示例：

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"这份文档主要讲了什么？","session_id":"demo"}'
```

## CLI

在项目根目录执行：

```bash
python -m frontend.local_rag.cli upload frontend/local_rag/data/sample.txt
python -m frontend.local_rag.cli ask "RAG 的工作流程是什么？"
python -m frontend.local_rag.cli status
python -m frontend.local_rag.cli reset
```

## 配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地 Embedding 模型 |
| `DASHSCOPE_API_KEY` | 空 | DashScope API Key，运行问答必须配置 |
| `CHAT_MODEL` | `qwen-plus` | 默认聊天模型 |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible endpoint |
| `CHUNK_SIZE` | `500` | 文本切片大小 |
| `CHUNK_OVERLAP` | `50` | Chunk 重叠长度 |
| `RETRIEVAL_TOP_K` | `4` | 每次检索返回片段数 |

## 项目亮点

- 完整实现“文档摄取 → Chunk → Embedding → FAISS → Top-K Retrieval → LLM”的 RAG 链路
- Embedding 本地运行，知识库索引使用 FAISS 持久化保存
- 同一接口兼容知识库问答与无知识库时的普通聊天
- 基于 `session_id` 隔离多轮对话记忆
- API、Web UI、CLI 三种交互方式覆盖端到端应用流程
- 配置、业务服务、检索与 API 路由分层，便于继续扩展 Reranker、Hybrid Search 和评测模块

## Benchmark 评测与优化

项目提供 `evaluation/evaluate_rag.py`，用于对 RAG 链路进行离线 Benchmark 评测。

当前 Benchmark v2 包含：

- 50 条人工设计 QA
- 40 条知识库可回答问题
- 10 条知识库不可回答问题
- 4 份测试文档
- 5 个文本 Chunk
- 检索指标：Hit@K、MRR
- 生成指标：Answer Correctness、Faithfulness、Relevance、Safe Refusal
- 性能指标：Retrieval Latency、End-to-End Latency

当前评测配置：

```text
Embedding Model: BAAI/bge-small-zh-v1.5
CHUNK_SIZE: 500
CHUNK_OVERLAP: 50
RETRIEVAL_TOP_K: 4
LLM: qwen-plus
Vector Store: FAISS
```

### 最终检索效果

| Metric | Result |
| --- | ---: |
| Hit@1 | 85.0% |
| Hit@3 | 97.5% |
| Hit@5 | 100.0% |
| MRR | 91.46% |
| Retrieval Avg | 85.46 ms |
| Retrieval P95 | 126.75 ms |

检索侧结果表明，大多数问题的正确证据能够进入 Top-3 检索结果，全部可回答问题的目标证据能够进入 Top-5。

### Grounded Prompt 优化

初始 Prompt 对模型约束较弱。当知识库未明确提供答案时，模型可能继续基于自身知识进行推测，例如猜测并发人数、GPU 型号、云平台、SLA 或服务器成本。

因此在保持以下条件不变的情况下：

- Embedding Model 不变
- Chunk Size / Overlap 不变
- RETRIEVAL_TOP_K 不变
- Benchmark 数据集不变
- Qwen 模型不变

仅优化 System Prompt，引入更严格的 Grounding 约束：

```text
知识库存在明确证据
    ↓
仅依据参考资料回答

知识库没有明确证据
    ↓
回答：
“当前知识库资料未提供该信息。”
    ↓
禁止继续推测或补充
```

Prompt A/B 评测结果：

| Metric | Before | After |
| --- | ---: | ---: |
| Answer Correctness | 86.0% | 99.0% |
| Faithfulness | 83.6% | 100.0% |
| Relevance | 84.0% | 99.6% |
| Safe Refusal | 86.0% | 100.0% |
| Basic Correctness | 77.5% | 87.5% |
| Refusal Accuracy | 20.0% | 100.0% |

Prompt 优化后，10 条知识库不可回答问题均能够稳定拒绝推测，同时保留原有检索效果。

### 响应性能

Prompt 优化后的单次 Benchmark 中：

| Metric | Result |
| --- | ---: |
| End-to-End Avg | 1233.71 ms |
| End-to-End P50 | 883.89 ms |
| End-to-End P95 | 2603.17 ms |

严格 Prompt 减少了模型无依据扩展和冗余生成，在本次测试中同时降低了端到端响应耗时。

### 评测结果文件

```text
evaluation/
├── evaluate_rag.py
├── eval_dataset.jsonl
├── corpus/
└── results/
    ├── evaluation_details_v2_before_prompt.csv
    ├── evaluation_summary_v2_before_prompt.json
    ├── evaluation_details_v2_after_prompt.csv
    └── evaluation_summary_v2_after_prompt.json
```

> 当前 Benchmark 是项目自建的小规模离线评测集，主要用于验证 RAG 检索链路、Prompt Grounding 和参数优化效果，不代表通用生产级 Benchmark 性能。

## 后续计划

- Hybrid Search：BM25 + Vector Search
- Cross-Encoder / Reranker 二阶段重排
- Query Rewrite / Multi-Query Retrieval
- RAGAS / 更大规模多领域 Benchmark 评测
- SSE / WebSocket 流式输出
- Docker 部署与自动化测试
- 多知识库与权限隔离

## 安全说明

- 不要提交 `.env`、API Key、上传文档或本地 FAISS 索引。
- 项目只将自包含的 `index.html` 作为页面返回，不再把整个应用目录挂载为静态目录，避免配置文件被静态访问。
