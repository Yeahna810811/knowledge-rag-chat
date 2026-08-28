# 基于 LangChain 的智能知识库 RAG 问答平台

一个面向本地/私有文档的检索增强生成（RAG）应用。项目使用 **FastAPI + LangChain + Sentence-Transformers + FAISS + Qwen/DashScope**，支持多格式文档入库、Top-K 语义检索、多轮会话记忆、知识增强问答和 Web 交互。

> 当前项目是 RAG 应用，不包含 Agent 循环或工具规划逻辑；Embedding 在本地运行，LLM 默认通过阿里云百炼 DashScope 的 OpenAI-compatible API 调用 `qwen-plus`。

## 功能

- 支持 `.txt`、`.md`、`.pdf`、`.docx`、`.csv` 文档上传与解析
- 使用 `RecursiveCharacterTextSplitter` 完成可配置 Chunk 切分与重叠
- 使用 `all-MiniLM-L6-v2` / Sentence-Transformers 生成本地语义向量
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
knowledge-rag-chat-main/
├── run.py                         # Web 服务启动入口
├── evaluate.py                    # 基础检索相关性 / 响应耗时检查脚本
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
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 本地 Embedding 模型 |
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

## 基础评测

根目录提供 `evaluate.py`，用于检查：

- 多格式文档处理能力
- 检索结果与查询的向量相似性覆盖情况
- LLM 问答响应耗时

当前脚本没有人工标注的 ground truth，因此其中基于相似度阈值计算的数值应视为基础检索相关性检查，**不等同于严格定义的 Recall@K**。

## 后续计划

- Hybrid Search：BM25 + Vector Search
- Cross-Encoder / Reranker 二阶段重排
- Query Rewrite / Multi-Query Retrieval
- RAGAS 或人工 QA Ground Truth 评测
- SSE / WebSocket 流式输出
- Docker 部署与自动化测试
- 多知识库与权限隔离

## 安全说明

- 不要提交 `.env`、API Key、上传文档或本地 FAISS 索引。
- 项目只将自包含的 `index.html` 作为页面返回，不再把整个应用目录挂载为静态目录，避免配置文件被静态访问。
