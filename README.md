# 基于 LangChain 的智能知识库 RAG 问答平台（多 Agent）

面向本地/私有文档的检索增强生成（RAG）应用。技术栈：**FastAPI + Vue 3/TypeScript + LangChain 多 Agent + Sentence-Transformers + FAISS + Qwen/DashScope + LangSmith + Ragas**。

支持文档入库、Top-K 语义检索、多轮会话记忆、**RAG 知识库 / 普通 AI 对话双模式**、答案溯源，以及 Docker / CI 私有化部署工作流。

## 功能

- 支持 `.txt`、`.md`、`.pdf`、`.docx`、`.csv` 文档上传与解析
- 多 Agent 协同：`document_parse_agent` → `retrieval_agent` → `generation_agent`
- `RecursiveCharacterTextSplitter` 可配置 Chunk 切分与重叠
- 本地 `BAAI/bge-small-zh-v1.5` Embedding + FAISS 持久化 / 增量入库 / Top-K 检索
- 双模式问答：`rag`（客服 Grounded Prompt + 溯源）与 `chat`（普通 AI 对话）可切换
- 知识库无证据时提示联系人工客服；返回 `sources` 与 `agent_trace`
- 基于 `session_id` 的多会话历史（默认最多 10 轮）与聊天记录查询
- Vue 3 + TypeScript Web：实时对话、Markdown 渲染、模式切换、上传与清空
- LangSmith 可选追踪 Agent 链路；Ragas 可选评测 Faithfulness / Answer Relevancy
- Docker 容器化；GitHub Actions CI；`/api/webhook` 支持自动化工作流集成

## 系统流程

```text
上传文档
  → DocumentParseAgent（解析 / Chunk / Embedding / FAISS）

用户问题 + mode + session_id
  →（rag）RetrievalAgent Top-K
  → GenerationAgent（客服 Prompt 或普通对话 Prompt）
  → 回答 + Sources + agent_trace
```

## 快速开始

### 1. 后端依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r frontend/local_rag/requirements.txt
```

### 2. 配置

```bash
cp frontend/local_rag/.env.example frontend/local_rag/.env
```

至少填写 `DASHSCOPE_API_KEY`。可选开启 LangSmith：

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=knowledge-rag-chat
```

### 3. 前端构建

```bash
cd frontend/web
npm install
npm run build
cd ../..
```

开发时可另开终端：`npm run dev`（Vite 代理 `/api` → `8000`）。

### 4. 启动

```bash
python run.py
```

- Web：http://127.0.0.1:8000
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

自建 50 条 QA（40 可答 / 10 拒答）：

```bash
python evaluation/evaluate_rag.py --judge \
  --corpus evaluation/corpus/*.md
```

Ragas（可选）：

```bash
pip install -r evaluation/requirements-eval.txt
python evaluation/evaluate_with_ragas.py --limit 10
```

检索策略的权威结论见 `evaluation/EVALUATION_AUDIT.md`：100 题 benchmark（80 可答）、45 chunk 语料（含 35 个取自本项目真实文档的难负样本 chunk），BM25 单路 Hit@1 0.838 / MRR 0.893，显著优于稠密路与等权混合（配对 bootstrap，p < 0.05），口语化问法下混合检索亦无显著收益；运行时 45 chunk 入库 3.3ms vs 4920ms、单条查询 0.05ms vs 21.3ms。
早期 5 chunk 语料 + Ragas 口径的旧结果仍留在 `evaluation/results/`，仅作历史对照，**不作为结论引用**（语料过小，随机基线 Hit@3 就有 60%）。

## 项目结构

```text
knowledge-rag-chat/
├── run.py
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── evaluation/
│   ├── evaluate_rag.py
│   ├── evaluate_with_ragas.py
│   └── results/
└── frontend/
    ├── web/                 # Vue 3 + TypeScript
    └── local_rag/
        ├── app.py
        ├── api/routes.py
        ├── core/agents/     # 多 Agent 调度
        └── services/
```
