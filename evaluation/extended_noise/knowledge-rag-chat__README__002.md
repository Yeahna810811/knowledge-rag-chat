ub Actions CI；`/api/webhook` 支持自动化工作流集成

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