# 基于 LangChain 的智能知识库 RAG 问答平台（多 Agent）

面向本地/私有文档的检索增强生成（RAG）应用。技术栈：**FastAPI + Vue 3/TypeScript + LangChain 多 Agent + Sentence-Transformers + FAISS + Qwen/DashScope + LangSmith + Ragas**。

支持文档入库、Top-K 语义检索、多轮会话记忆、**RAG 知识库 / 普通 AI 对话双模式**、答案溯源，以及 Docker / CI 私有化部署工作流。

## 功能

- 支持 `.txt`、`.md`、`.pdf`、`.docx`、`.csv` 文档上传与解析
- 多 Agent 协同：`document_parse_agent` → `retrieval_agent` → `generation_agent`
- `RecursiveCharacterTextSplitter` 可配置 Chunk 切分与重叠