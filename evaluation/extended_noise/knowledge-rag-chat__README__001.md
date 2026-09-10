`RecursiveCharacterTextSplitter` 可配置 Chunk 切分与重叠
- 本地 `BAAI/bge-small-zh-v1.5` Embedding + FAISS 持久化 / 增量入库 / Top-K 检索
- 双模式问答：`rag`（客服 Grounded Prompt + 溯源）与 `chat`（普通 AI 对话）可切换
- 知识库无证据时提示联系人工客服；返回 `sources` 与 `agent_trace`
- 基于 `session_id` 的多会话历史（默认最多 10 轮）与聊天记录查询
- Vue 3 + TypeScript Web：实时对话、Markdown 渲染、模式切换、上传与清空
- LangSmith 可选追踪 Agent 链路；Ragas 可选评测 Faithfulness / Answer Relevancy
- Docker 容器化；GitHub Actions CI；`/api/webhook` 支持自动化工作流集成

## 系统流程