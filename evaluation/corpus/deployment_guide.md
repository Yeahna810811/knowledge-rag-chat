# 部署与配置说明（Benchmark v2）

默认 HOST 配置为 0.0.0.0。

默认 PORT 配置为 8000。

嵌入模型第一次使用时会从 Hugging Face 下载，并在后续运行中优先使用本地缓存。

当前聊天模型配置为 CHAT_MODEL=qwen-plus。

DashScope 的 OpenAI-compatible API 地址为 https://dashscope.aliyuncs.com/compatible-mode/v1。

真实 DASHSCOPE_API_KEY 应写入 frontend/local_rag/.env。

当前 RETRIEVAL_TOP_K 配置值为 4。

FAISS 持久化索引目录由 FAISS_INDEX_DIR 配置项指定。

真实 .env 文件不应提交到 GitHub；公开仓库只应保留 .env.example 占位模板。

项目评测脚本 evaluation/evaluate_rag.py 可以统计 Hit@K、MRR、回答正确性、Faithfulness、Relevance、Safe Refusal、检索延迟和端到端延迟等指标。
