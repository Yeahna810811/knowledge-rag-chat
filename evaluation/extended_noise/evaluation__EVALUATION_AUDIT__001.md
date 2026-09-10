，
特征为「术语高度重合、事实完全不同」——这是最难区分的负样本类型：

| 文档 | 干扰点 |
|---|---|
| `alt_stack.md` | 另一套向量库 / embedding / 切分选型（Milvus、Chroma、m3e、MarkdownHeaderTextSplitter） |
| `llm_ops.md` | 模型对比、prompt 工程、成本控制（同样谈 Qwen、DeepSeek、温度参数） |
| `deploy_ops.md` | Docker / K8s / 网关 / 可观测性（同样谈健康检查、P95、链路追踪） |
| `advanced_rag.md` | 查询改写、多路召回、重排、评测方法论（同样谈 RRF、k=60） |
| `kb_platform.md` | 多租户、版本管理、权限过滤、审计 |
| `search_engine.md` | 倒排索引、中文分词、BM25 原理 |
| `chatbot_nlu.md` | 意图识别、FAQ 匹配、多轮对话 |