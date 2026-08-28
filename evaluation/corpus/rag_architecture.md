# RAG 系统架构说明（Benchmark v2）

本项目是一个基于检索增强生成（RAG）的私有知识库问答系统。

## 核心检索链路

系统使用 Sentence-Transformers 生成文本向量，当前默认嵌入模型为 BAAI/bge-small-zh-v1.5。

向量索引使用 FAISS，并将索引持久化到本地磁盘。

文档切分使用 LangChain 的 RecursiveCharacterTextSplitter。

当前文本切分参数为 CHUNK_SIZE=500，CHUNK_OVERLAP=50。

用户提交知识问答请求后，系统先把问题编码为查询向量，再从 FAISS 中检索相关文本块。

当前 RETRIEVAL_TOP_K=4，因此每次知识检索默认最多返回 4 个候选文本块。

检索出的上下文会和用户问题一起组成 RAG Prompt，并交给 Qwen 生成最终答案。

项目通过 session_id 区分不同对话会话。

当前检索方式为单路稠密向量检索，没有启用 BM25 混合检索，也没有接入独立的 Cross-Encoder Reranker。
