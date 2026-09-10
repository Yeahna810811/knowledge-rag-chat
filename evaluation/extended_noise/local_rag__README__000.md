# Local RAG Module

该目录实现项目的核心 RAG Web/API 模块，技术栈为 **FastAPI + LangChain + FAISS + Sentence-Transformers + Qwen/DashScope**。

## 数据链路

```text
Upload -> Document Loader -> Text Splitter -> Local Embedding -> FAISS
Question + History -> Top-K Retrieval -> Prompt -> Qwen -> Answer + Sources
```

## 支持能力