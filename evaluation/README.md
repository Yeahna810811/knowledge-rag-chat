# RAG Benchmark v2

这个版本修复了上一版测试集的三个问题：

1. 默认嵌入模型已更新为 `BAAI/bge-small-zh-v1.5`
2. 40 个可回答问题的标准证据尽量只放在一个目标文档中，减少 Ground Truth 歧义
3. 10 个不可回答问题对应的信息完全不写进 corpus，避免“文档明确说没有该信息”导致题目实际上变成可回答

## 推荐最终配置

```env
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=4
```

## 使用方法

先备份旧测试集：

```bash
cd ~/Desktop/knowledge-rag-chat
cp evaluation/eval_dataset.jsonl evaluation/eval_dataset_before_v2.jsonl
cp -R evaluation/corpus evaluation/corpus_before_v2
```

然后把本包中的四个 `.md` 文件复制到：

```text
evaluation/corpus/
```

并把：

```text
evaluation/eval_dataset_v2.jsonl
```

复制为：

```text
evaluation/eval_dataset.jsonl
```

最后运行：

```bash
python evaluation/evaluate_rag.py \
  --judge \
  --corpus \
  evaluation/corpus/rag_architecture.md \
  evaluation/corpus/api_guide.md \
  evaluation/corpus/user_manual.md \
  evaluation/corpus/deployment_guide.md
```

## 注意

这一版先不要改 Prompt。先跑一次作为“修正后的 Benchmark 基线”。
基线确认后，再修改 `frontend/local_rag/core/rag_chain.py` 的 RAG Prompt，重新跑同一套 v2 测试，才能公平比较 Prompt 优化前后差异。
