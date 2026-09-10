/deployment_guide.md
```

命令行参数（`parse_args()`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | `evaluation/eval_dataset.jsonl` | 测试集路径 |
| `--corpus` | `frontend/local_rag/sample.txt` | 语料文件，可传多个；每个都会切块入临时索引 |
| `--ks` | `1 3 5` | 计算 `hit@K` 的 K 列表 |
| `--judge` | 关闭 | 开启 LLM 打分，需要 `.env` 里配置 `DASHSCOPE_API_KEY`，否则报错 |
| `--judge-model` | 无 | 指定裁判模型，缺省跟随 `.env` 的 `CHAT_MODEL` |
| `--retrieval-only` | 关闭 | 只测检索，不生成回答 |
| `--limit` | 无 | 只跑前 N 题 |
| `--output-dir` | `evaluation/results` | 结果输出目录