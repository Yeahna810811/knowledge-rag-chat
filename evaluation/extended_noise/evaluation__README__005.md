使用类题目证据
- `deployment_guide.md` — 部署类题目证据

## 运行

```bash
cd ~/Desktop/knowledge-rag-chat
python evaluation/evaluate_rag.py --judge \
  --corpus evaluation/corpus/rag_architecture.md \
  evaluation/corpus/api_guide.md \
  evaluation/corpus/user_manual.md \
  evaluation/corpus/deployment_guide.md
```

命令行参数（`parse_args()`）：