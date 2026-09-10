bash
docker compose up --build
```

## Benchmark

自建 50 条 QA（40 可答 / 10 拒答）：

```bash
python evaluation/evaluate_rag.py --judge \
  --corpus evaluation/corpus/*.md
```

Ragas（可选）：

```bash
pip install -r evaluation/requirements-eval.txt
python evaluation/evaluate_with_ragas.py --limit 10
```

已沉淀结果见 `evaluation/results/`（Hit@3 97.5%、Hit@5 100%、MRR 91.46%；Prompt 优化后 Correctness / Faithfulness / SafeRefusal 显著提升）。

## 项目结构