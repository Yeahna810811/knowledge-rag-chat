0 污染）
python evaluation/validate_distractors.py

# 2) 生成扩展噪声语料（自动剔除污染块）
python evaluation/build_extended_corpus.py

# 3) 跑 A/B（离线，零依赖）
python evaluation/evaluate_retrieval_ab.py \
    --corpus evaluation/corpus/*.md evaluation/extended_noise/*.md \
    --metric bigram --sweep-weights

# 4) 检索层单测（26 项）
python tests/test_retrieval.py
```

产物：`evaluation/results/ab_retrieval.json`、`ab_retrieval_details.csv`