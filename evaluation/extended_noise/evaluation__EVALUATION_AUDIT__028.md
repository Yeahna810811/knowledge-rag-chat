` 跑不通。
- **第一节/第三节的 TF-IDF 代理数字已被第三、四节取代**，不要再引用。
  代理稠密路是字面方法，无法反映真实语义向量的行为——这也是为什么
  代理实验得出「混合检索有效」、真实实验却得出相反结论的原因。

### 在干净环境（能正常导入 transformers）中复现

```bash
pip install -r frontend/local_rag/requirements.txt
python evaluation/evaluate_retrieval_ab.py --dense faiss --metric bigram \
    --corpus evaluation/corpus/*.md evaluation/extended_noise/*.md --sweep-weights
```

---

## 八、复现命令

```bash
# 1) 校验难负样本未污染标注（必须为 0 污染）
python evaluation/validate_distractors.py