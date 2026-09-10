# 检索层（稀疏 / 稠密 / 混合）

**当前生产默认：`retrieval_mode = bm25`（BM25 主导 + 稠密兜底）。**

这个结论是实测出来的，不是拍脑袋定的。融合层（hybrid）作为可选项完整保留，
但在本项目的语料和 embedding 模型下**没有展现出显著收益**，所以默认不启用。
依据见 `evaluation/EVALUATION_AUDIT.md`。

---

## 目录结构