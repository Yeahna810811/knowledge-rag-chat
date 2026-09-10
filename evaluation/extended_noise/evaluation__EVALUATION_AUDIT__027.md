下；首次部署还要额外下载约 95MB 模型文件。

---

## 七、⚠️ 必须声明的实验局限

- **第三～六节的真实数据用的是 `--dense localbge`**：加载真实 bge-small-zh 权重，
  但前向由 `evaluation/local_bge_embedder.py` 的 numpy 实现完成（4 层 / 512 维 / CLS 池化），
  因为本环境的 transformers 导入被安全代理拦截。已用历史报告交叉验证（见上）。
- **切分用的是复刻实现**：`langchain_text_splitters` 同样依赖 transformers，无法导入。
  复刻版对原始语料切出 5 个 chunk，与历史记录的 `chunks=5` 一致。
- **DashScope embedding API 不可用**：免费额度已耗尽
  （`AllocationQuota.FreeTierOnly`），所以 `--dense dashscope` 跑不通。
- **第一节/第三节的 TF-IDF 代理数字已被第三、四节取代**，不要再引用。