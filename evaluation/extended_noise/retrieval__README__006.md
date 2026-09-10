True` 已经是妥协，这里没必要再开口子。
BM25 的倒排可以从原文无损重建，存原文就够了。

**3. 老部署自动迁移。**
已有 FAISS 索引但没有稀疏索引时，`_migrate_from_dense()` 会从
FAISS docstore 导出原文补建。读的是私有 API，整段包 try/except——
读不到最多不迁移，不能让服务起不来。

**4. `/status` 刻意不触发 runtime 初始化。**
它是健康检查入口，被探针频繁调用。若触发 embedding 模型加载，
冷启动会把健康检查拖到几十秒，容器里会被 liveness probe 判死。

**5. 三级降级。**
稀疏路异常 → 回落稠密；稠密路异常 → 记 warning 继续用稀疏；
两路都不可用 → 返回空列表（与「知识库为空」行为一致，generation_agent 已处理）。
降级状态通过 `KnowledgeRetriever.degraded` 暴露，可接监控。

---

## 实测收益