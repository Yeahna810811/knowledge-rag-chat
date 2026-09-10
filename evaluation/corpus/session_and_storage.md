# 会话与存储说明（Benchmark v3）

会话历史默认最多保留 10 轮，由 MAX_HISTORY_TURNS 控制。

前端状态用 Vue 3 的响应式 API 管理，会话上下文由后端按 session_id 维护。

上传的文件会以 UUID 十六进制前缀重命名后落盘，避免同名文件互相覆盖。

上传文件存放在 data/uploads 目录下。

FAISS 索引持久化到 data/faiss_index 目录。

知识库支持增量入库，新上传的文档会追加到已有索引而不需要全量重建。

清空对话记忆只删除会话历史，不影响 data/faiss_index 下的索引文件。

执行清空知识操作会同时清理知识库状态和相关会话记忆。
