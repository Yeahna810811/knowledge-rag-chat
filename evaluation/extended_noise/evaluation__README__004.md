source_contains` | 否 | 兜底：无 evidence 时按文件名匹配来源 |

v2 相对 v1 的修复：默认嵌入模型更新为 `BAAI/bge-small-zh-v1.5`；40 个可答题的 evidence 尽量只落在单个目标文档，减少 Ground Truth 歧义；10 个不可答题对应的信息完全不写入语料，避免"文档明确写了没有"导致题目实际可答。

## 语料 `corpus/`

4 个 Markdown 文档，通过 `DocumentProcessor` 切块（`CHUNK_SIZE=500`、`CHUNK_OVERLAP=50`）后写入临时 FAISS：

- `rag_architecture.md` — 架构类题目证据
- `api_guide.md` — API 类题目证据
- `user_manual.md` — 使用类题目证据
- `deployment_guide.md` — 部署类题目证据

## 运行