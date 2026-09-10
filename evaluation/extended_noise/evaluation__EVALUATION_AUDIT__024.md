riever.py` | 支持 `dense` / `bm25` / `hybrid` 三模式 |
| `config/settings.py` | 新增 `retrieval_mode`、`retrieval_lazy_dense`、`retrieval_dense_fallback`、`hybrid_*_weight`、`bm25_index_dir` |
| `services/knowledge_service.py` | 构造门面注入 orchestrator；**按模式决定是否加载 embedding 模型** |
| `agents/{retrieval,document}_agent.py`、`orchestrator.py` | 类型标注改为协议（运行时零改动）|
| `tests/test_knowledge_retriever.py` | **新增 38 项单测**，全绿 |

### 三个值得说的工程决策