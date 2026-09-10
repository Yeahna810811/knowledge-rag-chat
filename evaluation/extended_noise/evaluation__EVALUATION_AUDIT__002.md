BM25 原理 |
| `chatbot_nlu.md` | 意图识别、FAQ 匹配、多轮对话 |
| `data_pipeline.md` | 增量同步、清洗、调度重试、幂等 |
| `security_compliance.md` | 认证鉴权、密钥轮换、提示词注入 |
| `frontend_web.md` | 状态管理、富文本渲染、XSS、流式渲染 |

外加项目自身文档（3 份 README + sample.txt）作为真实噪声，
由 `build_extended_corpus.py` 自动切分并**剔除污染块**后并入。

---

## 二、硬伤 2：相关性判定指标有假阳性

`evaluate_rag.py:evidence_containment` 统计「evidence 里有多少个**字符**出现在 chunk 中」。