.8 / 3457.7 / 8887.4 | 1233.7 / 883.9 / 2603.2 |

检索指标两次完全一致（检索不依赖生成 Prompt）；延迟受模型服务负载影响，跨批次对比仅供参考。

## 注意

- `--judge` 会真实调用大模型（默认用 `.env` 的 `CHAT_MODEL`），产生 token 费用。
- 对比 Prompt 优化前后：固定同一份 v2 测试集，分别跑并保存结果到 `results/`（参考上面 before/after 两份文件的命名），保证可比性。当前生成 Prompt 位于 `frontend/local_rag/core/agents/generation_agent.py` 的 `RAG_SYSTEM_PROMPT`。