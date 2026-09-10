ess / Faithfulness / SafeRefusal 显著提升）。

## 项目结构

```text
knowledge-rag-chat/
├── run.py
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── evaluation/
│   ├── evaluate_rag.py
│   ├── evaluate_with_ragas.py
│   └── results/
└── frontend/
    ├── web/                 # Vue 3 + TypeScript
    └── local_rag/
        ├── app.py
        ├── api/routes.py
        ├── core/agents/     # 多 Agent 调度
        └── services/
```