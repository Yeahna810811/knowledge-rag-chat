# 智能知识库 RAG 问答与评测系统

基于 **FastAPI + Vue 3 + LangChain + BM25 + BGE/FAISS + Qwen** 构建的本地知识库 RAG 应用。

项目不仅实现文档入库、检索增强生成、答案溯源和多会话管理，还构建了完整的离线评测体系，对 **Dense、BM25、RRF Hybrid、MMR** 等检索策略进行 A/B 对比，并通过 **Paired Bootstrap + LLM-as-a-Judge** 完成检索选型与端到端质量验证。

当前正式默认检索方案为：

```text
BM25
```

Dense 与 Hybrid 继续作为可配置和实验方案保留。

---

## 1. 核心功能

- 支持 `.txt`、`.md`、`.pdf`、`.docx`、`.csv` 文档上传与解析
- `RecursiveCharacterTextSplitter` 文本切分
- BM25 稀疏检索与 JSON 持久化
- BGE Embedding + FAISS Dense Retrieval
- Dense / BM25 / RRF Hybrid 多检索模式
- MMR 去冗余能力
- PRF 查询改写能力
- `RetrievalAgent` + `GenerationAgent` 多 Agent 链路
- RAG / Chat 双模式
- Qwen / DashScope 大模型生成
- Answer Sources 来源溯源
- `session_id` 多会话上下文管理
- LangSmith 可选链路追踪
- FastAPI REST API
- Vue 3 + TypeScript Web 界面
- Docker 容器化
- GitHub Actions CI
- Webhook 自动化接口
- 完整 Retrieval / RAG 离线 Benchmark

---

## 2. 系统架构

```text
                     ┌─────────────────────┐
                     │       用户问题       │
                     └──────────┬──────────┘
                                │
                         RetrievalAgent
                                │
                     ┌──────────▼──────────┐
                     │ KnowledgeRetriever  │
                     └──────┬───────┬──────┘
                            │       │
                         BM25      Dense
                       默认主路   BGE + FAISS
                            │       │
                            └── RRF ┘
                               可选
                                │
                              Top-K
                                │
                        GenerationAgent
                                │
                              Qwen
                                │
                     Answer + Sources
```

文档入库链路：

```text
Upload
  ↓
Document Loader
  ↓
DocumentParseAgent
  ↓
Recursive Chunking
  ↓
KnowledgeRetriever
  ├── BM25 Index
  └── FAISS Index（按模式加载）
```

---

## 3. Retrieval 设计

当前应用默认配置：

```env
RETRIEVAL_MODE=bm25
RETRIEVAL_DENSE_FALLBACK=true
RETRIEVAL_LAZY_DENSE=true
RETRIEVAL_QUERY_REWRITE=prf
```

### BM25

当前正式默认方案。

适合技术知识库中的：

- API 名称
- 配置字段
- 文件名
- 类名
- 模型名
- 端口
- 固定技术术语

BM25 索引支持：

```text
add_documents
search
save
load
clear
```

并使用 JSON 进行本地持久化。

### Dense

使用：

```text
BAAI/bge-small-zh-v1.5
+
FAISS
```

用于语义向量检索。

### Hybrid

将：

```text
BM25 Ranking
+
Dense Ranking
↓
RRF
```

进行排名融合。

Hybrid 已完整实现，但当前 Benchmark 中没有超过 BM25，因此没有设为默认方案。

### Query Rewrite

检索层支持：

```text
off
prf
llm
```

三种查询改写策略。

当前配置默认：

```text
prf
```

正式 Retrieval Benchmark 的指标主要用于比较原始 Dense / BM25 / Hybrid 检索策略，不将查询改写收益与检索器选型结果混为一谈。

---

## 4. 快速开始

### 4.1 创建环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

安装后端依赖：

```bash
pip install -r frontend/local_rag/requirements.txt
```

---

### 4.2 配置环境变量

复制：

```bash
cp frontend/local_rag/.env.example frontend/local_rag/.env
```

至少配置：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

核心配置示例：

```env
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
CHAT_MODEL=qwen-plus

CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=4

RETRIEVAL_MODE=bm25
RETRIEVAL_DENSE_FALLBACK=true
RETRIEVAL_LAZY_DENSE=true

RETRIEVAL_QUERY_REWRITE=prf
QUERY_REWRITE_WEIGHT=0.3
```

---

### 4.3 构建前端

```bash
cd frontend/web
npm install
npm run build
cd ../..
```

开发模式：

```bash
cd frontend/web
npm run dev
```

---

### 4.4 启动项目

项目根目录：

```bash
python run.py
```

访问：

```text
Web:
http://127.0.0.1:8000

Swagger:
http://127.0.0.1:8000/docs
```

---

## 5. API

| Method | Endpoint | 功能 |
| --- | --- | --- |
| POST | `/api/upload` | 上传文档并写入知识库 |
| POST | `/api/ask` | RAG / Chat 问答 |
| GET | `/api/status` | 查询知识库与检索状态 |
| GET | `/api/history` | 查询指定会话历史 |
| DELETE | `/api/reset` | 清空知识库 |
| POST | `/api/clear_history` | 清空指定会话 |
| POST | `/api/webhook` | 自动化 / CI Webhook |

问答示例：

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question":"这份文档主要讲了什么？",
    "session_id":"demo",
    "mode":"rag"
  }'
```

---

## 6. Benchmark

### 6.1 正式评测配置

正式数据集：

```text
evaluation/eval_dataset_v3.jsonl
```

规模：

```text
100 questions
├── 80 answerable
└── 20 unanswerable
```

正式 Retrieval Benchmark：

```text
44 corpus files
46 chunks

chunk_size = 500
chunk_overlap = 50
```

其中包含真实项目文档生成的 Hard Negative。

---

## 7. Retrieval A/B 实验

正式运行命令：

```bash
python evaluation/evaluate_retrieval_ab.py \
  --dataset evaluation/eval_dataset_v3.jsonl \
  --dense localbge \
  --include-noise \
  --metric bigram \
  --ks 1 3 5 \
  --chunk-size 500 \
  --chunk-overlap 50 \
  --output-dir evaluation/results/retrieval_final
```

结果：

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense / BGE | 62.5% | 77.5% | 83.8% | 0.7060 |
| **BM25** | **82.5%** | **95.0%** | **97.5%** | **0.8869** |
| RRF Hybrid 1:1 | 70.0% | 88.7% | 92.5% | 0.7927 |
| Hybrid + MMR | 70.0% | 87.5% | 93.8% | 0.7860 |

BM25 相比 Dense：

```text
Hit@1:
62.5% → 82.5%

提升 20 个百分点
```

因此当前知识库场景最终选择：

```text
BM25
```

作为默认 Retrieval。

---

## 8. 统计显著性检验

运行：

```bash
python evaluation/significance_test.py \
  --dataset evaluation/eval_dataset_v3.jsonl \
  --rounds 10000
```

Dense 相对 BM25：

```text
ΔMRR = -0.1808
95% CI = [-0.2848, -0.0781]

Bootstrap p < 0.001
Sign Test p = 0.003
```

说明在当前 Benchmark 中，BM25 相比 BGE Dense 的优势具有统计支持。

对于不同权重的 RRF Hybrid，当前实验均未超过 BM25。

因此没有为了增加技术复杂度而强行采用 Hybrid。

---

## 9. End-to-End RAG Benchmark

最终使用：

```text
BM25
↓
RetrievalAgent
↓
GenerationAgent
↓
Qwen
↓
LLM-as-a-Judge
```

运行：

```bash
python evaluation/evaluate_rag.py \
  --retriever bm25 \
  --judge \
  --output-dir evaluation/results/rag_final
```

### Retrieval

| Metric | Result |
| --- | ---: |
| Hit@1 | 81.25% |
| Hit@3 | 95.0% |
| Hit@5 | 97.5% |
| MRR | 0.8806 |

### LLM-as-a-Judge

| Metric | Result |
| --- | ---: |
| Correctness | **97.0%** |
| Faithfulness | **99.3%** |
| Relevance | **96.9%** |
| Safe Refusal | **100%** |

Safe Refusal 仅统计：

```text
20 道知识库外不可答题
```

全部 100 道问题均完成 Judge。

> LLM-as-a-Judge 属于自动评测，不等同于人工专家标注。

---

## 10. Latency

最终 BM25-RAG：

### Retrieval

```text
Average = 0.82 ms
P50     = 0.66 ms
P95     = 1.13 ms
```

### End-to-End

```text
Average = 3.49 s
P50     = 3.29 s
P95     = 5.33 s
```

当前系统主要耗时来自 LLM Generation，而不是 BM25 Retrieval。

---

## 11. 为什么有两组 Retrieval 数字

Retrieval A/B：

```text
BM25 Hit@1 = 82.5%
MRR        = 0.8869
```

End-to-End RAG：

```text
Hit@1 = 81.25%
MRR   = 0.8806
```

两者用途不同：

```text
evaluate_retrieval_ab.py
→ Retrieval 策略 A/B 与选型

evaluate_rag.py
→ 完整 RAG End-to-End 评测
```

因此项目不会把两组指标混为同一实验。

---

## 12. 测试

Retrieval 算法测试：

```bash
python tests/test_retrieval.py
```

KnowledgeRetriever 测试：

```bash
python tests/test_knowledge_retriever.py
```

当前已覆盖：

```text
BM25
Dense
Hybrid
RRF
MMR
Persistence
Fallback
Restart Recovery
Retrieval Routing
异常路径
```

此前测试结果：

```text
26 passed
38 passed
```

---

## 13. Docker

```bash
docker compose up --build
```

---

## 14. 项目结构

```text
knowledge-rag-chat/
├── README.md
├── run.py
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│
├── frontend/
│   ├── web/
│   │   └── src/
│   │
│   └── local_rag/
│       ├── api/
│       ├── config/
│       ├── core/
│       │   ├── agents/
│       │   └── retrieval/
│       ├── services/
│       └── utils/
│
├── tests/
│   ├── test_retrieval.py
│   └── test_knowledge_retriever.py
│
└── evaluation/
    ├── EVALUATION_AUDIT.md
    ├── README.md
    ├── eval_dataset_v3.jsonl
    ├── eval_dataset_oral_v3.jsonl
    ├── corpus/
    ├── extended_noise/
    ├── evaluate_retrieval_ab.py
    ├── significance_test.py
    ├── benchmark_runtime.py
    ├── evaluate_rag.py
    └── results/
        ├── retrieval_final/
        ├── rag_final/
        └── archive/
```

---

## 15. 正式结果

Retrieval：

```text
evaluation/results/retrieval_final/
├── ab_retrieval.json
├── ab_retrieval_details.csv
└── significance_test.txt
```

End-to-End RAG：

```text
evaluation/results/rag_final/
├── evaluation_details.csv
└── evaluation_summary.json
```

历史 Prompt 对照实验：

```text
evaluation/results/archive/prompt_ablation/
```

历史实验只用于记录项目迭代，不作为当前正式 Benchmark 结论。

完整评测设计与审计过程见：

```text
evaluation/EVALUATION_AUDIT.md
```

---

## 16. 实验结论与边界

当前实验表明：

```text
BM25
```

更适合本项目当前技术文档型知识库。

但该结论只适用于当前：

```text
数据集
语料规模
问题类型
bge-small-zh-v1.5
```

并不代表 BM25 在所有 RAG 场景中都优于 Dense Retrieval。

如果后续：

- 知识库扩大；
- 用户提问更加口语化；
- 换用更强 Embedding；
- 增加 Cross Encoder Reranker；

需要重新运行 Benchmark 决定新的 Retrieval 策略。