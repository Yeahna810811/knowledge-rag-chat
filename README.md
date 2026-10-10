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
- `session_id` 多会话上下文管理（SQLAlchemy + MySQL 持久化）
- Redis Retrieval 结果缓存（Knowledge-Base Version 失效，见第 5 节）
- Redis 固定窗口限流（`/api/ask`，按 `session_id`，故障时 fail-open）
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

整体架构（含 Stage 3~5 的工程化层）：

```mermaid
flowchart TB
    subgraph Client["Client"]
        VUE["Vue 3 + TypeScript<br/>frontend/web"]
    end

    subgraph API["FastAPI · frontend/local_rag/api"]
        RATE["RateLimiter<br/>Redis Lua / 事务管道 · fail-open"]
        ROUTES["Routes<br/>POST /api/ask<br/>POST /api/ask/stream<br/>POST /api/upload"]
    end

    subgraph OBS["Observability · Stage 4"]
        RID["Request ID<br/>X-Request-ID + ContextVar"]
        LOG["Structured Logging<br/>ttft / retrieval / generation / total"]
        ERR["Error Classification<br/>9 类 + HTTP 语义映射"]
        GUARD["Retry · Timeout · Stream Guard<br/>首 token 前可重试，之后绝不重试"]
    end

    subgraph Core["Service & Agents"]
        KS["KnowledgeService"]
        ORCH["AgentOrchestrator"]
        RA["RetrievalAgent"]
        GA["GenerationAgent"]
    end

    subgraph Retrieval["Retrieval"]
        KR["KnowledgeRetriever<br/>BM25 主路 · Dense 兜底 · RRF 可选"]
        BM25["LexicalStore / BM25"]
        FAISS["VectorStore / FAISS + BGE"]
    end

    subgraph Store["Storage"]
        MYSQL[("MySQL / SQLite<br/>会话持久化")]
        REDIS[("Redis<br/>检索缓存 + 限流")]
    end

    LLM["Qwen / DashScope"]

    VUE -->|HTTP| RATE
    VUE -->|SSE| RATE
    RATE --> ROUTES
    ROUTES --> RID --> KS
    KS --> ORCH
    ORCH --> RA
    ORCH --> GA
    RA --> KR
    KR --> BM25
    KR --> FAISS
    GA --> LLM
    KS --> MYSQL
    RA -.->|cache| REDIS
    RATE -.-> REDIS
    KS -.-> LOG
    KS -.-> ERR
    GA -.-> GUARD
    GUARD -.-> LOG
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

## 4. Conversation Persistence / MySQL

会话历史已经从进程内存（`dict`）迁移到数据库：

```text
SQLAlchemy 2.x ORM
+
MySQL 8 (utf8mb4)
```

浏览器刷新、FastAPI 重启、容器重建之后，历史都还在。

### 4.1 数据库存什么 / LLM 拿什么

这两件事是分开的，不要混为一谈：

| 维度 | 内容 |
| --- | --- |
| 数据库保存 | **完整历史**，不限轮数 |
| LLM Context | **最近 10 轮**（`MAX_HISTORY_TURNS = 10`） |

即：`get_recent_history(session_id, limit=10)` 只取最近 10 轮喂给 `GenerationAgent`，
避免 context 无限增长和 token 成本失控；而 `messages` 表保留全部问答记录，
`/api/history` 可以查到完整历史。

### 4.2 表结构

```text
conversations
├── id            BIGINT PK
├── session_id    VARCHAR(128) UNIQUE, INDEX
├── created_at    DATETIME
└── updated_at    DATETIME

messages
├── id               BIGINT PK
├── conversation_id  BIGINT FK -> conversations.id  ON DELETE CASCADE
├── question         LONGTEXT
├── answer           LONGTEXT
├── mode             VARCHAR(16)   -- 'rag' / 'chat'
└── created_at       DATETIME       INDEX (conversation_id, created_at)
```

删除 `conversation` 时，其 `message` 一并删除（ORM 级联 + 外键 `ON DELETE CASCADE`）。

只有 `question / answer / mode` 会落库，API Key 等敏感信息不进数据库。

### 4.3 DATABASE_URL

```bash
# Docker Compose 内部网络（host 写服务名 mysql，不要写 localhost）
# docker-compose.yml 里已经配好了，容器里不需要再设
DATABASE_URL=mysql+pymysql://rag_user:rag_password@mysql:3306/knowledge_rag?charset=utf8mb4

# 本机 Python 进程（python run.py）直连容器里的 MySQL：
# 先把 docker-compose.yml 里 mysql 的端口映射注释打开（默认关闭，避免和宿主机
# 已有的 MySQL 抢 3306），然后用 3307 连
DATABASE_URL=mysql+pymysql://rag_user:rag_password@127.0.0.1:3307/knowledge_rag?charset=utf8mb4

# 不配置时回落到 SQLite 文件 frontend/local_rag/data/rag_app.db
#（本地调试 / CI 用它，不需要 MySQL）
```

密码等配置一律走环境变量或 `frontend/local_rag/.env`，不写死在代码里。

### 4.4 本地启动 MySQL（不用 Docker Compose 跑应用）

```bash
docker compose up -d mysql

# 等 healthcheck 通过（看到 healthy 才算真的可用）
docker compose ps mysql
```

healthcheck 用的是 `mysqladmin ping`。注意 compose 文件里必须写成 `$$MYSQL_ROOT_PASSWORD`
（双美元符转义）：单 `$` 会被 compose 按**宿主机**变量展开成空串，探测命令退化成
`mysqladmin ping -p`，只会打印 `Enter password:` 却仍然返回 0——一个假阳性的健康检查。

### 4.5 Docker Compose 启动

```bash
docker compose up --build
```

`rag-app` 会等 `mysql` 的 healthcheck 通过后才启动。数据库数据在 `mysql_data` volume 中：

```bash
docker compose down      # 数据保留，重新 up 后历史仍在
docker compose down -v   # 才会删除数据库 volume
```

### 4.6 测试

```bash
# 会话持久化测试（默认用临时 SQLite，不需要 MySQL）
python tests/test_database.py

# 指向真实 MySQL 做集成测试（端口映射打开后用 3307）
TEST_DATABASE_URL=mysql+pymysql://rag_user:rag_password@127.0.0.1:3307/knowledge_rag?charset=utf8mb4 \
  python tests/test_database.py

# 在容器网络内直接跑（host 写服务名 mysql，无需端口映射）
docker run --rm --network knowledge-rag-chat_default -v "$PWD":/app -w /app python:3.11-slim \
  sh -c "pip install -q 'SQLAlchemy>=2.0,<3.0' 'PyMySQL>=1.1,<2.0' && \
    TEST_DATABASE_URL=mysql+pymysql://rag_user:rag_password@mysql:3306/knowledge_rag?charset=utf8mb4 \
    python tests/test_database.py"
```

不设 `TEST_DATABASE_URL` 时 MySQL 集成用例显示 `[skip]` 而不是失败，
所以 CI 里永远可以安全执行这条命令。

CI 里有一个独立的 `db-mysql` job，真正拉起 `mysql:8.4` 服务容器跑一遍
（MySQL 特有的 utf8mb4、LONGTEXT、DATETIME 秒级精度排序并列，只有在这里覆盖得到）。
它是独立 job，MySQL 服务出问题时不会影响 SQLite 那条冒烟流水线。

### 4.7 验证持久化

```bash
# 用同一个 session_id 发两条消息
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"这份文档讲了什么？","session_id":"demo","mode":"rag"}'

curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"再展开说一下","session_id":"demo","mode":"rag"}'

# 查历史
curl "http://127.0.0.1:8000/api/history?session_id=demo"

# 重启服务后历史必须仍在；换一个 session_id 则没有这段历史
curl "http://127.0.0.1:8000/api/history?session_id=other"

# 清空
curl -X POST http://127.0.0.1:8000/api/clear_history \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo"}'
```

`/api/status` 会额外返回数据库状态：

```json
"database": { "backend": "mysql", "healthy": true }
```

数据库临时不可用时 `/api/status` 不会崩，只会把 `healthy` 置为 `false`。

---

## 5. Redis Cache & Rate Limit

### 5.1 Redis 承担什么 / MySQL 承担什么

Redis 在本项目里是**加速器，不是数据源**。两者的职责严格分开：

| 组件 | 承担 | 数据丢了会怎样 |
|---|---|---|
| MySQL | `Conversation` / `Message` 会话历史（唯一持久化数据源） | 会话历史丢失，**不可接受** |
| Redis | Retrieval 结果缓存 | 缓存全部 miss，检索照常执行，只是慢 |
| Redis | `/api/ask` 限流计数 | 限流暂时失效（fail-open），请求照常放行 |

会话历史**不会**迁往 Redis。MySQL 仍然是它的唯一落点。

### 5.2 架构图

```text
FastAPI
├── MySQL
│   └── Conversation History
│       ├── Conversation (session_id, created_at, updated_at)
│       └── Message (question, answer, mode, created_at)
│
├── Redis
│   ├── Retrieval Cache      rag:retrieval:v{n}:{mode}:{rewrite}:{k}:{hash}
│   ├── Knowledge Version    rag:kb_version
│   └── Rate Limit           rag:rate_limit:{session_id}:{window}
│
└── RAG
    ├── BM25        （默认主路，稀疏索引落盘）
    ├── Dense       （BGE + FAISS，兜底 / A-B）
    └── Hybrid      （RRF 融合，实验用）
```

一次 RAG 请求的实际链路：

```text
question
  ↓
构造 Cache Key（含 kb_version）
  ↓
Redis GET ── Hit ──→ 反序列化回检索结果 ──┐
  │                                        │
  └── Miss ──→ KnowledgeRetriever ──→ Redis SETEX ──┘
                                          ↓
                                   GenerationAgent（照常执行）
                                          ↓
                                        Answer
                                          ↓
                              MySQL append_turn()
```

### 5.3 为什么只缓存 Retrieval，不缓存最终答案

项目支持多会话上下文，同一个问题在不同 session 里含义可能完全不同——
「它为什么这样设计？」里的「它」指代什么，取决于前文。
所以 **`query -> final answer` 是不能做全局缓存的**，那会把 A 会话的答案错给 B 会话。

本阶段只缓存 `query -> documents/chunks`。LLM 生成照常执行：
省掉的是重复的 BM25 / 向量检索开销，不会污染带上下文的最终答案。

缓存内容序列化成明确 JSON，格式沿用本项目 `KnowledgeRetriever.search()`
的真实返回结构（不是 LangChain 的 `page_content`，分数在 metadata 里）：

```json
[
  {
    "content": "……",
    "metadata": { "source": "doc.md", "retrieval_score": 0.87, "retrieved_by": "bm25" }
  }
]
```

不做 pickle：pickle 反序列化会执行任意代码，缓存被污染就等于 RCE。

### 5.4 Cache Key 设计

```text
rag:retrieval:{kb_version}:{retrieval_mode}:{rewrite_mode}:{top_k}:{query_hash}
```

五个维度缺一不可，少任何一个都会串数据：

| 维度 | 作用 |
|---|---|
| `kb_version` | 知识库更新后旧缓存必须失效 |
| `retrieval_mode` | `bm25` / `dense` / `hybrid` 召回的文档不同 |
| `rewrite_mode` | 查询改写开与不开，召回结果不同 |
| `top_k` | k=4 的结果不能给 k=8 的请求用 |
| `query_hash` | `SHA-256`，不把原文放进 key |

`query_hash` 用 SHA-256 是因为：原始问题可能很长、含任意字符，还可能带隐私信息，
不适合直接进 Redis key。

归一化只做 `strip` + 合并连续空白。**刻意不做**小写化 / 去标点 / 繁简转换——
那些会改变语义，属于「缓存里最难排查的一类 bug」。

### 5.5 Knowledge Base Version 失效

知识库内容变了，旧缓存必须失效。做法是把版本号拼进 key：

```text
redis> GET rag:kb_version
"12"

上传新文档 / 清空知识库成功
  ↓
INCR rag:kb_version  →  13
  ↓
新检索用 rag:retrieval:v13:*，旧 key 不再命中，等 TTL 自动过期
```

- **不用 `KEYS rag:*` 扫全库**：生产上会阻塞 Redis 单线程，是自杀式操作。
- **不做 `FLUSHDB`**：会连带清掉限流计数和其它共用实例的数据。
- **只在确认成功后 bump**：`upload` / `reset` 失败时版本号不变，不白白丢弃缓存。
- **Redis 全丢时**：`rag:kb_version` 重新从 1 起。旧缓存本来就跟 Redis 一起没了，
  版本号从头开始不会串数据。

### 5.6 TTL

```bash
REDIS_CACHE_ENABLED=true
REDIS_CACHE_TTL_SECONDS=600
```

TTL 负责「缓存别永生」，**不负责正确性**——正确性由 5.5 的版本号保证。
所以 TTL 设长一点是安全的。

### 5.7 Rate Limit 算法

对 `/api/ask` 做固定窗口限流，默认 60 次 / 60 秒，按 `session_id` 计数
（项目暂无 User / JWT 体系，`session_id` 是现成的隔离维度）。

Key：

```text
rag:rate_limit:{session_id}:{window_index}
# window_index = int(time.time()) // window_seconds，窗口自动滚动，无需清理任务
```

计数必须是原子的。不能用「GET 然后 SET」——两个请求同时 GET 到 59，
都判为未超限、都 SET 60，窗口内实际放行几百个请求，限流形同虚设。

首选实现是一次 EVAL 完成 INCR / EXPIRE / TTL 的 Lua 脚本：

```lua
local count = redis.call("INCR", KEYS[1])
if count == 1 then redis.call("EXPIRE", KEYS[1], ARGV[1]) end
local ttl = redis.call("TTL", KEYS[1])
if ttl < 0 then redis.call("EXPIRE", KEYS[1], ARGV[1]); ttl = tonumber(ARGV[1]) end
return {count, ttl}
```

部分托管 Redis 与 `fakeredis` 不支持 EVAL，此时自动回落到事务管道
（`MULTI/EXEC` 包住 INCR + TTL，事后再补 EXPIRE），原子性等价。

这里有个容易写错的点：**「不支持 Lua」和「这次连不上」必须分开处理**。
两者拿到的结果都是「EVAL 没返回」，但如果一律判定为不支持，
一次几秒钟的网络抖动就会让限流永久退化到管道实现。
`RedisClient.scripting_supported` 用三态（未试过 / 支持 / 确认不支持）区分：
只有服务端明确报 `unknown command` / `NOSCRIPT` 才永久关闭 Lua，
连接类的临时失败下次仍会重试 Lua。

超限返回 `429 Too Many Requests`，响应体带 `detail`，响应头带 `Retry-After`。
成功响应的结构完全不变。

### 5.8 Fail-open 策略（重要）

**Redis 故障时，限流采用 fail-open：请求照常放行，同时记 warning。**

理由：限流的目的是保护系统，不是拒绝服务。Redis 已经挂了，
再让所有请求 429，等于把 Redis 的单点故障放大成全站不可用。

Redis 完全关闭时的完整行为：

| 能力 | 行为 |
|---|---|
| RAG 问答 | 缓存 miss → 走原检索 → 正常生成答案（只是没有加速） |
| Chat 问答 | 正常工作 |
| MySQL 会话历史 | 正常读写 |
| Retrieval Cache | 退化为「每次都真检索」，写缓存静默跳过 |
| Rate Limit | fail-open，放行并记录 warning |
| `/api/status` | 返回 `redis.healthy=false`，自身不 500 |

Redis **不是**系统的单点故障。

### 5.9 环境变量

```bash
# Redis 连接（密码直接写在 URL 里，不另设明文变量）
REDIS_URL=redis://127.0.0.1:6379/0          # 本机
REDIS_URL=redis://redis:6379/0              # Docker Compose 内部网络（host 写服务名）
REDIS_URL=redis://:your_password@redis:6379/0

# 检索缓存
REDIS_CACHE_ENABLED=true
REDIS_CACHE_TTL_SECONDS=600

# 限流
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW_SECONDS=60
```

### 5.10 启动方式

Docker Compose（推荐，Redis 随应用一起起）：

```bash
docker compose up -d redis
docker compose ps          # redis 应为 healthy
docker compose exec redis redis-cli ping   # 应返回 PONG
```

只起 Redis 给本机 Python 进程用：

```bash
docker compose up -d redis
# 打开 docker-compose.yml 里 redis 的 ports 注释（127.0.0.1:6379:6379），
# 然后 REDIS_URL=redis://127.0.0.1:6379/0 python run.py
```

`compose` 里 Redis **没有挂 volume**：里面放的是缓存和限流计数，
全是可重建数据，丢了最坏结果是「缓存全 miss 一次」。
不为一堆可丢弃的数据维护备份与恢复流程。

rag-app 声明了 `depends_on: redis: service_healthy`，但**应用本身不依赖这个**——
Docker 的健康检查和应用容错是两个层级，代码侧始终按 Redis 随时会挂来写。

### 5.11 测试

```bash
# 单元用例（fakeredis，不需要 Redis 服务）
python tests/test_redis.py

# 指向真实 Redis 7 做集成测试
TEST_REDIS_URL=redis://127.0.0.1:6379/0 python tests/test_redis.py
```

两层覆盖的差异值得说明：`fakeredis` **不支持 Lua**（连 EVAL 命令都没有），
所以「限流的 Lua 分支」「真实 TTL 过期」「跨客户端读缓存」只有连真实 Redis
时才覆盖得到。CI 里这两层分别在 `backend-smoke` 和 `db-redis` 两个 job 执行。

---

## 6. Async I/O & True SSE Streaming

阶段 3 只解决两件事：**别让阻塞 I/O 堵死事件循环**，以及**让回答真的逐字吐出来**。

### 6.1 先修掉一个真实的"假异步"

`/api/upload` 原本长这样：`async def` 端点里直接调用同步的入库（解析 + 切分 + embedding）。
这是异步代码里最典型也最隐蔽的坑——`async def` 并不会让里面的同步代码变异步，
它只是承诺"我不阻塞事件循环"，而阻塞的入库把这个承诺打破了：
一次上传期间，整个进程所有并发请求（包括正在流的 SSE 连接）全部卡死。

```python
# 修好后：显式的线程池卸载，async 的收益才真的拿得到
result = await anyio.to_thread.run_sync(
    lambda: service.ingest_upload(file.filename or "upload.bin", content)
)
```

反过来，`/api/ask` **刻意保留同步 `def`**：FastAPI 会自动把同步端点丢进线程池，
效果和手写 `to_thread` 一样，事件循环不会被堵住。把它改成 `async def`
却不去卸载内部的阻塞调用，反而会把"自动卸载"变成"真阻塞"。

### 6.2 事件协议

`POST /api/ask/stream` 返回 `text/event-stream`，事件顺序固定：

| event | 何时 | data |
| --- | --- | --- |
| `meta` | 立刻（握手后第一帧） | `question` / `session_id` / `mode` |
| `sources` | 检索完成后、首个 token 之前（仅 rag） | `sources[]` |
| `trace` | 生成开始前 | `agent_trace[]` |
| `delta` | 每收到一块模型输出 | `text`（增量，非全文） |
| `done` | 生成结束 | 完整 `answer` + `sources` + `history_turns` |
| `error` | 任一步失败 | `message` |
| `: ping` | 静默超过 `SSE_HEARTBEAT_SECONDS` | 无（注释帧，前端忽略） |

`sources` 必须排在 `delta` 之前：用户要能边看答案边对照出处，
而不是等答案说完才看到引用。

出错时发 `error` **事件**而不是抛异常——响应头已经在 SSE 握手时发出去了，
这时候抛异常只能中断连接，前端拿不到任何可读信息。

### 6.3 请求示例

```bash
curl -N -X POST http://localhost:8000/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"报销流程是什么","session_id":"s1","mode":"rag"}'
```

```text
event: meta
data: {"question": "报销流程是什么", "session_id": "s1", "mode": "rag"}

event: sources
data: {"sources": [...], "mode": "rag"}

event: delta
data: {"text": "报销"}
...
event: done
data: {"answer": "报销流程是…", "history_turns": 1, ...}
```

### 6.4 为什么不用浏览器的 EventSource

`EventSource` 只能发 GET：既装不下长问题（URL 长度限制），
也带不了 `AbortSignal`——而"生成一半点停止"恰恰是流式最刚需的交互。
所以前端用 `fetch` + `ReadableStream` 自己按 `\n\n` 切帧，
配合 `AbortController` 实现停止生成。

### 6.5 心跳与断连

- **心跳**是注释帧 `: ping`（冒号开头），前端天然忽略，不会污染事件流。
  作用是穿透 nginx / 代理的空闲超时（很多默认 60s 就掐连接）。
- 心跳实现上必须做**生产者/消费者分离**。直接写
  `with anyio.fail_after(t): await gen.__anext__()` 会把超时取消砸进上游
  生成器内部的 await（比如那次还没返回的 LLM 网络读），
  **把生成器本身取消掉**——心跳反而成了杀死流的东西。
  实测朴素实现下 3 个事件只能收到 1 个。
- **客户端断开**（关页面 / AbortController / 代理超时）、`CancelledError`、
  `GeneratorExit`、以及 LLM 中途报错，**都不写 MySQL**，只打一条结构化日志
  （`reason=client_disconnected / stream_closed / generation_error`，带已丢弃
  内容的长度与预览）。保留半截答案看着是"不浪费"，代价是它会被当作
  assistant 上下文喂回下一轮，模型会学着说半截话——比丢掉这一轮糟得多。
  真要保留半成品，应该加 `status=cancelled/failed/incomplete` 的数据模型，
  而不是复用正常 `messages` 表。本阶段不扩表。
- **MySQL 只在生成正常走完时写一次**，不是 per-token 写：
  `done` 之前那唯一一次 `_append_turn`。
- 那唯一一次写入用 `CancelScope(shield=True)` 兜住：最后一个 token 发出后，
  客户端随时可能断开，取消会在下一个 `await` 点投递，不屏蔽就变成
  "答完了却没存"——和上面"没答完却存了"是两个方向的同一种错。
  注意 shield 只对 anyio 自己投的取消生效（Starlette 收到 `http.disconnect`
  后 cancel 包住 `body_iterator` 的 task group 正是这条路径）；
  裸 `asyncio.Task.cancel()` 由 asyncio 直接往协程里抛，会绕过 shield。

### 6.6 限流

`/api/ask/stream` 与 `/api/ask` 共用同一套按 `session_id` 计数的固定窗口限流。
流式场景下 429 **必须在流开始之前**判定：响应头一旦发出去就改不了状态码了。
被限掉的连接不会消耗任何 LLM 额度。

### 6.7 流式链路的线程池边界

| 环节 | 处理方式 | 原因 |
| --- | --- | --- |
| 检索（BM25 / FAISS / Redis 缓存） | `to_thread` | 同步代码，直接 await 会堵死循环 |
| 读/写会话历史 | `to_thread` | 同步 SQLAlchemy |
| LLM 生成 | 原生 `await` | 真正的网络 I/O，这才是异步的收益所在 |
| 首次 runtime 初始化 | `to_thread` | 可能加载几百 MB 的 embedding 模型 |

---

## 7. Reliability & Observability

阶段 4 不增加任何业务能力，只解决一件事：**出故障时能说清发生了什么**。
能回答"是哪一类错、等了多久、重试了几次、这次请求是哪一次"。

### 7.1 Request ID

每次 HTTP 请求都带一个 `request_id`：

- 客户端传了合法的 `X-Request-ID`（`[A-Za-z0-9._:-]{1,64}`）就复用，便于跨服务串联
- 否则服务端生成 UUID hex
- 不合法时**静默替换**而不是报 400——请求头是客户端可控输入，
  为它的格式问题惩罚业务请求不划算；而且这个值会进日志和 SSE 帧，
  放行任意字符串等于给别人一个日志注入的入口
- 响应头 `X-Request-ID` 一定会返回；SSE 的 `meta` 事件里也带一份

传播用 `ContextVar` 而不是全局变量或显式层层传参：

- 全局变量：并发请求之间必然串号——这是最糟的一类 bug，日志看着正常，
  但 A 请求的耗时被记到 B 头上
- 显式传参：靠"每个函数都记得往下传"，漏一个就断链
- ContextVar：一次绑定，当前任务（含其 await 链）内都能读到，并发天然隔离

两个已实测过的边界：

1. `anyio.to_thread.run_sync` 会复制当前 context，所以卸载到线程池的
   检索 / 读写库同样拿得到 `request_id`
2. 异步生成器（`astream_ask`）是在路由返回之后才被迭代的，路由里的绑定
   已经出栈了。所以 **service 层必须在生成器内部再绑一次**，并在 `finally`
   里复位——否则 ContextVar 会泄漏到下一个请求

### 7.2 结构化日志

只依赖标准库 `logging`，格式是单行 `k=v`：

```text
event=llm_complete request_id=9f2c… session_id=s1 mode=rag status=ok generation_ms=1820.4 answer_len=137
```

选 `k=v` 而不是 JSON 的理由很实际：出故障时人是拿着终端在 grep，
`grep request_id=abc` 比 `jq` 快得多。前四个字段（`event` / `request_id` /
`session_id` / `mode`）永远固定在前，扫日志可以靠肌肉记忆。
上下文字段由 `log_event` 自动补齐，调用方不手抄——手抄就容易抄成上一个请求的。

事件清单：

| event | 何时 | 默认级别 |
| --- | --- | --- |
| `request_start` | 请求进入业务层 | INFO |
| `retrieval_complete` | 检索完成（含 `cache_hit` / `retrieval_ms`） | INFO |
| `llm_start` | 调用 LLM 前 | INFO |
| `llm_first_token` | 流式首个真实 token | INFO |
| `llm_complete` | 生成结束或失败 | INFO / WARNING |
| `llm_retry` | 发生一次重试 | WARNING |
| `persistence_complete` / `persistence_failed` | 落库结果 | INFO / WARNING |
| `request_complete` / `request_failed` | 请求结束 | INFO / WARNING |
| `stream_cancelled` | 客户端断开 | INFO |
| `redis_unavailable` | Redis 故障降级 | WARNING |
| `rate_limited` | 自己的限流命中 | WARNING |

字段纪律：**不写完整 question / answer / 文档内容**，只写 `question_len` /
`answer_len`。原因不是洁癖——日志会被收集、留存、被第三个人看到；而完整内容
对排障几乎没有帮助，"这次请求 5123ms"有用，"这次请求的内容是……"没用。
异常消息统一过 `safe_message()`：脱敏 `sk-` / `Bearer` / key-value 形式的凭据、
压掉控制字符（防日志注入）、截断到 200 字符。

### 7.3 错误分类

上游 SDK 抛出来的异常是个大杂烩，直接往上抛调用方只能 `except Exception`，
然后要么一股脑重试（把 401 也重试了），要么一股脑不重试（把网络抖动也放过了）。
`frontend/local_rag/observability/errors.py` 只回答三个问题：是什么类型、该不该重试、给客户端什么状态码。

| error_type | 触发 | 可重试 | HTTP |
| --- | --- | --- | --- |
| `LLM_TIMEOUT` | 超时 | ✅ | 504 |
| `LLM_RATE_LIMITED` | 上游 429 | ✅ | 502 |
| `LLM_UPSTREAM_ERROR` | 上游 5xx / 连接重置 | ✅ | 502 |
| `LLM_AUTH_ERROR` | 401 / 403 | ❌ | 502 |
| `VALIDATION_ERROR` | 400 / 422 等参数问题 | ❌ | 400 |
| `RETRIEVAL_ERROR` | 检索失败 | ❌ | 500 |
| `DATABASE_ERROR` | 数据库异常 | ❌ | 500 |
| `REDIS_UNAVAILABLE` | Redis 故障 | ❌ | **不映射成 5xx** |
| `RATE_LIMITED` | 自己的限流 | ❌ | 429 |
| `STREAM_CANCELLED` | 取消 / 断开 | ❌ | 499 |
| `INTERNAL_ERROR` | 兜底 | ❌ | 500 |

判定顺序从"最确定"到"最模糊"：已分类 → 取消 → 状态码 → 异常类名 → 异常文本 → 兜底。
文本匹配放最后是因为它最不可靠，但也是唯一能兜住"第三方库抛了个裸
`RuntimeError('connection reset')`"的手段。

兜底分支刻意**保留**脱敏截断后的原始信息，而不是清空成一句"服务内部错误"：
把消息清空是把排障成本和安全性一起丢掉了。

### 7.4 Retry 策略

`LLM_RETRY_MAX_ATTEMPTS` 是**总尝试次数**（首次 + 重试），3 表示最多调 3 次 LLM，
写成"重试 3 次"很容易被理解成总共 4 次。退避是指数的但封顶：
不封顶的话第 10 次重试要等几分钟，用户早把页面关了。

**先关掉 SDK 内建重试。** openai SDK 默认 `max_retries=2`，它在 transport 层
静默重发：不打日志、不带 `request_id`，流式场景下还会把整条流重跑一遍。
留着它就是 `SDK retry × 应用层 retry`——失败时根本数不清到底发了几遍。
所以 `LLM_SDK_MAX_RETRIES=0`，重试逻辑全部收敛到本项目自己的
`run_with_retry` / `guarded_astream`。

重试只有一条判据：**再试一次有可能成功**。所以超时、连接重置、上游限流、
上游 5xx 重试；鉴权失败、参数错误、业务逻辑错误、数据库约束错误、
用户主动取消一律不重试。

取消必须原样向上抛——这是最容易写错的一条：如果为了重试而 `except Exception`
一把抓，用户按了停止也会被当成"失败"去重试，abort 语义整个失效。

### 7.5 流式：first-token 是重试的分界线

这是阶段 4 最重要的约束。

```text
before first token 失败  → 可以重试（用户还没收到任何内容，重试是透明的）
after  first token 失败  → 绝不重试，发 error 事件，不保存半截答案
```

原因很直观：第一次已经吐给用户"你好，我是…"，第二次重试又从头生成一遍，
用户看到的就是两遍开头。所以 `guarded_astream` 里的 `sent_any` 一旦为真，
任何错误都直接向上抛。

实现上 `sent_any = True` 必须**在 yield 之前**置位：yield 之后控制权就交给下游了，
等下游抛回来再置位已经晚了。

### 7.6 Timeout

三段超时，全部走 Settings，不硬编码：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `LLM_REQUEST_TIMEOUT_SECONDS` | 60 | 非流式一次调用的总超时 |
| `LLM_FIRST_TOKEN_TIMEOUT_SECONDS` | 30 | 流式：发起请求到首个真实 token |
| `LLM_STREAM_IDLE_TIMEOUT_SECONDS` | 60 | 流式：两个 token 之间的最大静默 |

首 token 超时和空闲超时分开是因为"排队 40s 才开始出字"和"出了两个字之后卡住"
是两种完全不同的故障。流式**不用总时长**卡——长回答天然耗时不可预测，
用总时长会把正常长回答误杀。

实现上复用阶段 3 的生产者/消费者分离：

```python
# 错误做法：取消会砸进生成器内部的 await，把生成器本身取消掉
with anyio.fail_after(t):
    item = await gen.__anext__()

# 正确做法：上游跑在独立任务里往内存流送，超时只作用在"等内存流"这一步
async with anyio.create_task_group() as tg:
    tg.start_soon(producer)
    item = await receive_stream.receive()   # 超时落在这里
```

用 `fail_after` 直接包 `__anext__()` 是阶段 3 做心跳时踩过的坑，不重新引入。
每次尝试都开一个 task group：异常从 `async with` 抛出去时 `__aexit__` 会
取消并等待子任务，不会留 dangling task 继续烧额度。
`gen.aclose()` 放 shield 里——取消已生效时不屏蔽的话清理动作自己就会被取消。

### 7.7 数据库写纪律

`append conversation turn` **不做自动重试**。

数据库写可能已经成功、只是客户端不知道结果，盲目重跑会产生重复 message。
所以保持：正常完整 generation → 单次 append；append 失败 → 记
`DATABASE_ERROR` + `persistence_failed` 日志，然后降级（前端仍拿到完整回答）。

不因为写库失败重新调用 LLM，也不因为写库失败重新生成回答。

### 7.8 Redis 仍然 fail-open

Redis 是加速器不是数据源。故障时继续记 `redis_unavailable` +
人话 warning，但**绝不因为缓存挂了让 RAG / Chat 请求失败**——
不为了统一错误处理把它改成 HTTP 500。

### 7.9 Timing 指标

统一用 `time.perf_counter()`，不用 wall clock（会被 NTP 调整、能被人为改，
算差值可能得到负数或跳变几十秒）。

| 指标 | 定义 |
| --- | --- |
| `ttft_ms` | 请求开始 → **第一个真实 LLM token 被发送**（不是 meta / heartbeat / trace） |
| `generation_ms` | LLM 调用开始 → 生成结束 |
| `retrieval_ms` | 检索耗时（chat 模式为 null / 省略） |
| `cache_ms` | 缓存查询耗时 |
| `total_ms` | 请求开始 → 请求结束 |

`ttft_ms` 只在 service 层记一次。两层各记一次的话同一条流会出现两个
`ttft_ms` 且数值不同，排障时反而多一个"到底以哪个为准"的疑问。

### 7.10 HTTP 与 SSE 的错误语义

响应头**尚未发出**时，可以按分类给状态码：429（自己的限流）/ 504（超时）/
502（上游故障）/ 400（参数）/ 500（其余）。错误响应体形如：

```json
{"request_id": "9f2c…", "error_type": "LLM_TIMEOUT", "message": "上游调用超时"}
```

但 SSE 一旦 `text/event-stream` 开始，状态码已经改不了了。此时发 `error` 事件：

```text
event: error
data: {"request_id": "9f2c…", "error_type": "LLM_TIMEOUT", "message": "上游调用超时"}
```

`message` 一律过 `safe_message()`，不含 API key、完整堆栈、provider 原始敏感信息。
服务端日志保留 `cause` 供深挖，但不进响应体。

### 7.11 环境变量

```bash
LLM_REQUEST_TIMEOUT_SECONDS=60
LLM_FIRST_TOKEN_TIMEOUT_SECONDS=30
LLM_STREAM_IDLE_TIMEOUT_SECONDS=60

LLM_RETRY_ENABLED=true
LLM_RETRY_MAX_ATTEMPTS=3        # 总尝试次数
LLM_RETRY_BASE_DELAY_SECONDS=0.5
LLM_RETRY_MAX_DELAY_SECONDS=8

LLM_SDK_MAX_RETRIES=0           # 必须保持 0，避免双层 retry
STRUCTURED_LOGGING_ENABLED=true
LOG_LEVEL=INFO
```

## 8. Performance Benchmark

压测工具在 `evaluation/performance/`，与业务代码完全隔离，可重复运行：

```bash
python -m uvicorn evaluation.performance.fake_app:app --port 8010

python evaluation/performance/run_load_test.py \
  --base-url http://127.0.0.1:8010 \
  --llm-type fake \
  --mode ask stream \
  --concurrency 1 5 10 20 \
  --requests-per-level 20
```

详细方法、CLI 参数与完整结果见：

```text
evaluation/performance/README.md
evaluation/performance/results/
```

### 8.1 两类结果必须分开看

| | Application-level | Real-LLM |
| --- | --- | --- |
| LLM | FakeLLM（确定性，无网络） | Qwen / DashScope |
| 反映 | 应用层：路由 / 限流 / Redis / 检索 / 持久化 / SSE | 用户真实体感 |
| QPS 能否当线上指标 | **不能** | 才是真实值 |

### 8.2 Application-level（FakeLLM，n=100/档 × 5 轮中位数）

| mode | 并发 | QPS | P50 | P95 | P99 | TTFT P50 | TTFT P95 | 错误率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ask | 1 | 36.45 | 27.23 | 33.30 | 38.37 | – | – | 0% |
| ask | 5 | 189.13 | 23.41 | 37.89 | 54.82 | – | – | 0% |
| ask | 10 | 235.66 | 29.83 | 88.62 | 163.68 | – | – | 0% |
| ask | 20 | 191.78 | 43.24 | 270.04 | 414.06 | – | – | 0% |
| stream | 1 | 14.13 | 68.63 | 81.40 | 104.27 | 29.12 | 39.42 | 0% |
| stream | 5 | 79.21 | 58.91 | 79.68 | 97.94 | 26.04 | 37.41 | 0% |
| stream | 10 | 142.00 | 64.82 | 102.30 | 148.80 | 28.40 | 52.48 | 0% |
| stream | 20 | 178.18 | 77.58 | 177.77 | 333.92 | 34.12 | 79.13 | 0% |

单位 ms。其中 FakeLLM 自身注入约 40ms 模拟生成耗时，
因此 stream 的真实应用层开销约在 25~30ms 量级。

**不要用 P50 谈容量**：c=20 时 ask 的 P99 是 414ms、stream 是 334ms，
而 P50 只有 43 / 78 ms，差 5~9 倍。

取中位数而不是单轮，是因为本机单轮噪声太大——
n=20 时 5 轮峰谷差 19~35%，甚至出现过
「命中缓存比未命中还慢」这种违背因果的结果。
加大样本量也只能压到 19~27%（噪声主要来自机器负载漂移）。
方法见 `evaluation/performance/README.md` §5.5。

### 8.3 Real-LLM（qwen3.8-max-0902，n=10/档）

| mode | 并发 | QPS | P50 | P95 | TTFT P50 | TTFT P95 | 错误率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ask | 1 | 0.10 | 7672.74 | 23687.86 | – | – | 0% |
| ask | 5 | 0.41 | 7104.86 | 17415.61 | – | – | 0% |
| stream | 1 | 0.15 | 5120.04 | 12819.59 | 4247.94 | 8204.22 | 0% |
| stream | 5 | 0.54 | 5580.33 | 15055.61 | 5031.65 | 10189.36 | 0% |

样本量仅 10，P95 统计意义很弱，只能看量级。
端到端耗时主要由上游模型决定。

> 这组是修复前测的，未重测：修复省掉的约 2~3ms 本地检索，
> 相对 5000ms 量级的真实链路占比 <0.1%，远低于 n=10 的噪声。

### 8.4 Cache hit vs miss

| 并发 | 状态 | 检索 mean | E2E mean |
| ---: | --- | ---: | ---: |
| 1 | miss | 6.74 | 73.57 |
| 1 | hit | 3.93 | 70.14 |
| 10 | miss | 6.46 | 65.64 |
| 10 | hit | 3.56 | 60.53 |

必须分开表述：**检索延迟**改善 41.7%~44.9%，
而**端到端**只改善 4.7%~7.8%（个位数百分点），因为端到端里生成占大头。

> 这个"端到端收益"被修正过三次：n=20 单轮是 −17%（噪声）→
> 服务状态不对等时 −4.7% → 两服务全新对等目录后 −7.8%。
> 检索那组数字（−42%~−45%）三轮都稳定，端到端一直在 5%~8% 之间晃——
> **因此只写"个位数百分点"，不写精确值。**

### 8.5 修复重复检索的收益（A/B 实测）

压测中发现 `astream()` 里检索被执行了两次（第一次返回值被丢弃），
已修复。起两个服务交错测量（修复版 8010 / 未修复版 8011，各 5 轮，n=100/档）：

| mode | 并发 | 基线 P50 | 修复后 P50 | delta | 基线 TTFT P50 | 修复后 TTFT P50 | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ask | 1 | 26.07 | 27.23 | +4.4% | – | – | – |
| ask | 5 | 25.66 | 23.41 | −8.8% | – | – | – |
| ask | 10 | 31.05 | 29.83 | −3.9% | – | – | – |
| ask | 20 | 44.37 | 43.24 | −2.5% | – | – | – |
| stream | 1 | 72.23 | 68.63 | −5.0% | 32.07 | 29.12 | **−9.2%** |
| stream | 5 | 61.12 | 58.91 | −3.6% | 28.02 | 26.04 | **−7.1%** |
| stream | 10 | 63.07 | 64.82 | **+2.8%** | 30.03 | 28.40 | **−5.4%** |
| stream | 20 | 79.55 | 77.58 | −2.5% | 37.51 | 34.12 | **−9.0%** |

被直接影响的**检索耗时**信号最强：−24.2%~−42.7%，**4 组全是 5/5 轮为负**。

**不要写"端到端延迟降低 X%"。** `ask` 是天然对照组（修复只动流式路径），
它四档 delta 符号就不一致（+4.4% ~ −8.8%）；`stream` 端到端同样符号不一致
（c=10 甚至是 +2.8%），幅度全部落在本机噪声内（基线 5 轮峰谷差 19~27%）。

能站住的是两条：**检索耗时 −24%~−43%**、**TTFT −5%~−9%**。

修复的正确性由**计数**而非延迟证明——压测 app 有 `/_metrics` 端点，
发一个全新问题看计数器：未修复版 `miss+1` 且 `hit+1`（检索两次），
修复版 `miss+1`、`hit+0`（检索一次）。另有 22 项单测与进程内探针。

### 8.6 测试环境

Apple M1 Pro / 8 核 / 16 GB，macOS 26.2；Python 3.9.13（项目 `.venv`）；
Redis 真机 `127.0.0.1:6379`；数据库 SQLite 临时文件；客户端与服务端同机。
基线 load average 常驻 ~4，**微基准抖动大，结论一律取多轮中位数**。

---

## 9. 快速开始

### 9.1 创建环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

安装后端依赖：

```bash
pip install -r frontend/local_rag/requirements.txt
```

---

### 9.2 配置环境变量

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

### 9.3 构建前端

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

### 9.4 启动项目

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

## 10. API

| Method | Endpoint | 功能 |
| --- | --- | --- |
| POST | `/api/upload` | 上传文档并写入知识库 |
| POST | `/api/ask` | RAG / Chat 问答（一次性返回） |
| POST | `/api/ask/stream` | RAG / Chat 问答（SSE 流式，逐 token 下发） |
| GET | `/api/status` | 查询知识库与检索状态 |
| GET | `/api/history` | 查询指定会话历史 |
| DELETE | `/api/reset` | 清空知识库 |
| POST | `/api/clear_history` | 清空指定会话 |
| POST | `/api/webhook` | 自动化 / CI Webhook |

问答与流式接口都支持可选的 `X-Request-ID` 请求头，响应头 `X-Request-ID`
返回实际使用的 ID（流式还会在 `meta` 事件里带一份）。传了合法值就复用，
不传或格式不合法由服务端生成。

问答示例：

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-req-001" \
  -d '{
    "question":"这份文档主要讲了什么？",
    "session_id":"demo",
    "mode":"rag"
  }'
```

出错时（响应头尚未发出的场景）返回对应的状态码与分类后的错误体：

```json
{"request_id": "demo-req-001", "error_type": "LLM_TIMEOUT", "message": "上游调用超时"}
```

---

## 11. Benchmark

### 11.1 正式评测配置

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

## 12. Retrieval A/B 实验

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

## 13. 统计显著性检验

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

## 14. End-to-End RAG Benchmark

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

## 15. Latency

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

## 16. 为什么有两组 Retrieval 数字

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

## 17. 测试

Retrieval 算法测试：

```bash
python tests/test_retrieval.py
```

KnowledgeRetriever 测试：

```bash
python tests/test_knowledge_retriever.py
```

会话持久化测试：

```bash
python tests/test_database.py
```

Redis 缓存与限流测试：

```bash
python tests/test_redis.py
```

Async I/O + SSE 测试（帧格式 / 服务端到端 / HTTP 字节级）：

```bash
python tests/test_async_sse.py
```

可靠性与可观测性测试（request_id / 重试 / 超时 / 错误分类 / 结构化日志）：

```bash
python tests/test_reliability_observability.py
```

性能统计自检（分位数 / 空样本 / CSV 契约；不起服务、不调 LLM，CI 必跑）：

```bash
python tests/test_performance_stats.py
```

流式路径检索次数回归（22 项；不起服务，CI 必跑）：

```bash
python tests/test_orchestrator_retrieval_once.py
```

性能压测（**不在 CI 里跑真实 LLM**，只做脚本自检；真实 benchmark 手动运行）：

```bash
python -m uvicorn evaluation.performance.fake_app:app --port 8010
python evaluation/performance/run_load_test.py \
  --base-url http://127.0.0.1:8010 --llm-type fake \
  --concurrency 1 5 10 20 --requests-per-level 20
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
Conversation / Message 持久化
跨进程重启恢复
多会话隔离
最近 10 轮窗口
并发创建会话（唯一键冲突重试）
utf8mb4 四字节字符往返
超长答案（LONGTEXT）
Retrieval Cache Miss / Hit
不同 query / mode / top_k / rewrite 不串缓存
Cache TTL 到期重新 Miss
Knowledge Base Version bump 失效
upload / reset 后版本递增（失败不 bump）
Redis 故障后 RAG / Chat / MySQL 均正常
Rate Limit 放行 / 429 / 窗口滚动 / 会话隔离 / fail-open
Lua 瞬时故障后自动恢复（不被永久禁用）
SSE 帧边界 / data 单行 / 中文不转义 / 心跳是注释帧
事件顺序 meta → sources → trace → delta* → done
delta 拼接等于 done.answer
chat 模式不下发 sources
上游异常转成 error 事件（不中断连接）
正常完成：恰好写 1 次库、恰好 1 条完整 turn（无 per-token 写）
abort（CancelledError）/ GeneratorExit / 中途取消 / LLM streaming error
  均「不写库」，历史中不残留半截 assistant answer
生成结束后立刻取消：最终写入受 shield 保护仍然成功
Redis 故障下流式 fail-open
流式端点限流 429 + Retry-After
并发两路流不串台
流式期间事件循环未被阻塞
request_id 自动生成 / 响应头 / SSE meta / 复用客户端传入值
并发请求的 request_id 与上下文互不串台
结构化日志含 request_id，且不出现 API key / 完整 question / 完整 answer
非流式：瞬时错误重试后成功、达上限停止、鉴权与参数错误不重试、退避次数正确
流式：首 token 前瞬时错误重试成功、首 token 后失败不重试、error 只发一次
首 token 超时 / 流式空闲超时 / 非流式超时，超时后不留半截答案、无 dangling task
数据库写失败不重新调用 LLM、不重复写
Redis 故障仍 fail-open 且有 warning 日志、不变成 500
自己限流 429 + Retry-After，且不进入 LLM generation
total_ms / ttft_ms / generation_ms / retrieval_ms 定义正确
abort / aclose 清理：无跨 task CancelScope、无残留 task、不落库
流式路径检索恰好执行一次（chat 模式零次）
缓存命中归因 cache_hit 在 hit / miss / 无检索 / 指标不可用四种情形下的取值
分位数计算（已知答案钉死）、空样本不顶替、CSV 字段契约
```

此前测试结果（各自带 runner 的**断言项数**；pytest 收集到的是函数数，见下）：

```text
tests/test_retrieval.py             26 passed
tests/test_knowledge_retriever.py   38 passed
tests/test_database.py              51 passed  （SQLite 默认）
                                    56 passed  （连真实 MySQL 8.4）
tests/test_redis.py                 71 passed  （fakeredis 默认）
                                    80 passed  （连真实 Redis 7）
tests/test_async_sse.py             82 passed  （httpx + ASGITransport）
tests/test_reliability_observability.py  158 passed  （全部 deterministic stub）
tests/test_performance_stats.py     44 passed
tests/test_orchestrator_retrieval_once.py 22 passed
```

同一批文件用 `pytest -q` 跑出来的**函数数**是：
reliability 34、database 13、retrieval 5、redis 16、async_sse 17、
knowledge_retriever 6（后五项合计 57）。
两套数字口径不同，不要互相比较。

---

## 18. Docker

```bash
docker compose up --build
```

---

## 19. 项目结构

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
│       │   ├── routes.py
│       │   └── sse.py              # SSE 帧构造 / 心跳（零依赖手写）
│       ├── cache/
│       │   └── redis_client.py
│       ├── config/
│       ├── core/
│       │   ├── agents/
│       │   └── retrieval/
│       ├── db/
│       │   ├── database.py
│       │   └── models.py
│       ├── observability/                 # Stage 4
│       │   ├── request_context.py   # request_id 与上下文传播（ContextVar）
│       │   ├── errors.py            # 错误分类 / 是否可重试 / 状态码 / 脱敏
│       │   ├── structured_log.py    # 结构化事件日志与计时
│       │   ├── retry.py             # 指数退避重试（非流式）
│       │   └── stream_guard.py      # 流式超时 + 首 token 前重试 + 任务清理
│       ├── services/
│       │   ├── knowledge_service.py
│       │   ├── conversation_store.py
│       │   ├── retrieval_cache.py
│       │   └── rate_limiter.py
│       └── utils/
│
├── tests/
│   ├── test_retrieval.py
│   ├── test_knowledge_retriever.py
│   ├── test_database.py
│   ├── test_redis.py
│   ├── test_async_sse.py
│   ├── test_reliability_observability.py
│   ├── test_performance_stats.py           # Stage 5 压测脚本自检
│   └── test_orchestrator_retrieval_once.py # Stage 5 流式检索次数回归
│
└── evaluation/
    ├── EVALUATION_AUDIT.md
    ├── README.md
    ├── final_regression_report.md      # Stage 5 回归判定
    ├── eval_dataset_v3.jsonl
    ├── eval_dataset_oral_v3.jsonl
    ├── corpus/
    ├── extended_noise/
    ├── evaluate_retrieval_ab.py
    ├── significance_test.py
    ├── benchmark_runtime.py
    ├── evaluate_rag.py
    ├── performance/                    # Stage 5
    │   ├── fake_app.py            # 真实栈 + FakeLLM 的压测应用（含 /_metrics 计数器端点）
    │   ├── run_load_test.py       # 压测执行器（CLI 可配置）
    │   ├── make_charts.py         # 从 CSV 出图
    │   ├── aggregate_ab.py        # 配对 A/B：30 个原始 CSV -> 中位数表
    │   ├── probe_retrieval_count.py  # 进程内计数探针（证明检索次数）
    │   ├── README.md
    │   └── results/
    │       ├── performance_summary.csv
    │       ├── performance_requests.csv
    │       ├── real_summary.csv
    │       ├── cache_miss_summary.csv
    │       ├── cache_hit_summary.csv
    │       ├── ab_summary.csv           # 5 轮交错中位数（README 表格数据源）
    │       ├── ab_fix_vs_baseline/      # 30 个原始 CSV + 代码切换计数证据
    │       └── *.png
    └── results/
        ├── retrieval_final/
        ├── rag_final/
        ├── retrieval_stage5/            # Stage 5 重跑的 A/B
        ├── rag_stage5_retrieval_only/   # Stage 5 重跑的 100 题检索
        ├── rag_stage5_after_fix/        # 修复重复检索后的复跑（零回归）
        ├── rag_stage5_judge_subset/     # LLM Judge 20 题子集
        └── archive/
```

---

## 20. 正式结果

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

Stage 5 重跑（回归验证，与上面历史结果逐位相同）：

```text
evaluation/results/retrieval_stage5/            # Retrieval A/B，148 个数值单元零差异
evaluation/results/rag_stage5_retrieval_only/   # 100 题 RAG 检索
evaluation/results/rag_stage5_after_fix/        # 修复重复检索后的复跑，同样零差异
evaluation/results/rag_stage5_judge_subset/     # LLM Judge 20 题子集（仅链路验证）
evaluation/final_regression_report.md           # 判定依据
```

历史实验只用于记录项目迭代，不作为当前正式 Benchmark 结论。

完整评测设计与审计过程见：

```text
evaluation/EVALUATION_AUDIT.md
```

---

## 21. 实验结论与边界

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

## 22. Limitations

* **压测规模有限**：最高测到 20 并发、每档 100 个样本，且客户端与服务端同机。
  这不构成线上容量结论，不能据此宣称"高并发"或"数千 QPS"。
* **测量噪声不可忽略**：本机基线 load average 常驻 ~4。
  同一份代码 5 轮跑出的 P50 峰谷差，n=20 时 19~35%、n=100 时仍有 19~27%
  ——**加大样本量并不能解决**，主要噪声源是机器负载漂移而非样本量。
  所有结论均取 5 轮交错中位数，单轮数字不具可复现性
  （详见 `evaluation/performance/README.md` §5.5）。
* **A/B 只在 FakeLLM + SQLite 下做，且只对部分指标作数值声称**：
  修复收益只对**检索耗时（−24%~−43%）和 TTFT（−5%~−9%）**给出数字；
  **端到端延迟不声称任何数值**——它落在本机噪声内，连符号都不稳定。
  另外第一版 A/B 因两个服务的 `PERF_WORK_DIR` 状态不对等
  （一个已积累几千条会话历史）得出过假的单调结论，已撤回并重测
  （详见 `evaluation/performance/README.md` §5.6）。
* **Real-LLM 样本量小**：每档仅 10 次调用，P95/P99 只能看量级，
  且受上游网络波动影响明显（P50 7.1s、P95 17.4s 同一档位）。
* **数据库压测用的是 SQLite**，不是 Docker Compose 里的 MySQL。
  SQLite 的写锁行为与 MySQL 不同，持久化相关的延迟数字不能直接外推。
* **Real-LLM 数字是修复前测的**：本次修复只影响本地检索（约 2~3ms），
  相对 5000ms 量级的真实链路占比 <0.1%，未重测（详见 §8.3）。
  若换成低延迟模型（<500ms）必须重跑。
* **评测语料规模小**：正式 benchmark 为 44 篇文档 / 46 个 chunk。
  随机基线 Hit@3 已经达到 6.5%，指标天花板受限，
  结论只在这套语料下成立。
* **LLM Judge 未做全量重跑**：Stage 5 只重跑了 retrieval，
  历史 Judge 结果保留但未复现（详见 `evaluation/final_regression_report.md`）。
* **单机单进程**：没有多副本、没有负载均衡、没有连接池调优，
  也没有做长时间稳定性 / 内存增长测试。

## 23. Future Work

* 在 MySQL + 多 worker 下重跑压测，拿到更接近部署形态的基线。
  （流式路径的重复检索已修复，见 §8.5；`timing.cache_hit` 随之恢复正确。）
* 引入真正的 metrics 导出（Prometheus / OpenTelemetry），
  替换当前"结构化日志 + 进程内计数器"的做法。
* 扩大评测语料规模，降低随机基线、拉开指标区分度。
* Agent / LangGraph / MCP / Tool Calling：本项目刻意不做，
  留给下一个项目。
