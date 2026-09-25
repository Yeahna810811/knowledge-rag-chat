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
- **客户端断开**（关页面 / AbortController / 代理超时）时，
  已经生成出来的部分照常落库。关键是 `CancelScope(shield=True)`：
  当前作用域已被取消，不屏蔽的话这条写库 `await` 会立刻跟着被取消，等于白写。

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

## 7. 快速开始

### 7.1 创建环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

安装后端依赖：

```bash
pip install -r frontend/local_rag/requirements.txt
```

---

### 7.2 配置环境变量

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

### 7.3 构建前端

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

### 7.4 启动项目

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

## 8. API

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

## 9. Benchmark

### 9.1 正式评测配置

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

## 10. Retrieval A/B 实验

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

## 11. 统计显著性检验

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

## 12. End-to-End RAG Benchmark

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

## 13. Latency

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

## 14. 为什么有两组 Retrieval 数字

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

## 15. 测试

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
客户端断开后已生成部分落库（cancel 与 aclose 两条路径）
Redis 故障下流式 fail-open
流式端点限流 429 + Retry-After
并发两路流不串台
流式期间事件循环未被阻塞
```

此前测试结果：

```text
tests/test_retrieval.py             26 passed
tests/test_knowledge_retriever.py   38 passed
tests/test_database.py              51 passed  （SQLite 默认）
                                    56 passed  （连真实 MySQL 8.4）
tests/test_redis.py                 71 passed  （fakeredis 默认）
                                    80 passed  （连真实 Redis 7）
tests/test_async_sse.py             63 passed  （httpx + ASGITransport）
```

---

## 16. Docker

```bash
docker compose up --build
```

---

## 17. 项目结构

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
│   └── test_async_sse.py
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

## 18. 正式结果

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

## 19. 实验结论与边界

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