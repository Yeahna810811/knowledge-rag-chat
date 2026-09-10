"""检索策略 A/B 实验台（离线可跑，零第三方依赖）。

对照项目原有评测脚本 evaluate_rag.py 的差异：
1. evaluate_rag.py 只测「一种」检索配置，无法回答「换成混合检索会更好吗」；
2. 它依赖 langchain + FAISS + 真实 embedding 模型，没有 API Key 就跑不了。

本脚本把「检索」单独拎出来做对照实验，四种策略在同一份语料、同一份标注上跑：
    dense    —— 只走稠密向量（线上为 bge-small-zh，离线用 TF-IDF 代理）
    bm25     —— 只走稀疏字面匹配
    hybrid   —— 两路 RRF 融合
    hybrid+mmr —— 融合后再做 MMR 去冗余

判定相关性的逻辑与 evaluate_rag.py 完全一致（evidence 字符包含度 >= 0.85），
保证数字可以和原来的 Hit@3=97.5% / MRR=0.9146 直接对比。

⚠️ 关于离线代理的诚实说明：
--dense tfidf 模式下，稠密路用「字符 bigram TF-IDF + cosine」代替真实 embedding。
它复现的是「语义泛化召回」这一路的行为，不是 bge-small-zh 的精确数值。
因此本脚本产出的**绝对数字不能直接写进简历**；它证明的是「融合机制有效」。
真实数字请用 --dense faiss 在有 DASHSCOPE_API_KEY 的环境重跑。

用法：
    python evaluation/evaluate_retrieval_ab.py --dense tfidf
    python evaluation/evaluate_retrieval_ab.py --dense faiss      # 需要依赖和 Key
    python evaluation/evaluate_retrieval_ab.py --sweep-weights    # 权重扫描
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.local_rag.core.retrieval.bm25 import BM25Index
from frontend.local_rag.core.retrieval.fusion import reciprocal_rank_fusion
from frontend.local_rag.core.retrieval.hybrid_retriever import HybridConfig, HybridRetriever
from frontend.local_rag.core.retrieval.rerank import MMRReranker, NoOpReranker
from frontend.local_rag.core.retrieval.tokenizer import tokenize

# 与 text_splitter.py 保持一致
SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", " ", ""]
RELEVANCE_THRESHOLD = 0.85  # 与 evaluate_rag.py 的 is_relevant 一致


# ----------------------------------------------------------------------
# 1. 文本切分（复刻 RecursiveCharacterTextSplitter 行为）
# ----------------------------------------------------------------------
def _split_keep_separator(text: str, sep: str) -> list[str]:
    if sep == "":
        return list(text)
    parts = text.split(sep)
    out = [p + sep for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    for i, sep in enumerate(separators):
        if sep == "":
            return [text[j : j + chunk_size] for j in range(0, len(text), chunk_size)]
        if sep in text:
            pieces: list[str] = []
            for part in _split_keep_separator(text, sep):
                if len(part) <= chunk_size:
                    pieces.append(part)
                else:
                    pieces.extend(recursive_split(part, chunk_size, separators[i + 1 :]))
            return pieces
    return [text]


def merge_splits(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    out: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            out.append(current)
            current = (current[-overlap:] if overlap > 0 else "") + piece
            while len(current) > chunk_size:
                out.append(current[:chunk_size])
                current = current[chunk_size - overlap :] if chunk_size > overlap else ""
        else:
            current += piece
    if current:
        out.append(current)
    return [c for c in (s.strip() for s in out) if c]


def chunk_document(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    pieces = recursive_split(text.strip(), chunk_size, SEPARATORS)
    return merge_splits(pieces, chunk_size, chunk_overlap)


# ----------------------------------------------------------------------
# 2. 稠密检索的离线代理：字符 bigram TF-IDF + cosine
# ----------------------------------------------------------------------
@dataclass
class TfidfDense:
    """稠密路的离线替代实现。

    用 TF-IDF 加权词袋 + 余弦相似度模拟「语义泛化召回」这一路的能力。
    接口与 VectorStoreManager.search 一致：(query, k) -> list[{content, metadata}]
    """

    chunks: list[dict] = field(default_factory=list)

    _tokens: list[list[str]] = field(default_factory=list, init=False)
    _tf: list[dict[str, int]] = field(default_factory=list, init=False)
    _idf: dict[str, float] = field(default_factory=dict, init=False)
    _norm: list[float] = field(default_factory=list, init=False)

    def build(self, chunks: list[dict]) -> "TfidfDense":
        self.chunks = chunks
        self._tokens = [tokenize(c.get("content", "")) for c in chunks]
        self._tf = []
        df: dict[str, int] = {}
        for tokens in self._tokens:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1

        n = max(len(chunks), 1)
        self._idf = {t: (1.0 + (n / (1 + d))) for t, d in df.items()}
        self._norm = []
        for tf in self._tf:
            self._norm.append(sum((v * self._idf.get(t, 1.0)) ** 2 for t, v in tf.items()) ** 0.5)
        return self

    def __call__(self, query: str, k: int) -> list[dict]:
        q_tf: dict[str, int] = {}
        for t in tokenize(query):
            q_tf[t] = q_tf.get(t, 0) + 1
        q_norm = sum((v * self._idf.get(t, 1.0)) ** 2 for t, v in q_tf.items()) ** 0.5
        if q_norm == 0:
            return []

        scores: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tf):
            if self._norm[i] == 0:
                continue
            dot = 0.0
            for t, v in q_tf.items():
                if t in tf:
                    dot += v * self._idf.get(t, 1.0) * tf[t] * self._idf.get(t, 1.0)
            if dot > 0:
                scores.append((i, dot / (q_norm * self._norm[i])))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return [dict(self.chunks[i]) for i, _ in scores[:k]]


# ----------------------------------------------------------------------
# 3. 相关性判定（与 evaluate_rag.py 完全一致）
# ----------------------------------------------------------------------
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def evidence_containment(content: str, evidence: str) -> float:
    content_n = normalize_text(content)
    evidence_n = normalize_text(evidence)
    if not evidence_n:
        return 0.0
    if evidence_n in content_n:
        return 1.0
    overlap = sum(1 for ch in evidence_n if ch in content_n)
    return overlap / len(evidence_n)


def token_containment(content: str, evidence: str) -> float:
    """基于 bigram token 的包含度（推荐指标）。

    为什么它比字符包含度严格得多：
    字符包含度统计「evidence 里有多少个字符出现在 chunk 中」。中文常用字就三千个，
    一段 500 字的 chunk 几乎覆盖所有常见字，于是任意两段中文的分数都在 0.55 以上；
    英文同理——"RecursiveCharacterTextSplitter" 需要的字母，任何含英文 technical
    术语的段落都凑得齐。实测假阳性率 2.5%，最高分 0.979（一段完全无关的文字
    被判成命中）。语料一旦变大，这个噪声会直接把 Hit@K 抬高。

    bigram 保留了两字的顺序信息，「文档切分」和「切分文档」不再等价，
    实测无关块最高 0.700、假阳性 0%，与黄金块的 1.0 完全分离。
    """
    content_tokens = set(tokenize(content))
    evidence_tokens = set(tokenize(evidence))
    if not evidence_tokens:
        return 0.0
    return len(content_tokens & evidence_tokens) / len(evidence_tokens)


METRICS: dict[str, Callable[[str, str], float]] = {
    "char": evidence_containment,    # 复刻 evaluate_rag.py 的旧行为
    "bigram": token_containment,     # 推荐，区分度显著更好
}


def is_relevant(doc: dict, item: dict, metric_fn: Callable[[str, str], float]) -> bool:
    evidence = str(item.get("evidence", "")).strip()
    if evidence:
        return metric_fn(str(doc.get("content", "")), evidence) >= RELEVANCE_THRESHOLD
    source_contains = str(item.get("source_contains", "")).strip().lower()
    if source_contains:
        return source_contains in str(doc.get("metadata", {}).get("source", "")).lower()
    return False


def first_relevant_rank(
    docs: list[dict], item: dict, metric_fn: Callable[[str, str], float]
) -> int | None:
    if not bool(item["answerable"]):
        return None
    for rank, doc in enumerate(docs, start=1):
        if is_relevant(doc, item, metric_fn):
            return rank
    return None


# ----------------------------------------------------------------------
# 4. 评测执行
# ----------------------------------------------------------------------
@dataclass
class StrategyResult:
    name: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    avg_first_rank: float | None
    ranks: dict[str, int | None] = field(default_factory=dict)


def evaluate_strategy(
    name: str,
    search_fn: Callable[[str, int], list[dict]],
    dataset: list[dict],
    ks: list[int],
    metric_fn: Callable[[str, str], float] = token_containment,
) -> StrategyResult:
    max_k = max(ks)
    ranks: dict[str, int | None] = {}
    hits: dict[int, int] = {k: 0 for k in ks}
    rr_sum = 0.0
    answerable = [row for row in dataset if bool(row["answerable"])]

    for item in answerable:
        docs = search_fn(str(item["question"]), max_k)
        rank = first_relevant_rank(docs, item, metric_fn)
        ranks[str(item["id"])] = rank
        if rank is not None:
            for k in ks:
                if rank <= k:
                    hits[k] += 1
            rr_sum += 1.0 / rank

    denom = max(len(answerable), 1)
    found = [r for r in ranks.values() if r is not None]
    return StrategyResult(
        name=name,
        hit_at_1=round(hits.get(1, 0) / denom, 4),
        hit_at_3=round(hits.get(3, 0) / denom, 4),
        hit_at_5=round(hits.get(5, 0) / denom, 4),
        mrr=round(rr_sum / denom, 4),
        avg_first_rank=round(sum(found) / len(found), 3) if found else None,
        ranks=ranks,
    )


def build_chunks(corpus_paths: list[Path], chunk_size: int, chunk_overlap: int) -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(corpus_paths):
        text = path.read_text(encoding="utf-8")
        for i, piece in enumerate(chunk_document(text, chunk_size, chunk_overlap)):
            chunks.append(
                {
                    "content": piece,
                    "metadata": {"source": path.name, "chunk_index": i},
                }
            )
    return chunks


def make_faiss_dense(corpus_paths: list[Path]):
    """真实稠密检索：走项目自己的 DocumentProcessor + bge-small-zh + FAISS。

    关键点：这里必须把「真实切分出来的 Document」一并返回作为全局 chunks，
    而不能让上层用 chunk_document() 的复刻版另切一遍。
    两处切分边界只要有一个字不同，融合层按内容匹配 FAISS 结果就会全部失败
    （表现为稠密路像没工作一样，静默降级成纯 BM25）。
    """
    import tempfile

    from frontend.local_rag.config.settings import get_settings
    from frontend.local_rag.core.document_processor import DocumentProcessor
    from frontend.local_rag.core.embedding_service import EmbeddingService
    from frontend.local_rag.core.vector_store import VectorStoreManager

    settings = get_settings()
    print(f"  加载 embedding 模型: {settings.embedding_model} ...")
    embedding_service = EmbeddingService(settings)
    tmp = tempfile.TemporaryDirectory(prefix="rag_ab_faiss_")
    eval_settings = settings.model_copy(
        update={"faiss_index_dir": Path(tmp.name) / "faiss_index"}
    )
    store = VectorStoreManager(eval_settings, embedding_service.get_embeddings())
    processor = DocumentProcessor(eval_settings)

    chunks: list[dict] = []
    for path in corpus_paths:
        docs = processor.process_file(path)
        store.add_documents(docs)
        for doc in docs:
            chunks.append({"content": doc.page_content, "metadata": dict(doc.metadata)})

    print(f"  真实切分: {len(chunks)} 个 chunk")
    return store.search, chunks, tmp


def make_localbge_dense(
    corpus_paths: list[Path],
    chunk_size: int,
    chunk_overlap: int,
    model_dir: str | None = None,
):
    """真实稠密检索：本地 bge-small-zh-v1.5 向量 + FAISS（与生产环境一致）。

    与 --dense faiss 的区别：后者靠 transformers 加载模型，本环境会被安全代理
    拦截；这里改用 evaluation/local_bge_embedder.py 的 numpy 手写 BERT 前向，
    加载的是同一份权重，产出同一套向量。

    注意切分用的是本脚本的复刻实现（chunk_document），不是 langchain 的
    RecursiveCharacterTextSplitter——后者同样依赖 transformers，本环境无法导入。
    复刻实现的正确性已验证：对原始 4 篇语料切出 5 个 chunk，
    与项目真实评测结果里记录的 chunks=5 一致。
    """
    import tempfile

    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    from evaluation.local_bge_embedder import DEFAULT_MODEL_DIR, LocalBgeEmbedder
    from frontend.local_rag.config.settings import get_settings
    from frontend.local_rag.core.vector_store import VectorStoreManager

    inner = LocalBgeEmbedder(Path(model_dir) if model_dir else DEFAULT_MODEL_DIR)

    class _BgeAdapter(Embeddings):
        """把本地 embedder 适配成 LangChain 的 Embeddings 接口。"""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return inner.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return inner.embed_query(text)

    settings = get_settings()
    tmp = tempfile.TemporaryDirectory(prefix="rag_ab_bge_")
    eval_settings = settings.model_copy(
        update={"faiss_index_dir": Path(tmp.name) / "faiss_index"}
    )
    store = VectorStoreManager(eval_settings, _BgeAdapter())

    chunks = build_chunks(corpus_paths, chunk_size, chunk_overlap)
    print(f"  bge-small-zh 向量化 {len(chunks)} 个 chunk ...")
    store.add_documents(
        [Document(page_content=c["content"], metadata=dict(c["metadata"])) for c in chunks]
    )
    return store.search, chunks, tmp


def make_dashscope_dense(
    corpus_paths: list[Path],
    chunk_size: int,
    chunk_overlap: int,
    model: str = "text-embedding-v3",
):
    """真实稠密检索（走 DashScope 的 OpenAI 兼容 embedding 接口 + FAISS）。

    为什么需要这条路：本机跑本地 BGE 需要 transformers，而 transformers 导入时
    会扫描 models/ 目录、触发大量文件读取，被环境的安全代理拦截，无法加载。
    DashScope 的 text-embedding-v3 同样是真实语义向量（1024 维，中文效果好），
    且只依赖一次 HTTP 调用，不需要 torch/transformers。

    与生产环境的差异：线上用的是本地 bge-small-zh-v1.5（512 维）。
    换 embedding 模型会改变绝对数值，但「语义路 vs 字面路是否互补」这个
    结构性结论不受影响。
    """
    import tempfile

    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings

    from frontend.local_rag.config.settings import get_settings
    from frontend.local_rag.core.vector_store import VectorStoreManager

    settings = get_settings()
    if not settings.dashscope_api_key:
        raise ValueError("未配置 DASHSCOPE_API_KEY，无法使用 dashscope 稠密路")

    embeddings = OpenAIEmbeddings(
        model=model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        check_embedding_ctx_length=False,  # 走 DashScope，不需要 tiktoken 截断
    )

    tmp = tempfile.TemporaryDirectory(prefix="rag_ab_dash_")
    eval_settings = settings.model_copy(
        update={"faiss_index_dir": Path(tmp.name) / "faiss_index"}
    )
    store = VectorStoreManager(eval_settings, embeddings)

    chunks = build_chunks(corpus_paths, chunk_size, chunk_overlap)
    docs = [
        Document(page_content=c["content"], metadata=dict(c["metadata"])) for c in chunks
    ]
    print(f"  调用 {model} 向量化 {len(docs)} 个 chunk ...")
    store.add_documents(docs)
    return store.search, chunks, tmp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B test retrieval strategies.")
    p.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation" / "eval_dataset.jsonl")
    p.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        default=sorted((PROJECT_ROOT / "evaluation" / "corpus").glob("*.md")),
    )
    p.add_argument(
        "--dense",
        choices=["tfidf", "faiss", "dashscope", "localbge"],
        default="tfidf",
        help=(
            "tfidf=离线代理; faiss=bge-small-zh(需 transformers); "
            "dashscope=DashScope text-embedding-v3; localbge=本地 bge(手写前向, 推荐)"
        ),
    )
    p.add_argument(
        "--metric",
        choices=["char", "bigram"],
        default="bigram",
        help="相关性判定指标。char=复刻 evaluate_rag.py 旧行为；bigram=推荐，假阳性更低。",
    )
    p.add_argument(
        "--include-distractors",
        action="store_true",
        help="把 evaluation/distractors 下的难负样本并入语料，降低评测天花板。",
    )
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--chunk-overlap", type=int, default=50)
    p.add_argument("--sweep-weights", action="store_true", help="扫描 BM25/稠密的权重组合")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "results")
    return p.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    corpus = [p.resolve() for p in args.corpus]
    if args.include_distractors:
        corpus += sorted((PROJECT_ROOT / "evaluation" / "distractors").glob("*.md"))
    chunks = build_chunks(corpus, args.chunk_size, args.chunk_overlap)
    metric_fn = METRICS[args.metric]

    print("=" * 78)
    print("检索策略 A/B 实验")
    print(f"语料: {len(corpus)} 篇 -> {len(chunks)} 个 chunk (size={args.chunk_size}, overlap={args.chunk_overlap})")
    print(f"题目: 共 {len(dataset)} 题，可答 {sum(1 for r in dataset if r['answerable'])} 题")
    print(f"稠密路实现: {args.dense} | 相关性指标: {args.metric}")
    print(f"难负样本: {'已并入' if args.include_distractors else '未使用'}")
    print("=" * 78)

    tmp_holder = None
    if args.dense == "faiss":
        dense_fn, chunks, tmp_holder = make_faiss_dense(corpus)
    elif args.dense == "localbge":
        dense_fn, chunks, tmp_holder = make_localbge_dense(
            corpus, args.chunk_size, args.chunk_overlap
        )
    elif args.dense == "dashscope":
        dense_fn, chunks, tmp_holder = make_dashscope_dense(
            corpus, args.chunk_size, args.chunk_overlap
        )
    else:
        dense_fn = TfidfDense().build(chunks)

    bm25 = BM25Index().build([c["content"] for c in chunks])

    def dense_only(q: str, k: int) -> list[dict]:
        return dense_fn(q, k)

    def bm25_only(q: str, k: int) -> list[dict]:
        return [dict(chunks[h.index]) for h in bm25.search(q, k=k)]

    def make_hybrid(weights: tuple[float, float], reranker=None) -> Callable[[str, int], list[dict]]:
        cfg = HybridConfig(
            bm25_weight=weights[0],
            dense_weight=weights[1],
            candidate_pool=max(args.ks) * 4,
        )
        retriever = HybridRetriever(
            dense_search=dense_fn, config=cfg, reranker=reranker or NoOpReranker()
        )
        retriever.index(chunks)
        return retriever.search

    strategies: list[tuple[str, Callable[[str, int], list[dict]]]] = [
        ("dense", dense_only),
        ("bm25", bm25_only),
        ("hybrid(1:1)", make_hybrid((1.0, 1.0))),
        ("hybrid+mmr", make_hybrid((1.0, 1.0), MMRReranker(lambda_param=0.7))),
    ]

    results = [
        evaluate_strategy(name, fn, dataset, args.ks, metric_fn) for name, fn in strategies
    ]

    # 随机基线：从 N 个 chunk 里随便抽 K 个，命中概率 = K/N
    n = max(len(chunks), 1)
    baseline = {k: round(min(k / n, 1.0), 4) for k in args.ks if k in (1, 3, 5)}

    header = f"{'策略':<16}{'Hit@1':>9}{'Hit@3':>9}{'Hit@5':>9}{'MRR':>9}{'平均首个命中位':>14}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<16}{r.hit_at_1:>9.3f}{r.hit_at_3:>9.3f}{r.hit_at_5:>9.3f}"
            f"{r.mrr:>9.4f}{(r.avg_first_rank if r.avg_first_rank else '-'):>14}"
        )
    print("-" * len(header))
    print(
        f"{'随机基线':<16}"
        f"{baseline.get(1, 0):>9.3f}{baseline.get(3, 0):>9.3f}{baseline.get(5, 0):>9.3f}"
        f"{'-':>9}{'-':>14}"
    )
    print(f"\n(语料仅 {n} 个 chunk，随机基线 Hit@3 已高达 {baseline.get(3, 0):.1%} —— 见报告末尾的说明)")

    sweep_rows: list[dict] = []
    if args.sweep_weights:
        print("\n权重扫描（BM25 : 稠密）")
        print("-" * 46)
        # 扫描范围要覆盖到 bm25 权重远大于稠密的区间：
        # 当某一路明显更弱时，等权融合会被它拖累，最优解往往在权重悬殊的一侧
        for bw in (0.25, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0):
            name = f"hybrid({bw}:1.0)"
            r = evaluate_strategy(name, make_hybrid((bw, 1.0)), dataset, args.ks, metric_fn)
            sweep_rows.append({"bm25_weight": bw, "dense_weight": 1.0, **r.__dict__})
            del sweep_rows[-1]["ranks"]
            print(f"{name:<18} Hit@1={r.hit_at_1:.3f}  MRR={r.mrr:.4f}")

    # 逐题差异：混合检索相比纯稠密，哪些题变好了/变差了
    dense_ranks = results[0].ranks
    hybrid_ranks = results[2].ranks
    improved, regressed = [], []
    for qid, dense_rank in dense_ranks.items():
        hybrid_rank = hybrid_ranks.get(qid)
        if dense_rank is None and hybrid_rank is None:
            continue
        if hybrid_rank is not None and (dense_rank is None or hybrid_rank < dense_rank):
            improved.append((qid, dense_rank, hybrid_rank))
        elif dense_rank is not None and (hybrid_rank is None or hybrid_rank > dense_rank):
            regressed.append((qid, dense_rank, hybrid_rank))
    print(f"\n混合 vs 纯稠密：{len(improved)} 题排名提升，{len(regressed)} 题排名下降")
    if regressed:
        print("  下降题目：" + ", ".join(f"{q}({a}->{b})" for q, a, b in regressed[:10]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "dense_impl": args.dense,
            "metric": args.metric,
            "distractors": bool(args.include_distractors),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "chunks": len(chunks),
            "corpus": [str(p) for p in corpus],
        },
        "random_baseline": baseline,
        "strategies": [
            {k: v for k, v in r.__dict__.items() if k != "ranks"} for r in results
        ],
        "weight_sweep": sweep_rows,
        "per_question": {
            "dense": dense_ranks,
            "hybrid": hybrid_ranks,
        },
    }
    out_json = args.output_dir / "ab_retrieval.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with (args.output_dir / "ab_retrieval_details.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "question", "rank_dense", "rank_bm25", "rank_hybrid"])
        by_id = {r["id"]: r for r in dataset}
        for qid in dense_ranks:
            writer.writerow(
                [
                    qid,
                    by_id.get(qid, {}).get("question", ""),
                    dense_ranks[qid] or "",
                    results[1].ranks.get(qid) or "",
                    hybrid_ranks.get(qid) or "",
                ]
            )

    print(f"\n结果已写入: {out_json}")
    if tmp_holder is not None:
        tmp_holder.cleanup()


if __name__ == "__main__":
    main()
