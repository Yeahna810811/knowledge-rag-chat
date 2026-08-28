"""Reproducible RAG evaluation for the knowledge-rag-chat project.

Metrics:
- Retrieval: Hit@K and MRR based on manually labeled evidence/source.
- Answering: heuristic correctness/refusal accuracy.
- Optional LLM-as-judge: correctness, faithfulness, relevance, safe refusal.
- Latency: retrieval and end-to-end latency (average, P50, P95).

The benchmark uses an isolated temporary FAISS index, so it does NOT modify the
application's normal knowledge-base index.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI

from frontend.local_rag.config.settings import get_settings
from frontend.local_rag.core.document_processor import DocumentProcessor
from frontend.local_rag.core.embedding_service import EmbeddingService
from frontend.local_rag.core.rag_chain import RAGChain
from frontend.local_rag.core.vector_store import VectorStoreManager

REFUSAL_PATTERNS = (
    "无法确定",
    "无法从",
    "无法根据",
    "没有相关",
    "未提供",
    "未提及",
    "没有足够",
    "无法得知",
    "不知道",
    "不清楚",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality for RAG.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "eval_dataset.jsonl",
        help="JSONL benchmark with question/reference/evidence labels.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        default=[PROJECT_ROOT / "frontend" / "local_rag" / "sample.txt"],
        help="One or more benchmark documents. The live FAISS index is not touched.",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Use Qwen/DashScope as an LLM judge for answer quality.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional judge model name. Defaults to CHAT_MODEL from .env.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip answer generation and only evaluate retrieval.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

            for required in ("id", "question", "reference_answer", "answerable"):
                if required not in item:
                    raise ValueError(f"Line {line_no} is missing field: {required}")
            item.setdefault("evidence", "")
            item.setdefault("source_contains", "")
            rows.append(item)

    if not rows:
        raise ValueError("Evaluation dataset is empty.")
    return rows


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def evidence_containment(content: str, evidence: str) -> float:
    """Return how much of the labeled evidence appears in one retrieved chunk.

    Exact normalized containment scores 1.0. Otherwise use a conservative
    character-overlap ratio. This avoids calling semantic similarity a ground-truth
    relevance label.
    """
    content_n = normalize_text(content)
    evidence_n = normalize_text(evidence)
    if not evidence_n:
        return 0.0
    if evidence_n in content_n:
        return 1.0

    evidence_chars = set(evidence_n)
    if not evidence_chars:
        return 0.0
    overlap = sum(1 for ch in evidence_n if ch in content_n)
    return overlap / len(evidence_n)


def is_relevant(doc: dict[str, Any], item: dict[str, Any]) -> bool:
    evidence = str(item.get("evidence", "")).strip()
    source_contains = str(item.get("source_contains", "")).strip().lower()

    if evidence:
        if evidence_containment(str(doc.get("content", "")), evidence) >= 0.85:
            return True

    if not evidence and source_contains:
        source = str(doc.get("metadata", {}).get("source", "")).lower()
        return source_contains in source

    return False


def first_relevant_rank(docs: list[dict[str, Any]], item: dict[str, Any]) -> int | None:
    if not bool(item["answerable"]):
        return None
    for rank, doc in enumerate(docs, start=1):
        if is_relevant(doc, item):
            return rank
    return None


def basic_answer_correct(answer: str, reference_answer: str) -> bool:
    ref = normalize_text(reference_answer)
    pred = normalize_text(answer)
    if not ref:
        return False
    if ref in pred:
        return True

    # Many short factual labels (e.g. "FAISS", "12天") appear inside a longer
    # reference sentence; extract compact alphanumeric/CJK spans and check them.
    tokens = re.findall(r"[a-z0-9.+#_-]+|[\u4e00-\u9fff]{2,}", ref, flags=re.IGNORECASE)
    meaningful = [t for t in tokens if len(t) >= 2]
    return bool(meaningful) and all(t.lower() in pred for t in meaningful[:3])


def looks_like_safe_refusal(answer: str) -> bool:
    return any(pattern in answer for pattern in REFUSAL_PATTERNS)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "avg_ms": round(sum(values) / len(values), 2),
        "p50_ms": round(percentile(values, 0.50) or 0.0, 2),
        "p95_ms": round(percentile(values, 0.95) or 0.0, 2),
    }


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge did not return JSON: {text[:200]}")
    return json.loads(cleaned[start : end + 1])


def clip_score(value: Any) -> float:
    score = float(value)
    return max(0.0, min(1.0, score))


def judge_answer(
    llm: ChatOpenAI,
    *,
    question: str,
    reference_answer: str,
    answerable: bool,
    retrieved_context: str,
    generated_answer: str,
) -> dict[str, float]:
    prompt = f"""You are evaluating a RAG answer. Return ONLY valid JSON.

Score every field from 0.0 to 1.0.
- correctness: matches the reference answer when answerable; for unanswerable questions,
  give 1 only when the answer appropriately says the information is unavailable.
- faithfulness: every factual claim is supported by the retrieved context. Unsupported
  claims reduce the score.
- relevance: directly answers the user's question without unnecessary content.
- safe_refusal: for answerable=true use 1.0; for answerable=false give 1 only when the
  model does not invent an answer and clearly indicates insufficient knowledge-base evidence.

Question: {question}
Answerable: {answerable}
Reference answer: {reference_answer}
Retrieved context:
{retrieved_context}

Generated answer:
{generated_answer}

JSON schema:
{{"correctness": 0.0, "faithfulness": 0.0, "relevance": 0.0, "safe_refusal": 0.0}}
"""
    response = llm.invoke([{"role": "user", "content": prompt}])
    raw = parse_json_object(str(response.content))
    return {
        "correctness": clip_score(raw.get("correctness", 0)),
        "faithfulness": clip_score(raw.get("faithfulness", 0)),
        "relevance": clip_score(raw.get("relevance", 0)),
        "safe_refusal": clip_score(raw.get("safe_refusal", 0)),
    }


def build_isolated_store(corpus: list[Path]):
    settings = get_settings()
    embedding_service = EmbeddingService(settings)

    temp_dir = tempfile.TemporaryDirectory(prefix="rag_eval_faiss_")
    eval_settings = settings.model_copy(
        update={"faiss_index_dir": Path(temp_dir.name) / "faiss_index"}
    )
    vector_store = VectorStoreManager(eval_settings, embedding_service.get_embeddings())
    processor = DocumentProcessor(eval_settings)

    total_chunks = 0
    for path in corpus:
        path = path.expanduser().resolve()
        if not path.exists():
            temp_dir.cleanup()
            raise FileNotFoundError(f"Corpus file not found: {path}")
        docs = processor.process_file(path)
        total_chunks += vector_store.add_documents(docs)

    return eval_settings, vector_store, total_chunks, temp_dir


def main() -> None:
    args = parse_args()
    if any(k <= 0 for k in args.ks):
        raise ValueError("All K values must be positive integers.")

    dataset = load_dataset(args.dataset.expanduser().resolve())
    if args.limit is not None:
        dataset = dataset[: args.limit]

    corpus = [p.expanduser().resolve() for p in args.corpus]
    settings, vector_store, total_chunks, temp_dir = build_isolated_store(corpus)

    rag_chain: RAGChain | None = None
    judge_llm: ChatOpenAI | None = None
    if not args.retrieval_only:
        rag_chain = RAGChain(settings, vector_store)
    if args.judge:
        if not settings.dashscope_api_key:
            temp_dir.cleanup()
            raise ValueError("--judge requires DASHSCOPE_API_KEY in frontend/local_rag/.env")
        judge_llm = ChatOpenAI(
            model=args.judge_model or settings.chat_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            temperature=0,
        )

    max_k = max(args.ks)
    answerable_rows = [row for row in dataset if bool(row["answerable"])]
    details: list[dict[str, Any]] = []
    retrieval_latencies: list[float] = []
    total_latencies: list[float] = []

    print("=" * 72)
    print("RAG benchmark")
    print(f"Dataset: {args.dataset}")
    print(f"Corpus files: {len(corpus)} | chunks: {total_chunks}")
    print(f"Questions: {len(dataset)} | answerable: {len(answerable_rows)}")
    print(f"K values: {args.ks} | judge: {args.judge}")
    print("=" * 72)

    for idx, item in enumerate(dataset, start=1):
        question = str(item["question"])

        retrieval_start = time.perf_counter()
        docs = vector_store.search(question, k=max_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        retrieval_latencies.append(retrieval_ms)

        rank = first_relevant_rank(docs, item)
        row: dict[str, Any] = {
            "id": item["id"],
            "question": question,
            "answerable": bool(item["answerable"]),
            "reference_answer": item["reference_answer"],
            "source_contains": item.get("source_contains", ""),
            "first_relevant_rank": rank or "",
            "reciprocal_rank": round(1 / rank, 6) if rank else 0.0,
            "retrieval_ms": round(retrieval_ms, 2),
            "retrieved_sources": " | ".join(
                str(d.get("metadata", {}).get("source", "")) for d in docs
            ),
        }
        for k in args.ks:
            row[f"hit@{k}"] = int(rank is not None and rank <= k)

        if not args.retrieval_only and rag_chain is not None:
            total_start = time.perf_counter()
            result = rag_chain.run(question, history=[])
            total_ms = (time.perf_counter() - total_start) * 1000
            total_latencies.append(total_ms)

            answer = str(result["answer"])
            answer_sources = result.get("sources", [])
            retrieved_context = "\n\n".join(str(d.get("content", "")) for d in answer_sources)

            row["generated_answer"] = answer
            row["total_ms"] = round(total_ms, 2)
            if bool(item["answerable"]):
                row["basic_correct"] = int(
                    basic_answer_correct(answer, str(item["reference_answer"]))
                )
                row["safe_refusal"] = ""
            else:
                row["basic_correct"] = ""
                row["safe_refusal"] = int(looks_like_safe_refusal(answer))

            if judge_llm is not None:
                try:
                    scores = judge_answer(
                        judge_llm,
                        question=question,
                        reference_answer=str(item["reference_answer"]),
                        answerable=bool(item["answerable"]),
                        retrieved_context=retrieved_context,
                        generated_answer=answer,
                    )
                    row.update({f"judge_{k}": round(v, 4) for k, v in scores.items()})
                except Exception as exc:  # Keep the benchmark running if one judge call fails.
                    row["judge_error"] = f"{type(exc).__name__}: {exc}"

        details.append(row)
        print(
            f"[{idx:02d}/{len(dataset):02d}] {item['id']} "
            f"rank={rank or '-'} retrieval={retrieval_ms:.1f} ms"
        )

    summary: dict[str, Any] = {
        "dataset": str(args.dataset.expanduser().resolve()),
        "corpus": [str(p) for p in corpus],
        "questions": len(dataset),
        "answerable_questions": len(answerable_rows),
        "unanswerable_questions": len(dataset) - len(answerable_rows),
        "chunks": total_chunks,
        "retrieval": {},
        "latency": {
            "retrieval": latency_summary(retrieval_latencies),
            "end_to_end": latency_summary(total_latencies),
        },
    }

    denom = max(len(answerable_rows), 1)
    for k in args.ks:
        summary["retrieval"][f"hit@{k}"] = round(
            sum(int(row[f"hit@{k}"]) for row in details if row["answerable"]) / denom,
            4,
        )
    summary["retrieval"]["mrr"] = round(
        sum(float(row["reciprocal_rank"]) for row in details if row["answerable"]) / denom,
        4,
    )

    if not args.retrieval_only:
        answerable_scored = [
            row for row in details if row["answerable"] and row.get("basic_correct") != ""
        ]
        unanswerable_scored = [
            row for row in details if not row["answerable"] and row.get("safe_refusal") != ""
        ]
        summary["answering"] = {
            "basic_correctness": round(
                sum(int(row["basic_correct"]) for row in answerable_scored)
                / max(len(answerable_scored), 1),
                4,
            ),
            "refusal_accuracy": round(
                sum(int(row["safe_refusal"]) for row in unanswerable_scored)
                / max(len(unanswerable_scored), 1),
                4,
            )
            if unanswerable_scored
            else None,
        }

    if args.judge:
        judged_rows = [row for row in details if "judge_correctness" in row]
        if judged_rows:
            summary["llm_judge"] = {
                metric: round(
                    sum(float(row[f"judge_{metric}"]) for row in judged_rows)
                    / len(judged_rows),
                    4,
                )
                for metric in ("correctness", "faithfulness", "relevance", "safe_refusal")
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "evaluation_details.csv"
    summary_path = args.output_dir / "evaluation_summary.json"

    fieldnames: list[str] = []
    for row in details:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with details_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 72)
    print(f"Details: {details_path}")
    print(f"Summary: {summary_path}")

    temp_dir.cleanup()


if __name__ == "__main__":
    main()
