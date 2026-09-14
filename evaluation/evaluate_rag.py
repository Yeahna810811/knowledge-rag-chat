"""Final RAG evaluation for knowledge-rag-chat.

Evaluation:
1. Retrieval:
   - Hit@1 / Hit@3 / Hit@5
   - MRR

2. Answer quality:
   - Rule-based correctness
   - Rule-based refusal accuracy

3. Optional LLM-as-a-Judge:
   - correctness
   - faithfulness
   - relevance
   - safe refusal

4. Latency:
   - retrieval latency
   - end-to-end RAG latency

Supported retrievers:
- dense: BGE + FAISS
- bm25: BM25 sparse retrieval

The benchmark builds an isolated temporary index and does NOT modify
the application's normal knowledge-base index.
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

from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from evaluation.evaluate_retrieval_ab import (
    first_relevant_rank as official_first_relevant_rank,
    token_containment,
)

from frontend.local_rag.config.settings import get_settings
from frontend.local_rag.core.document_processor import DocumentProcessor
from frontend.local_rag.core.embedding_service import EmbeddingService
from frontend.local_rag.core.rag_chain import RAGChain
from frontend.local_rag.core.vector_store import VectorStoreManager
from frontend.local_rag.core.retrieval.lexical_store import LexicalStore


# ---------------------------------------------------------------------
# Refusal patterns
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and final answer quality for RAG."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "eval_dataset_v3.jsonl",
        help="Evaluation JSONL. Default: evaluation/eval_dataset_v3.jsonl",
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Benchmark documents. "
            "Default: evaluation/corpus + evaluation/extended_noise"
        ),
    )

    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25"],
        default="bm25",
        help=(
            "Retrieval backend. "
            "dense=BGE+FAISS; bm25=BM25 sparse retrieval. "
            "Default: bm25"
        ),
    )

    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="K values for Hit@K. Default: 1 3 5",
    )

    parser.add_argument(
        "--judge",
        action="store_true",
        help="Use Qwen/DashScope as LLM-as-a-Judge.",
    )

    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model. Defaults to CHAT_MODEL in settings.",
    )

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only evaluate retrieval. Skip answer generation.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N questions.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. "
            "Default: evaluation/results/rag_<retriever>"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_no}: {exc}"
                ) from exc

            required_fields = (
                "id",
                "question",
                "reference_answer",
                "answerable",
            )

            for required in required_fields:
                if required not in item:
                    raise ValueError(
                        f"Line {line_no} is missing field: {required}"
                    )

            item.setdefault("evidence", "")
            item.setdefault("source_contains", "")

            rows.append(item)

    if not rows:
        raise ValueError("Evaluation dataset is empty.")

    return rows


# ---------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return re.sub(
        r"\s+",
        "",
        str(text),
    ).lower()


# ---------------------------------------------------------------------
# Retrieval relevance
# ---------------------------------------------------------------------

def first_relevant_rank(
    docs: list[dict[str, Any]],
    item: dict[str, Any],
) -> int | None:
    """Use the same relevance metric as evaluate_retrieval_ab.py."""

    return official_first_relevant_rank(
        docs,
        item,
        token_containment,
    )


# ---------------------------------------------------------------------
# Rule-based answer evaluation
# ---------------------------------------------------------------------

def basic_answer_correct(
    answer: str,
    reference_answer: str,
) -> bool:
    """Simple rule-based answer correctness.

    This is NOT semantic LLM evaluation.
    """

    ref = normalize_text(reference_answer)
    pred = normalize_text(answer)

    if not ref:
        return False

    if ref in pred:
        return True

    tokens = re.findall(
        r"[a-z0-9.+#_-]+|[\u4e00-\u9fff]{2,}",
        ref,
        flags=re.IGNORECASE,
    )

    meaningful = [
        token
        for token in tokens
        if len(token) >= 2
    ]

    return (
        bool(meaningful)
        and all(
            token.lower() in pred
            for token in meaningful[:3]
        )
    )


def looks_like_safe_refusal(answer: str) -> bool:
    return any(
        pattern in answer
        for pattern in REFUSAL_PATTERNS
    )


# ---------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------

def percentile(
    values: list[float],
    p: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    pos = (len(ordered) - 1) * p

    lo = int(pos)
    hi = min(
        lo + 1,
        len(ordered) - 1,
    )

    frac = pos - lo

    return (
        ordered[lo] * (1 - frac)
        + ordered[hi] * frac
    )


def latency_summary(
    values: list[float],
) -> dict[str, float | None]:
    if not values:
        return {
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
        }

    return {
        "avg_ms": round(
            sum(values) / len(values),
            2,
        ),
        "p50_ms": round(
            percentile(values, 0.50) or 0.0,
            2,
        ),
        "p95_ms": round(
            percentile(values, 0.95) or 0.0,
            2,
        ),
    }


# ---------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------

def parse_json_object(
    text: str,
) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        raise ValueError(
            f"Judge did not return JSON: {text[:200]}"
        )

    return json.loads(
        cleaned[start : end + 1]
    )


def clip_score(value: Any) -> float:
    score = float(value)

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


# ---------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------

def judge_answer(
    llm: ChatOpenAI,
    *,
    question: str,
    reference_answer: str,
    answerable: bool,
    retrieved_context: str,
    generated_answer: str,
) -> dict[str, float]:

    prompt = f"""You are evaluating a RAG answer.

Return ONLY valid JSON.

Score every field from 0.0 to 1.0.

correctness:
- For answerable questions, judge whether the generated answer matches
  the reference answer.
- For unanswerable questions, give 1.0 only when the answer correctly
  states that the required information is unavailable.

faithfulness:
- Every factual claim should be supported by the retrieved context.
- Unsupported claims reduce the score.

relevance:
- The answer should directly address the user's question.
- Unnecessary or unrelated content reduces the score.

safe_refusal:
- For answerable=true, return 1.0.
- For answerable=false, return 1.0 only when the model does not invent
  an answer and clearly states that the knowledge base lacks sufficient
  information.

Question:
{question}

Answerable:
{answerable}

Reference answer:
{reference_answer}

Retrieved context:
{retrieved_context}

Generated answer:
{generated_answer}

Return JSON exactly in this structure:

{{
  "correctness": 0.0,
  "faithfulness": 0.0,
  "relevance": 0.0,
  "safe_refusal": 0.0
}}
"""

    response = llm.invoke(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    raw = parse_json_object(
        str(response.content)
    )

    return {
        "correctness": clip_score(
            raw.get(
                "correctness",
                0,
            )
        ),
        "faithfulness": clip_score(
            raw.get(
                "faithfulness",
                0,
            )
        ),
        "relevance": clip_score(
            raw.get(
                "relevance",
                0,
            )
        ),
        "safe_refusal": clip_score(
            raw.get(
                "safe_refusal",
                0,
            )
        ),
    }


# ---------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------

def default_corpus() -> list[Path]:
    corpus_dir = (
        PROJECT_ROOT
        / "evaluation"
        / "corpus"
    )

    noise_dir = (
        PROJECT_ROOT
        / "evaluation"
        / "extended_noise"
    )

    paths = sorted(
        corpus_dir.glob("*.md")
    )

    paths += sorted(
        noise_dir.glob("*.md")
    )

    return paths


# ---------------------------------------------------------------------
# Build isolated retriever
# ---------------------------------------------------------------------

def build_isolated_store(
    corpus: list[Path],
    retriever: str,
):
    settings = get_settings()

    temp_dir = tempfile.TemporaryDirectory(
        prefix="rag_eval_"
    )

    # Keep chunking identical to the official retrieval benchmark.
    eval_settings = settings.model_copy(
        update={
            "faiss_index_dir": (
                Path(temp_dir.name)
                / "faiss_index"
            ),
            "chunk_size": 500,
            "chunk_overlap": 50,
        }
    )

    # --------------------------------------------------------------
    # BM25
    # --------------------------------------------------------------

    if retriever == "bm25":
        retrieval_store = LexicalStore(
            Path(temp_dir.name)
            / "bm25_index"
        )

    # --------------------------------------------------------------
    # Dense BGE + FAISS
    # --------------------------------------------------------------

    elif retriever == "dense":
        embedding_service = EmbeddingService(
            eval_settings
        )

        retrieval_store = VectorStoreManager(
            eval_settings,
            embedding_service.get_embeddings(),
        )

    else:
        temp_dir.cleanup()

        raise ValueError(
            f"Unsupported retriever: {retriever}"
        )

    processor = DocumentProcessor(
        eval_settings
    )

    total_chunks = 0

    for path in corpus:
        path = (
            path
            .expanduser()
            .resolve()
        )

        if not path.exists():
            temp_dir.cleanup()

            raise FileNotFoundError(
                f"Corpus file not found: {path}"
            )

        docs = processor.process_file(
            path
        )

        total_chunks += (
            retrieval_store.add_documents(
                docs
            )
        )

    return (
        eval_settings,
        retrieval_store,
        total_chunks,
        temp_dir,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if any(
        k <= 0
        for k in args.ks
    ):
        raise ValueError(
            "All K values must be positive integers."
        )

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    dataset_path = (
        args.dataset
        .expanduser()
        .resolve()
    )

    dataset = load_dataset(
        dataset_path
    )

    if args.limit is not None:
        dataset = dataset[
            : args.limit
        ]

    # --------------------------------------------------------------
    # Corpus
    # --------------------------------------------------------------

    if args.corpus:
        corpus = [
            path
            .expanduser()
            .resolve()
            for path in args.corpus
        ]

    else:
        corpus = default_corpus()

    if not corpus:
        raise ValueError(
            "No corpus documents found."
        )

    # --------------------------------------------------------------
    # Build retriever
    # --------------------------------------------------------------

    (
        settings,
        retrieval_store,
        total_chunks,
        temp_dir,
    ) = build_isolated_store(
        corpus,
        args.retriever,
    )

    # --------------------------------------------------------------
    # RAG
    # --------------------------------------------------------------

    rag_chain: RAGChain | None = None

    if not args.retrieval_only:
        rag_chain = RAGChain(
            settings,
            retrieval_store,
        )

    # --------------------------------------------------------------
    # Judge
    # --------------------------------------------------------------

    judge_llm: ChatOpenAI | None = None

    if args.judge:
        if args.retrieval_only:
            temp_dir.cleanup()

            raise ValueError(
                "--judge cannot be used together with --retrieval-only"
            )

        if not settings.dashscope_api_key:
            temp_dir.cleanup()

            raise ValueError(
                "--judge requires DASHSCOPE_API_KEY "
                "in frontend/local_rag/.env"
            )

        judge_llm = ChatOpenAI(
            model=(
                args.judge_model
                or settings.chat_model
            ),
            api_key=(
                settings.dashscope_api_key
            ),
            base_url=(
                settings.dashscope_base_url
            ),
            temperature=0,
        )

    max_k = max(
        args.ks
    )

    answerable_rows = [
        row
        for row in dataset
        if bool(
            row["answerable"]
        )
    ]

    unanswerable_rows = [
        row
        for row in dataset
        if not bool(
            row["answerable"]
        )
    ]

    details: list[
        dict[str, Any]
    ] = []

    retrieval_latencies: list[
        float
    ] = []

    total_latencies: list[
        float
    ] = []

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    print("=" * 76)
    print("FINAL RAG BENCHMARK")
    print("=" * 76)

    print(
        f"Dataset: {dataset_path}"
    )

    print(
        f"Retriever: {args.retriever}"
    )

    print(
        f"Corpus files: {len(corpus)}"
    )

    print(
        f"Chunks: {total_chunks}"
    )

    print(
        f"Questions: {len(dataset)}"
    )

    print(
        f"Answerable: {len(answerable_rows)}"
    )

    print(
        f"Unanswerable: {len(unanswerable_rows)}"
    )

    print(
        f"Chunk size: 500"
    )

    print(
        f"Chunk overlap: 50"
    )

    print(
        f"K values: {args.ks}"
    )

    print(
        f"Judge: {args.judge}"
    )

    print("=" * 76)

    # --------------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------------

    for idx, item in enumerate(
        dataset,
        start=1,
    ):
        question = str(
            item["question"]
        )

        # ----------------------------------------------------------
        # Retrieval
        # ----------------------------------------------------------

        retrieval_start = (
            time.perf_counter()
        )

        docs = retrieval_store.search(
            question,
            k=max_k,
        )

        retrieval_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        retrieval_latencies.append(
            retrieval_ms
        )

        rank = first_relevant_rank(
            docs,
            item,
        )

        row: dict[str, Any] = {
            "id": item["id"],
            "question": question,
            "answerable": bool(
                item["answerable"]
            ),
            "reference_answer": item[
                "reference_answer"
            ],
            "source_contains": item.get(
                "source_contains",
                "",
            ),
            "first_relevant_rank": (
                rank
                if rank is not None
                else ""
            ),
            "reciprocal_rank": (
                round(
                    1 / rank,
                    6,
                )
                if rank
                else 0.0
            ),
            "retrieval_ms": round(
                retrieval_ms,
                2,
            ),
            "retrieved_sources": " | ".join(
                str(
                    doc.get(
                        "metadata",
                        {},
                    ).get(
                        "source",
                        "",
                    )
                )
                for doc in docs
            ),
        }

        for k in args.ks:
            row[
                f"hit@{k}"
            ] = int(
                rank is not None
                and rank <= k
            )

        # ----------------------------------------------------------
        # Answer generation
        # ----------------------------------------------------------

        if (
            not args.retrieval_only
            and rag_chain is not None
        ):
            total_start = (
                time.perf_counter()
            )

            result = rag_chain.run(
                question,
                history=[],
            )

            total_ms = (
                time.perf_counter()
                - total_start
            ) * 1000

            total_latencies.append(
                total_ms
            )

            answer = str(
                result.get(
                    "answer",
                    "",
                )
            )

            answer_sources = (
                result.get(
                    "sources",
                    [],
                )
                or []
            )

            retrieved_context = (
                "\n\n".join(
                    str(
                        source.get(
                            "content",
                            "",
                        )
                    )
                    for source
                    in answer_sources
                )
            )

            row[
                "generated_answer"
            ] = answer

            row[
                "total_ms"
            ] = round(
                total_ms,
                2,
            )

            # ------------------------------------------------------
            # Rule metrics
            # ------------------------------------------------------

            if bool(
                item["answerable"]
            ):
                row[
                    "basic_correct"
                ] = int(
                    basic_answer_correct(
                        answer,
                        str(
                            item[
                                "reference_answer"
                            ]
                        ),
                    )
                )

                row[
                    "safe_refusal"
                ] = ""

            else:
                row[
                    "basic_correct"
                ] = ""

                row[
                    "safe_refusal"
                ] = int(
                    looks_like_safe_refusal(
                        answer
                    )
                )

            # ------------------------------------------------------
            # LLM Judge
            # ------------------------------------------------------

            if judge_llm is not None:
                try:
                    scores = judge_answer(
                        judge_llm,
                        question=question,
                        reference_answer=str(
                            item[
                                "reference_answer"
                            ]
                        ),
                        answerable=bool(
                            item[
                                "answerable"
                            ]
                        ),
                        retrieved_context=(
                            retrieved_context
                        ),
                        generated_answer=(
                            answer
                        ),
                    )

                    for (
                        metric,
                        value,
                    ) in scores.items():
                        row[
                            f"judge_{metric}"
                        ] = round(
                            value,
                            4,
                        )

                except Exception as exc:
                    row[
                        "judge_error"
                    ] = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

        details.append(
            row
        )

        print(
            f"[{idx:03d}/{len(dataset):03d}] "
            f"{item['id']} "
            f"rank={rank or '-'} "
            f"retrieval={retrieval_ms:.2f} ms"
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    summary: dict[
        str,
        Any,
    ] = {
        "dataset": str(
            dataset_path
        ),
        "retriever": (
            args.retriever
        ),
        "corpus_files": len(
            corpus
        ),
        "chunks": (
            total_chunks
        ),
        "chunk_size": 500,
        "chunk_overlap": 50,
        "questions": len(
            dataset
        ),
        "answerable_questions": len(
            answerable_rows
        ),
        "unanswerable_questions": len(
            unanswerable_rows
        ),
        "retrieval": {},
        "latency": {
            "retrieval": latency_summary(
                retrieval_latencies
            ),
            "end_to_end": latency_summary(
                total_latencies
            ),
        },
    }

    # -----------------------------------------------------------------
    # Retrieval metrics
    # -----------------------------------------------------------------

    answerable_details = [
        row
        for row in details
        if row["answerable"]
    ]

    denom = max(
        len(
            answerable_details
        ),
        1,
    )

    for k in args.ks:
        summary[
            "retrieval"
        ][f"hit@{k}"] = round(
            sum(
                int(
                    row[
                        f"hit@{k}"
                    ]
                )
                for row
                in answerable_details
            )
            / denom,
            4,
        )

    summary[
        "retrieval"
    ]["mrr"] = round(
        sum(
            float(
                row[
                    "reciprocal_rank"
                ]
            )
            for row
            in answerable_details
        )
        / denom,
        4,
    )

    # -----------------------------------------------------------------
    # Rule-based answer metrics
    # -----------------------------------------------------------------

    if not args.retrieval_only:
        answerable_scored = [
            row
            for row in details
            if (
                row["answerable"]
                and row.get(
                    "basic_correct"
                )
                != ""
            )
        ]

        unanswerable_scored = [
            row
            for row in details
            if (
                not row[
                    "answerable"
                ]
                and row.get(
                    "safe_refusal"
                )
                != ""
            )
        ]

        basic_correctness = (
            sum(
                int(
                    row[
                        "basic_correct"
                    ]
                )
                for row
                in answerable_scored
            )
            / max(
                len(
                    answerable_scored
                ),
                1,
            )
        )

        refusal_accuracy = None

        if unanswerable_scored:
            refusal_accuracy = (
                sum(
                    int(
                        row[
                            "safe_refusal"
                        ]
                    )
                    for row
                    in unanswerable_scored
                )
                / len(
                    unanswerable_scored
                )
            )

        summary[
            "answering"
        ] = {
            "basic_correctness": round(
                basic_correctness,
                4,
            ),
            "refusal_accuracy": (
                round(
                    refusal_accuracy,
                    4,
                )
                if refusal_accuracy
                is not None
                else None
            ),
        }

    # -----------------------------------------------------------------
    # LLM Judge summary
    # -----------------------------------------------------------------

    if args.judge:
        judged_rows = [
            row
            for row in details
            if (
                "judge_correctness"
                in row
            )
        ]

        if judged_rows:
            summary[
                "llm_judge"
            ] = {}

            # Overall metrics
            for metric in (
                "correctness",
                "faithfulness",
                "relevance",
            ):
                values = [
                    float(
                        row[
                            f"judge_{metric}"
                        ]
                    )
                    for row
                    in judged_rows
                    if (
                        f"judge_{metric}"
                        in row
                    )
                ]

                summary[
                    "llm_judge"
                ][metric] = (
                    round(
                        sum(values)
                        / len(values),
                        4,
                    )
                    if values
                    else None
                )

            # Safe refusal must only be measured on
            # unanswerable questions.
            refusal_judged = [
                row
                for row in judged_rows
                if not row[
                    "answerable"
                ]
                and (
                    "judge_safe_refusal"
                    in row
                )
            ]

            if refusal_judged:
                safe_refusal = (
                    sum(
                        float(
                            row[
                                "judge_safe_refusal"
                            ]
                        )
                        for row
                        in refusal_judged
                    )
                    / len(
                        refusal_judged
                    )
                )

                summary[
                    "llm_judge"
                ][
                    "safe_refusal_unanswerable"
                ] = round(
                    safe_refusal,
                    4,
                )

            else:
                summary[
                    "llm_judge"
                ][
                    "safe_refusal_unanswerable"
                ] = None

            summary[
                "llm_judge"
            ][
                "judged_questions"
            ] = len(
                judged_rows
            )

            summary[
                "llm_judge"
            ][
                "judged_unanswerable"
            ] = len(
                refusal_judged
            )

    # -----------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------

    if args.output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "evaluation"
            / "results"
            / f"rag_{args.retriever}"
        )
    else:
        output_dir = (
            args.output_dir
            .expanduser()
            .resolve()
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    details_path = (
        output_dir
        / "evaluation_details.csv"
    )

    summary_path = (
        output_dir
        / "evaluation_summary.json"
    )

    # -----------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------

    fieldnames: list[str] = []

    for row in details:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with details_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            details
        )

    # -----------------------------------------------------------------
    # JSON
    # -----------------------------------------------------------------

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------

    print("\n" + "=" * 76)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("=" * 76)

    print(
        f"Details: {details_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    temp_dir.cleanup()


if __name__ == "__main__":
    main()