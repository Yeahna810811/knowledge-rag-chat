"""Optional Ragas-backed evaluation on top of the existing JSONL benchmark.

Keeps Hit@K / MRR in evaluate_rag.py, and adds Ragas faithfulness /
answer relevancy when --ragas is enabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.local_rag.config.settings import get_settings
from frontend.local_rag.core.document_processor import DocumentProcessor
from frontend.local_rag.core.embedding_service import EmbeddingService
from frontend.local_rag.core.rag_chain import RAGChain
from frontend.local_rag.core.vector_store import VectorStoreManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate answers with Ragas metrics.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "eval_dataset.jsonl",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        default=[
            PROJECT_ROOT / "evaluation" / "corpus" / "rag_architecture.md",
            PROJECT_ROOT / "evaluation" / "corpus" / "api_guide.md",
            PROJECT_ROOT / "evaluation" / "corpus" / "user_manual.md",
            PROJECT_ROOT / "evaluation" / "corpus" / "deployment_guide.md",
        ],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "ragas_summary.json",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as exc:
        raise SystemExit(
            "Ragas extras missing. Install with:\n"
            "  pip install -r evaluation/requirements-eval.txt\n"
            f"Original error: {exc}"
        )

    args = parse_args()
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise SystemExit("DASHSCOPE_API_KEY is required for Ragas judge/embeddings bridge.")

    import tempfile

    rows = load_dataset(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_settings = settings.model_copy(
            update={"faiss_index_dir": Path(tmp) / "faiss", "upload_dir": Path(tmp) / "uploads"}
        )
        embeddings = EmbeddingService(tmp_settings).get_embeddings()
        store = VectorStoreManager(tmp_settings, embeddings)
        processor = DocumentProcessor(tmp_settings)
        for path in args.corpus:
            chunks = processor.process_file(path)
            store.add_documents(chunks)
        chain = RAGChain(tmp_settings, store)

        questions: list[str] = []
        answers: list[str] = []
        contexts: list[list[str]] = []
        ground_truths: list[str] = []

        for row in rows:
            result = chain.run(row["question"], mode="rag")
            questions.append(row["question"])
            answers.append(result["answer"])
            contexts.append([src["content"] for src in result["sources"]])
            ground_truths.append(row.get("reference_answer", ""))

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    judge = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        temperature=0,
    )
    # Ragas still expects an embeddings object; reuse DashScope-compatible OpenAI client if available,
    # otherwise fall back to local sentence-transformers via a thin wrapper is not supported here.
    # Prefer local embeddings through a dummy OpenAIEmbeddings only when a compatible base is set.
    emb = OpenAIEmbeddings(
        model="text-embedding-v3",
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge,
        embeddings=emb,
    )
    summary = dict(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
