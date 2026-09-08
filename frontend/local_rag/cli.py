#!/usr/bin/env python3
"""CLI entry point for local RAG operations."""

import argparse
from pathlib import Path

from frontend.local_rag.config.settings import get_settings
from frontend.local_rag.services.knowledge_service import KnowledgeService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local RAG Knowledge Base CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="Upload a local file")
    upload_parser.add_argument("file", type=Path, help="Path to document")

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", type=str, help="Question text")
    ask_parser.add_argument(
        "--mode",
        choices=["rag", "chat"],
        default="rag",
        help="rag=knowledge-base QA, chat=plain AI dialogue",
    )

    subparsers.add_parser("status", help="Show knowledge base status")
    subparsers.add_parser("reset", help="Clear vector index")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = get_settings()
    service = KnowledgeService(settings)

    if args.command == "upload":
        file_path: Path = args.file
        if not file_path.exists():
            raise SystemExit(f"File not found: {file_path}")
        result = service.ingest_upload(file_path.name, file_path.read_bytes())
        print(result)
        return

    if args.command == "ask":
        result = service.ask(args.question, mode=args.mode)
        print("\nMode:", result.get("mode", args.mode))
        print("\nQuestion:", result["question"])
        print("\nAnswer:", result["answer"])
        if result.get("agent_trace"):
            print("\nAgents:", " -> ".join(result["agent_trace"]))
        if result["sources"]:
            print("\nSources:")
            for idx, source in enumerate(result["sources"], start=1):
                print(f"  [{idx}] {source['metadata']} -> {source['content'][:120]}...")
        return

    if args.command == "status":
        print(service.status())
        return

    if args.command == "reset":
        print(service.reset())
        return


if __name__ == "__main__":
    main()
