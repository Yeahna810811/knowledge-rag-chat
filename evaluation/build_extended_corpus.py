"""构建扩展语料：把项目自身文档作为真实噪声并入评测。

为什么用项目自己的文档做干扰：
- 它们是真实的工程文本，不是我编的假噪声，术语分布和真实知识库一致；
- 它们和评测语料同属一个项目，词汇高度重合，属于「难负样本」；
- 零成本，直接就能拿到几十个 chunk。

关键步骤是**自动剔除污染块**：项目 README 里也会提到 FastAPI、8000 端口这些事实，
如果某个 chunk 与任一 evidence 的 bigram 包含度 >= 0.85，检索到它会被误判为命中，
评测就虚高了。这类 chunk 必须剔掉，宁可少几个噪声也不能牺牲标签正确性。

用法：
    python evaluation/build_extended_corpus.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_retrieval_ab import build_chunks, token_containment

THRESHOLD = 0.85

# 项目自身的文本：作为噪声并入。刻意排除了 HTML/JS 产物（噪音类型不匹配）。
NOISE_SOURCES = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "frontend" / "local_rag" / "README.md",
    PROJECT_ROOT / "frontend" / "local_rag" / "sample.txt",
    PROJECT_ROOT / "evaluation" / "README.md",
]

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "extended_noise"


def main() -> None:
    dataset = [
        json.loads(line)
        for line in (
            PROJECT_ROOT / "evaluation" / "eval_dataset.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    evidences = [
        str(row.get("evidence", "")).strip()
        for row in dataset
        if str(row.get("evidence", "")).strip()
    ]

    existing = sorted((PROJECT_ROOT / "evaluation" / "distractors").glob("*.md"))
    sources = [p for p in NOISE_SOURCES if p.exists()] + existing

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    kept, dropped = 0, 0
    for src in sources:
        if src.suffix.lower() not in {".md", ".txt"}:
            continue
        text = src.read_text(encoding="utf-8")
        # 复用同一套切分逻辑，保证与评测时的 chunk 边界一致
        from evaluation.evaluate_retrieval_ab import chunk_document

        pieces = chunk_document(text, 500, 50)
        for i, piece in enumerate(pieces):
            worst = max((token_containment(piece, e) for e in evidences), default=0.0)
            if worst >= THRESHOLD:
                dropped += 1
                print(f"  [剔除] {src.name}#{i} 最高包含度 {worst:.3f}")
                continue
            # 文件名必须带上来源目录，否则项目根目录 README.md、
            # frontend/local_rag/README.md、evaluation/README.md 三个同名文件
            # 会互相覆盖，实际生成的 chunk 数远少于预期。
            prefix = src.parent.name or "root"
            name = f"{prefix}__{src.stem}__{i:03d}.md"
            (OUTPUT_DIR / name).write_text(piece, encoding="utf-8")
            kept += 1

    print()
    print(f"扩展噪声语料：保留 {kept} 个 chunk，剔除 {dropped} 个污染块")
    print(f"输出目录：{OUTPUT_DIR}")
    print()
    print("之后这样跑评测：")
    print("  python evaluation/evaluate_retrieval_ab.py \\")
    print("      --corpus evaluation/corpus/*.md evaluation/extended_noise/*.md --metric bigram")


if __name__ == "__main__":
    main()
