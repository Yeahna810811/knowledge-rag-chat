"""合并并校验 v3 评测集（原 50 题 + 新增 50 题 = 100 题）。

为什么必须先校验再合并：
新增题目的 evidence 是我照着语料手写的，只要有一个字对不上，
这道題就永远检索不到、永远判为未命中——它会静默地把 Hit@K 拉低，
而且你根本看不出来是"检索差"还是"标签写错了"。
所以合并前逐条做原文比对，对不上的直接报错并列出。

校验项：
1. 可答题的 evidence 必须是语料某个文件的**逐字子串**（归一化空白后比对）；
2. 不可答题的 evidence 必须为空；
3. id 不能重复；
4. 可答题的 reference_answer 不能为空。

用法：
    python evaluation/build_v3_dataset.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "evaluation" / "corpus"
SOURCES = [
    PROJECT_ROOT / "evaluation" / "eval_dataset.jsonl",
    PROJECT_ROOT / "evaluation" / "eval_dataset_extra.jsonl",
]
OUTPUT = PROJECT_ROOT / "evaluation" / "eval_dataset_v3.jsonl"


def norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def main() -> int:
    corpus_text = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))
    )
    corpus_norm = norm(corpus_text)

    rows: list[dict] = []
    seen: set[str] = set()
    for src in SOURCES:
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))

    errors: list[str] = []
    for row in rows:
        qid = str(row.get("id", "?"))
        if qid in seen:
            errors.append(f"{qid}: id 重复")
        seen.add(qid)

        if bool(row.get("answerable")):
            ev = norm(str(row.get("evidence", "")))
            if not ev:
                errors.append(f"{qid}: 可答题但 evidence 为空")
            elif ev not in corpus_norm:
                errors.append(f"{qid}: evidence 不是语料原文 -> {str(row.get('evidence'))[:40]}")
            if not str(row.get("reference_answer", "")).strip():
                errors.append(f"{qid}: 可答题但 reference_answer 为空")
        else:
            if str(row.get("evidence", "")).strip():
                errors.append(f"{qid}: 不可答题却有 evidence")

    if errors:
        print("校验未通过：")
        for e in errors:
            print("  -", e)
        return 1

    answerable = sum(1 for r in rows if r["answerable"])
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"校验通过：{len(rows)} 题（可答 {answerable} / 不可答 {len(rows) - answerable}）")
    print(f"已写入: {OUTPUT}")
    print()
    print("下一步：")
    print("  python evaluation/build_extended_corpus.py    # 用全量 evidence 重新剔除污染块")
    print("  python evaluation/validate_distractors.py     # 确认 0 污染")
    return 0


if __name__ == "__main__":
    sys.exit(main())
