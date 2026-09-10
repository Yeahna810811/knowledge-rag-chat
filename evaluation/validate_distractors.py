"""难负样本校验器：确保干扰文档不会污染原有标注。

构造难负样本最大的风险是「假阳性」——如果干扰文档里恰好出现了某条 evidence
的原文，那么检索到它也会被判定为「命中」，评测结果就虚高了。

本脚本对每个干扰 chunk 与全部 50 条 evidence 做包含度计算，
超过阈值的 chunk 会被标记出来并要求剔除。

用法：
    python evaluation/validate_distractors.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_retrieval_ab import build_chunks, token_containment

THRESHOLD = 0.85  # 与 is_relevant 一致
CONTAINMENT = token_containment  # 与 --metric bigram 保持一致


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def main() -> None:
    dataset = [
        json.loads(line)
        for line in (PROJECT_ROOT / "evaluation" / "eval_dataset.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    evidences = [
        (row["id"], str(row.get("evidence", "")).strip())
        for row in dataset
        if str(row.get("evidence", "")).strip()
    ]

    distractor_dir = PROJECT_ROOT / "evaluation" / "distractors"
    distractor_paths = sorted(distractor_dir.glob("*.md"))
    chunks = build_chunks(distractor_paths, 500, 50)

    print("=" * 74)
    print("难负样本校验")
    print(f"干扰文档: {len(distractor_paths)} 篇 -> {len(chunks)} 个 chunk")
    print(f"待比对 evidence: {len(evidences)} 条 | 阈值: {THRESHOLD}")
    print("=" * 74)

    contaminated: list[tuple[str, str, float, str]] = []
    worst: list[tuple[str, str, float]] = []

    for chunk in chunks:
        content = chunk["content"]
        local_worst = 0.0
        local_id = ""
        for qid, evidence in evidences:
            score = CONTAINMENT(content, evidence)
            if score > local_worst:
                local_worst, local_id = score, qid
            if score >= THRESHOLD:
                contaminated.append((chunk["metadata"]["source"], qid, score, evidence))
        worst.append((chunk["metadata"]["source"] + f"#{chunk['metadata']['chunk_index']}",
                      local_id, local_worst))

    worst.sort(key=lambda x: -x[2])
    print("\n包含度最高的 5 个干扰 chunk（越接近 1.0 越危险）：")
    for name, qid, score in worst[:5]:
        flag = "  <== 超标" if score >= THRESHOLD else ""
        print(f"  {score:.3f}  {name:<34} 最接近 {qid}{flag}")

    print()
    if contaminated:
        print(f"[FAIL] 发现 {len(contaminated)} 处污染，必须修改干扰文档：")
        for source, qid, score, evidence in contaminated[:20]:
            print(f"  {source} 命中 {qid} ({score:.3f}): {evidence[:50]}")
        raise SystemExit(1)

    max_score = worst[0][2] if worst else 0.0
    print(f"[OK] 无污染。最高包含度 {max_score:.3f} < {THRESHOLD}，"
          f"干扰样本可以安全加入评测。")


if __name__ == "__main__":
    main()
