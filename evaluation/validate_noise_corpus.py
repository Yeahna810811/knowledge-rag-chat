"""难负样本校验器：确保噪声语料不会污染评测标注。

构造难负样本最大的风险是「假阳性」——如果噪声 chunk 里恰好出现了某条 evidence
的原文，那么检索到它也会被判定为「命中」，评测结果就虚高了。

本脚本对 evaluation/extended_noise 下每个噪声 chunk 与**全部** eval_dataset*.jsonl
的 evidence 做 bigram 包含度计算，超过阈值的 chunk 必须从噪声集中剔除
（build_extended_corpus.py 已按同一阈值自动剔除，本脚本用于独立复核）。

用法：
    python evaluation/validate_noise_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_retrieval_ab import build_chunks, token_containment

THRESHOLD = 0.85  # 与 is_relevant 一致
CONTAINMENT = token_containment  # 与 --metric bigram 保持一致
NOISE_DIR = PROJECT_ROOT / "evaluation" / "extended_noise"


def load_all_evidences() -> list[tuple[str, str]]:
    """读取全部数据集的 evidence（含 v3 扩题），避免只校验旧标注。"""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path in sorted((PROJECT_ROOT / "evaluation").glob("eval_dataset*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            evidence = str(row.get("evidence", "")).strip()
            if evidence and evidence not in seen:
                seen.add(evidence)
                out.append((str(row.get("id", "?")), evidence))
    return out


def main() -> None:
    evidences = load_all_evidences()
    noise_paths = sorted(NOISE_DIR.glob("*.md"))
    if not noise_paths:
        print(f"[FAIL] 噪声目录为空：{NOISE_DIR}\n请先运行 python evaluation/build_extended_corpus.py")
        raise SystemExit(1)

    chunks = build_chunks(noise_paths, 500, 50)

    print("=" * 74)
    print("难负样本校验")
    print(f"噪声 chunk: {len(chunks)} 个（来自 {len(noise_paths)} 个文件）")
    print(f"待比对 evidence: {len(evidences)} 条 | 阈值: {THRESHOLD}")
    print("=" * 74)

    contaminated: list[tuple[str, str, float, str]] = []
    worst: list[tuple[str, str, float]] = []

    for chunk in chunks:
        content = chunk["content"]
        local_worst, local_id = 0.0, ""
        for qid, evidence in evidences:
            score = CONTAINMENT(content, evidence)
            if score > local_worst:
                local_worst, local_id = score, qid
            if score >= THRESHOLD:
                contaminated.append(
                    (chunk["metadata"]["source"], qid, score, evidence)
                )
        name = f"{chunk['metadata']['source']}#{chunk['metadata']['chunk_index']}"
        worst.append((name, local_id, local_worst))

    worst.sort(key=lambda x: -x[2])
    print("\n包含度最高的 5 个噪声 chunk（越接近 1.0 越危险）：")
    for name, qid, score in worst[:5]:
        flag = "  <== 超标" if score >= THRESHOLD else ""
        print(f"  {score:.3f}  {name:<40} 最接近 {qid}{flag}")

    print()
    if contaminated:
        print(f"[FAIL] 发现 {len(contaminated)} 处污染，噪声集不可用：")
        for source, qid, score, evidence in contaminated[:20]:
            print(f"  {source} 命中 {qid} ({score:.3f}): {evidence[:50]}")
        raise SystemExit(1)

    max_score = worst[0][2] if worst else 0.0
    print(f"[OK] 无污染。最高包含度 {max_score:.3f} < {THRESHOLD}，"
          f"噪声样本可以安全加入评测。")


if __name__ == "__main__":
    main()
