"""构建扩展语料：把项目自身文档作为真实噪声并入评测。

为什么用项目自己的文档做干扰：
- 它们是仓库里真实存在的工程文本，不是编造的假文档，术语分布与真实知识库一致；
- 它们和评测语料同属一个项目，词汇高度重合，属于「难负样本」；
- 新增文档后重跑一次即可，噪声集始终与仓库同步，无需人工维护。

关键步骤是**自动剔除污染块**：项目 README 里也会提到 FastAPI、8000 端口这些事实，
如果某个 chunk 与任一 evidence 的 bigram 包含度太高，检索到它会被误判为命中，
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

# 判定「命中」用的是 0.85；构建噪声时用更严的 0.80 留出安全边际——
# 包含度 0.80~0.85 的 chunk 一旦换个切分边界就可能越线，从「难负样本」
# 变成「假阳性来源」，这类边缘块宁可不要。
THRESHOLD = 0.80

# 项目自身的文本：作为噪声并入。刻意排除 HTML/JS 产物（噪音类型不匹配），
# 也排除 eval 数据集与评测语料本身（那是标准答案来源，不是噪声）。
NOISE_SOURCES = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "frontend" / "local_rag" / "README.md",
    PROJECT_ROOT / "frontend" / "local_rag" / "sample.txt",
    PROJECT_ROOT / "frontend" / "local_rag" / "core" / "retrieval" / "README.md",
    PROJECT_ROOT / "evaluation" / "README.md",
    # 刻意不把 EVALUATION_AUDIT.md 纳入噪声：
    # 它是评测的产物而不是知识库文档，且正文引用了大量事实原文，
    # 纳进来会形成「报告改数字 -> 语料变 -> 指标变」的自指循环，
    # 也会带来跨文档污染的风险。评测报告应当只描述基准，不参与基准。
]

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "extended_noise"


def load_all_evidences() -> list[str]:
    """读取所有 eval_dataset*.jsonl 的 evidence。

    为什么用 glob 而不是写死一个文件：v3 扩题后有了 eval_dataset_extra.jsonl，
    如果只按原来的 50 条 evidence 剔除污染，新增题目对应的原文块会漏网——
    噪声语料里留着标准答案，评测就会虚高。
    """
    out: list[str] = []
    seen: set[str] = set()
    for path in sorted((PROJECT_ROOT / "evaluation").glob("eval_dataset*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = str(json.loads(line).get("evidence", "")).strip()
            if ev and ev not in seen:
                seen.add(ev)
                out.append(ev)
    return out


def main() -> None:
    evidences = load_all_evidences()
    print(f"载入 {len(evidences)} 条 evidence 用于污染剔除")

    sources = [p for p in NOISE_SOURCES if p.exists()]
    missing = [p.name for p in NOISE_SOURCES if not p.exists()]
    if missing:
        print(f"[警告] 以下噪声源不存在，已跳过：{missing}")

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
