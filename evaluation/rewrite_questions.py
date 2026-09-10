"""把 50 道评测题改写成口语化 / 间接提问。

为什么改写：
原始题目几乎是把文档原句换个语气再问一遍（"当前默认嵌入模型是什么？"），
与原文共享大量字面 token。这让 BM25 天然占优——它不需要理解语义，
数一数词频重叠就能答对。这样的 benchmark 测不出稠密检索的真实能力，
也无法判断混合检索到底该不该上。

改写原则：
1. 只改 question，evidence / reference_answer / answerable 一律不动
   （相关性判定只看 evidence，所以标签体系完全不受影响）；
2. 用日常说法替换文档原词（"向量索引" -> "存和查语义向量的组件"）；
3. 不引入新的事实，不改变题意。

产物：evaluation/eval_dataset_oral.jsonl

用法：
    python evaluation/rewrite_questions.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# id -> 口语化改写
REWRITTEN: dict[str, str] = {
    "q001": "你们是用什么组件来存和查那些语义向量的？",
    "q002": "把文字变成一串数字这个活儿，是哪部分代码负责的？",
    "q003": "开箱默认用的是哪一款 embedding？",
    "q004": "长文档进库之前是靠什么切开的？",
    "q005": "一块大概切多长？相邻两块之间会留多少重叠？",
    "q006": "用户问一句话进来，系统最先干的是哪件事？",
    "q007": "一次问答默认最多会捞回几个片段来参考？",
    "q008": "参考资料凑齐之后，最后那段回答是谁来写的？",
    "q009": "靠哪个字段把不同人的聊天记录区分开？",
    "q010": "现在除了向量召回，有没有再叠一路关键词召回？",
    "q011": "服务端用的是哪个 Web 框架？",
    "q012": "服务跑起来之后默认开在哪个端口上？",
    "q013": "我传一个文件上去，后台大概要过哪几道工序？",
    "q014": "问答接口能不能带上会话标识？",
    "q015": "除了回答本身，接口还会返回什么能让我知道依据出处？",
    "q016": "纯聊天和带知识库问答，这俩模式差在哪儿？",
    "q017": "我只清聊天记录的话，之前灌进去的资料会一起没了吗？",
    "q018": "想启动整套 Web 服务，根目录该执行哪个脚本？",
    "q019": "本地想打开界面，浏览器地址栏敲什么？",
    "q020": "真正在生产跑的那份配置，落在哪个文件里？",
    "q021": "哪些格式的文件是我能直接扔进去的？",
    "q022": "Excel 那种表格文件现在能认吗？",
    "q023": "页面上点完上传之后，剩下的事情还要我手动做吗？",
    "q024": "一个库里能同时塞好几份文档吗？",
    "q025": "这种带资料问答的模式，拿来干什么最合适？",
    "q026": "想让前后几次提问算同一轮对话，该怎么做？",
    "q027": "清完记忆，之前建好的那些资料还在不在？",
    "q028": "点了那个既清知识又清记忆的按钮，会发生什么？",
    "q029": "现在能不能直接说话，或者拍照识别文字？",
    "q030": "日志里刷出 Hugging Face 那个未登录告警，是不是就说明挂了？",
    "q031": "服务默认绑在哪个地址上？",
    "q032": "默认对外开的端口号是多少？",
    "q033": "那个向量模型头一回用的时候是从哪儿拉下来的？",
    "q034": "负责生成回答的那个大模型用的是哪一款？",
    "q035": "接 DashScope 的时候，兼容 OpenAI 的那个地址长什么样？",
    "q036": "我申请到的 DashScope 密钥该填到哪儿？",
    "q037": "每次检索最多取几个结果，这个上限现在设的是几？",
    "q038": "想把索引文件换个地方存，要改哪个配置项？",
    "q039": "带真实密钥的那个配置文件能推到公开仓库吗？",
    "q040": "你们的评测脚本具体能量化出哪些东西？",
    "q041": "单个文件最多能传多大？",
    "q042": "线上是跑在亚马逊还是阿里云的机器上？",
    "q043": "同时能扛多少人在用？",
    "q044": "线上推理用的是什么显卡？",
    "q045": "每个月服务器花多少钱？",
    "q046": "这个系统是哪天上线的？",
    "q047": "能不能用微信扫码登录？",
    "q048": "接了公司的单点登录吗？",
    "q049": "可用性承诺是几个九？",
    "q050": "后台管理员初始账号密码是什么？",
}


# v3 新增 50 题的口语化改写（id -> 改写）
EXTRA_REWRITTEN: dict[str, str] = {
    "q051": "页面那层是用什么写的？",
    "q052": "打包构建用的是哪个工具？",
    "q053": "构建完的文件落在哪个文件夹？",
    "q054": "想把前端跑起来要敲哪些命令？",
    "q055": "回答内容能显示富文本吗，用的什么语法？",
    "q056": "显示用户内容之前做了什么防注入处理？",
    "q057": "界面上能切哪两种对话方式？",
    "q058": "怎么让系统知道我这几句话属于同一轮对话？",
    "q059": "打镜像用的是哪个文件，分几步？",
    "q060": "多个服务一起起，靠哪个配置文件？",
    "q061": "自动化的流水线配置放在哪个文件？",
    "q062": "流水线一共分成哪几个环节？",
    "q063": "跑后端检查时用的是哪个 Python 版本？",
    "q064": "打包前端时用的 Node 是几？",
    "q065": "构建镜像那一步要等哪两步先过？",
    "q066": "外部系统要触发任务，调哪个地址？",
    "q067": "回调带的密钥不对，对方会收到什么？",
    "q068": "想看每次调用的完整链路，要接什么？",
    "q069": "跑检索指标的是哪个脚本？",
    "q070": "题目和答案放在哪个文件里？",
    "q071": "有多少题是故意答不上来、用来测拒答的？",
    "q072": "评价召回好坏主要看哪两个数？",
    "q073": "用 Ragas 主要看回答的哪两方面？",
    "q074": "一条问答会依次经过哪几个处理单元？",
    "q075": "返回结果里哪个字段记录了走过的步骤？",
    "q076": "资料里压根没这回事的时候，系统会怎么回？",
    "q077": "耗时方面会分别记录哪两个数字？",
    "q078": "聊多了会不会忘，最多能记几轮？",
    "q079": "传两个名字一样的文件会打架吗？",
    "q080": "我传上去的文件存哪了？",
    "q081": "索引文件写在哪个文件夹？",
    "q082": "再加一份资料，之前的索引要推倒重来吗？",
    "q083": "只清聊天记录，会不会连资料一起没？",
    "q084": "点那个清空知识，会动到哪些数据？",
    "q085": "做清洗用的那个库是哪个版本？",
    "q086": "密钥要通过哪个 header 传过去？",
    "q087": "要设哪几个环境变量才能开追踪？",
    "q088": "题目集默认有多少道？",
    "q089": "前端代码放在哪个文件夹里？",
    "q090": "Vite 里负责解析 Vue 文件的插件叫啥？",
    "q091": "能在微信里直接用吗？",
    "q092": "关系型数据用的是哪种数据库？",
    "q093": "不同公司的数据能隔开吗？",
    "q094": "答案能念出来吗？",
    "q095": "监控指标接到 Prometheus 了吗？",
    "q096": "一个库最多能塞几份资料？",
    "q097": "能按角色限制谁能看到什么吗？",
    "q098": "上 K8s 有现成的编排文件吗？",
    "q099": "能把库里的内容导成 PDF 吗？",
    "q100": "回答是一个字一个字往外蹦吗？",
}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="evaluation/eval_dataset_v3.jsonl",
        help="默认对 v3 全量 100 题做改写；传 eval_dataset.jsonl 则只处理原 50 题",
    )
    ap.add_argument("--out", default="evaluation/eval_dataset_oral_v3.jsonl")
    args = ap.parse_args()

    src = PROJECT_ROOT / args.dataset
    dst = PROJECT_ROOT / args.out
    table = {**REWRITTEN, **EXTRA_REWRITTEN}

    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]

    missing = [r["id"] for r in rows if r["id"] not in table]
    if missing:
        raise KeyError(f"以下题目缺少改写：{missing}")

    before, after = [], []
    for row in rows:
        evidence = str(row.get("evidence", "")).strip()
        rewritten = table[row["id"]]
        if evidence:
            before.append(_overlap(row["question"], evidence))
            after.append(_overlap(rewritten, evidence))
        row["question"] = rewritten
        row["question_style"] = "oral"

    with dst.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"已生成: {dst}  ({len(rows)} 题)")
    if before:
        print(f"\n问题与 evidence 的字面重合度（越低越依赖语义理解）：")
        print(f"  改写前 {sum(before) / len(before):.3f}")
        print(f"  改写后 {sum(after) / len(after):.3f}")
        print(f"  下降   {sum(before) / len(before) - sum(after) / len(after):.3f}")


def _overlap(question: str, evidence: str) -> float:
    """问题与 evidence 的字符重合度，用来量化「问法有多贴近原文」."""
    import re

    def norm(text: str) -> str:
        return re.sub(r"\s+", "", str(text)).lower()

    q, e = norm(question), norm(evidence)
    if not q:
        return 0.0
    return sum(1 for ch in q if ch in e) / len(q)


if __name__ == "__main__":
    main()
