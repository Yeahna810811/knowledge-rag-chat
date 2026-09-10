"""中英文混排的轻量 tokenizer。

设计取舍：中文不做分词，直接用 unigram + bigram 建索引。

为什么不用 jieba：
1. 少一个重型依赖，CI 冷启动和 Docker 镜像都更快；
2. 中文检索里字符 bigram 的召回不输分词器，而且对未登录词天然友好——
   「bge-small-zh」「FAISS」「HMAC」「RRF」这类词分词器很容易切错，
   切错就意味着这个词永远召回不到；
3. query 和 doc 用同一套切法，不存在「分词器版本不一致导致漏召回」的问题。

代价：索引体积约 2 倍，对「苹果公司 vs 苹果好吃」这类需要词边界的歧义弱一些。
对这个项目（企业内部文档、术语密集）来说，收益大于代价。
"""

from __future__ import annotations

import re
import unicodedata

# 拉丁文 / 数字词：bge-small-zh-v1.5、FAISS、2026、3.5 都能整体命中
_LATIN_TOKEN = re.compile(r"[a-z0-9][a-z0-9._+#/:-]*")

# CJK 统一表意文字 + 扩展 A + 兼容表意文字
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def normalize(text: str) -> str:
    """NFKC 归一 + 小写。

    NFKC 这一步很关键：全角ＦＡＩＳＳ 和半角 FAISS 必须变成同一个 token，
    否则用户从 Word 里复制出来的全角提问会召回不到半角文档。
    """
    return unicodedata.normalize("NFKC", str(text)).lower()


def tokenize(text: str, ngram: int = 2) -> list[str]:
    """把一段文本切成 token 列表。

    ngram=2 => 中文同时产出 unigram 和 bigram。unigram 保证短查询不落空，
    bigram 提供词序信息，压制「上海」「海上」这类倒序噪声。
    """
    text = normalize(text)
    tokens: list[str] = []

    for match in _LATIN_TOKEN.finditer(text):
        tokens.append(match.group(0))

    for match in _CJK_RUN.finditer(text):
        run = match.group(0)
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run)  # unigram
        for size in range(2, ngram + 1):
            if len(run) >= size:
                tokens.extend(run[i : i + size] for i in range(len(run) - size + 1))

    return tokens
