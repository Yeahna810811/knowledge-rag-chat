在 5 个 chunk 时这个噪声被掩盖；语料一大就会实打实抬高 Hit@K。

### 修复

`evaluate_retrieval_ab.py` 新增 `--metric {char,bigram}`，默认 `bigram`。
bigram 保留两字顺序，「文档切分」与「切分文档」不再等价，区分度显著提升。

---

## 三、三档语料趋势（TF-IDF 代理稠密路，已被第四节真实数据取代）