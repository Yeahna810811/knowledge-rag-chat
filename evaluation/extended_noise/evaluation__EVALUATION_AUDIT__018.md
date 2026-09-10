.md`），
   事实全部取自项目真实代码与配置（已逐条 grep 核实：Vue 3.5.13、
   DOMPurify 3.2.4、CI 三个 job 名、Node 20 / Python 3.11、
   webhook 401、`MAX_HISTORY_TURNS=10`、UUID 前缀重命名）。
2. 新增 50 道**独立**题目（40 可答 + 10 不可答）→ `eval_dataset_extra.jsonl`。
3. `build_v3_dataset.py` 逐条校验：**可答题的 evidence 必须是语料逐字子串**，
   否则直接报错。这防止「标签写错了」被误读成「检索差」。
4. 合并为 `eval_dataset_v3.jsonl`：**100 题（80 可答 + 20 不可答）**。
5. 同步生成 100 题的口语化版本 `eval_dataset_oral_v3.jsonl`
   （问题与 evidence 字面重合度 0.644 → 0.280）。