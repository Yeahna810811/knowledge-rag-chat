ls/file_utils.py
└── data/
```

## 配置

复制环境变量模板：

```bash
cp frontend/local_rag/.env.example frontend/local_rag/.env
```

至少填写：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

默认配置：

```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHAT_MODEL=qwen-plus
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=4
```

## 从项目根目录运行

```bash
pip install -r frontend/local_rag/requirements.txt
python run.py
```