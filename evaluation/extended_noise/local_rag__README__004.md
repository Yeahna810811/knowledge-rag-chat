end/local_rag/requirements.txt
python run.py
```

Web：`http://127.0.0.1:8000`  
Docs：`http://127.0.0.1:8000/docs`

## CLI

```bash
python -m frontend.local_rag.cli upload frontend/local_rag/data/sample.txt
python -m frontend.local_rag.cli ask "LangChain 在项目中的作用是什么？"
python -m frontend.local_rag.cli status
python -m frontend.local_rag.cli reset
```