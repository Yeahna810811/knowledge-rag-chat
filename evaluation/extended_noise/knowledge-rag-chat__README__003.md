cal_rag/.env.example frontend/local_rag/.env
```

至少填写 `DASHSCOPE_API_KEY`。可选开启 LangSmith：

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=knowledge-rag-chat
```

### 3. 前端构建

```bash
cd frontend/web
npm install
npm run build
cd ../..
```

开发时可另开终端：`npm run dev`（Vite 代理 `/api` → `8000`）。

### 4. 启动

```bash
python run.py
```

- Web：http://127.0.0.1:8000
- Swagger：http://127.0.0.1:8000/docs

## API