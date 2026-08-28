"""
启动入口：在项目根目录下运行
    python run.py
即可启动 RAG + AI聊天助手 服务，浏览器打开 http://127.0.0.1:8000 查看页面。
"""
import uvicorn

from frontend.local_rag.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
