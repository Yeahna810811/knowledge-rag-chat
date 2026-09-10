# 工程化与部署说明（Benchmark v3）

项目提供 Dockerfile，采用多阶段构建。

项目提供 docker-compose.yml 用于编排服务。

GitHub Actions 工作流文件为 .github/workflows/ci.yml。

CI 包含三个 job：backend-smoke、frontend-build、docker-build。

backend-smoke 使用 Python 3.11，执行 from frontend.local_rag.app import create_app 的导入冒烟测试。

frontend-build 使用 Node 20，执行 npm install 与 npm run build。

docker-build 依赖 backend-smoke 与 frontend-build 两个 job 成功后才会执行。

系统提供 /api/webhook 接口，用于自动化工作流集成。

webhook 请求需要携带 X-Webhook-Secret 请求头，校验失败返回 401 Invalid webhook secret。

可选的链路追踪通过 LangSmith 开启，需要配置 LANGCHAIN_TRACING_V2、LANGCHAIN_API_KEY 与 LANGCHAIN_PROJECT。
