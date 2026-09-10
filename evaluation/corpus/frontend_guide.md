# 前端实现说明（Benchmark v3）

前端技术栈为 Vue 3 与 TypeScript，版本为 vue 3.5.13。

构建工具为 Vite，构建插件为 @vitejs/plugin-vue。

前端源码目录为 frontend/web，构建产物输出到 frontend/web/dist。

前端构建命令为 npm install 与 npm run build。

回答区域支持 Markdown 渲染。

Markdown 渲染前会使用 DOMPurify 清洗，防止 XSS 注入，版本为 dompurify 3.2.4。

页面提供 rag 与 chat 两种模式切换。

前端通过在请求中携带 session_id 来维持同一会话上下文。

上传、清空知识库、清空记忆等操作都在页面上有对应按钮，无需调用命令行。
