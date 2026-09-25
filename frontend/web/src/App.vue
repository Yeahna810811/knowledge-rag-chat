<template>
  <div class="app">
    <!-- ===== 顶栏 ===== -->
    <header class="topbar">
      <div class="brand">
        <div class="logo">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1" />
            <circle cx="12" cy="12" r="4" />
          </svg>
        </div>
        <div>
          <h1>知识库 RAG 助手</h1>
          <p class="sub">文档解析 · 语义检索 · 多 Agent 协同</p>
        </div>
      </div>
      <div class="badges">
        <span class="badge">
          <i class="dot" :class="status?.vector_store_ready ? 'on' : ''"></i>
          知识库 {{ status?.vector_store_ready ? "就绪" : "未初始化" }}
        </span>
        <span class="badge"><i class="dot on"></i>模型 {{ status?.chat_model || "…" }}</span>
        <span class="badge"><i class="dot"></i>会话 {{ status?.active_sessions ?? "…" }}</span>
      </div>
    </header>

    <!-- ===== 主体 ===== -->
    <div class="layout">
      <!-- 左侧功能栏 -->
      <aside class="side">
        <section class="card">
          <h2 class="card-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
            文档入库
          </h2>
          <div
            class="upload"
            :class="{ over: dragOver }"
            @click="fileInput?.click()"
            @dragover.prevent="dragOver = true"
            @dragleave.prevent="dragOver = false"
            @drop.prevent="onDrop"
          >
            <div class="upload-ic">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7" /></svg>
            </div>
            <p class="upload-main">点击选择或拖拽文件</p>
            <p class="upload-hint">支持 TXT / PDF / DOCX / MD / CSV</p>
            <p class="upload-note">上传后自动切片并建立向量索引</p>
          </div>
          <input ref="fileInput" type="file" accept=".txt,.md,.pdf,.docx,.csv" class="hidden-input" @change="onUpload" :disabled="busy" />
          <p v-if="uploadTip" class="tip" :class="uploadOk ? 'ok' : 'err'">
            <svg v-if="uploadOk" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" /></svg>
            <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" /></svg>
            {{ uploadTip }}
          </p>
        </section>

        <section class="card">
          <h2 class="card-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" /></svg>
            运行状态
          </h2>
          <ul class="status-list">
            <li><span>对话模型</span><b>{{ status?.chat_model || "—" }}</b></li>
            <li><span>向量模型</span><b class="dim">{{ shortName(status?.embedding_model) || "—" }}</b></li>
            <li><span>向量索引</span><b :class="status?.vector_store_ready ? 'good' : 'warn'">{{ status?.vector_store_ready ? "已就绪" : "未初始化" }}</b></li>
            <li><span>API 配置</span><b :class="status?.dashscope_configured ? 'good' : 'warn'">{{ status?.dashscope_configured ? "已配置" : "未配置" }}</b></li>
            <li><span>活动会话</span><b>{{ status?.active_sessions ?? "—" }}</b></li>
            <li><span>智能体</span><b class="dim">{{ (status?.agents || []).join(" · ") || "—" }}</b></li>
          </ul>
        </section>

        <section class="card">
          <h2 class="card-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" /><line x1="16" y1="2" x2="16" y2="6" /><line x1="8" y1="2" x2="8" y2="6" /><line x1="3" y1="10" x2="21" y2="10" /></svg>
            会话管理
          </h2>
          <button class="op" @click="onClearHistory" :disabled="busy">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" /></svg>
            清空对话记忆
            <span class="op-note">不删知识库</span>
          </button>
          <button class="op danger" @click="onResetKb" :disabled="busy">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></svg>
            清空知识库
            <span class="op-note">不可恢复</span>
          </button>
        </section>
      </aside>

      <!-- 右侧聊天区 -->
      <main class="chat">
        <div class="chat-head">
          <div class="mode-switch">
            <button type="button" :class="{ active: mode === 'rag' }" @click="mode = 'rag'">RAG 知识库</button>
            <button type="button" :class="{ active: mode === 'chat' }" @click="mode = 'chat'">普通 AI 对话</button>
          </div>
          <span class="mode-desc">{{ mode === "rag" ? "自动检索知识库，回答标注来源" : "直接调用大模型对话" }}</span>
        </div>

        <div ref="chatBox" class="chat-box">
          <div v-if="!messages.length" class="empty">
            <div class="empty-ic">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>
            </div>
            <p class="empty-t">向知识库提问，开始对话</p>
            <p class="empty-d">回答将基于已上传的文档生成，并标注参考来源</p>
          </div>

          <div v-for="(msg, idx) in messages" :key="idx" class="msg" :class="msg.role">
            <div v-if="msg.role === 'assistant'" class="avatar bot">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1" /><circle cx="12" cy="12" r="4" /></svg>
            </div>
            <div v-else class="avatar me">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" /></svg>
            </div>
            <div class="bubble">
              <!-- 流式过程中按纯文本渲染：半截的 ``` 或未闭合的 ** 会被
                   markdown 解析器渲染成奇怪的东西，等 done 再转成富文本 -->
              <template v-if="msg.role === 'assistant'">
                <div v-if="msg.streaming" class="plain streaming-text">
                  {{ msg.content }}<span class="caret"></span>
                </div>
                <div v-else class="md" v-html="renderMarkdown(msg.content)" />
              </template>
              <div v-else class="plain">{{ msg.content }}</div>
              <div v-if="msg.sources?.length" class="sources">
                <details>
                  <summary>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6" /></svg>
                    参考 {{ msg.sources.length }} 段{{ msg.agents?.length ? " · Agents " + msg.agents.join(" → ") : "" }}
                  </summary>
                  <div v-for="(src, sIdx) in msg.sources" :key="sIdx" class="src-item">
                    <span class="src-meta">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>
                      片段 {{ sIdx + 1 }}{{ src.metadata?.source ? " · " + src.metadata.source : "" }}
                    </span>
                    {{ truncate(src.content, 180) }}
                  </div>
                </details>
              </div>
              <div v-else-if="msg.agents?.length" class="sources">{{ msg.agents.join(" → ") }}</div>
            </div>
          </div>

          <!-- 流式一旦开始，气泡里已经在长字了，再挂一个"正在思考"的
               三点动画就是自相矛盾的 UI -->
          <div v-if="busy && !error && !streaming" class="msg assistant">
            <div class="avatar bot">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1" /><circle cx="12" cy="12" r="4" /></svg>
            </div>
            <div class="bubble typing"><i></i><i></i><i></i></div>
          </div>
        </div>

        <div class="composer">
          <textarea
            v-model="draft"
            rows="1"
            placeholder="输入问题，Enter 发送，Shift+Enter 换行"
            @keydown="onKeydown"
            @input="autoGrow"
            :disabled="busy"
          />
          <div class="actions">
            <button class="ghost" type="button" @click="onClearHistory" :disabled="busy">清空记录</button>
            <button
              class="send"
              :class="{ stop: asking }"
              type="button"
              @click="asking ? stopGeneration() : onAsk()"
              :disabled="asking ? false : busy || !draft.trim()"
            >
              <svg v-if="!asking" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" /></svg>
              <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
              {{ asking ? "停止" : "发送" }}
            </button>
          </div>
        </div>
        <p v-if="error" class="error-line">{{ error }}</p>
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, onMounted, ref } from "vue";
import { marked } from "marked";
import DOMPurify from "dompurify";
import {
  clearHistory,
  fetchHistory,
  fetchStatus,
  resetKnowledgeBase,
  streamAsk,
  type ChatMode,
  type SourceItem,
  type StatusResponse,
  uploadDocument,
} from "./api";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
  agents?: string[];
  /** 正在流式输出：此时按纯文本渲染，避免半个 markdown 块被渲染成乱码 */
  streaming?: boolean;
}

const mode = ref<ChatMode>("rag");
const draft = ref("");
const busy = ref(false);
const asking = ref(false);
/** 已经有内容在往屏幕上吐了（此时不该再显示"正在思考"的打字动画） */
const streaming = ref(false);
const error = ref("");
const uploadTip = ref("");
const uploadOk = ref(false);
const dragOver = ref(false);
const status = ref<StatusResponse | null>(null);
const messages = ref<ChatMessage[]>([]);
const chatBox = ref<HTMLElement | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const controller = ref<AbortController | null>(null);

marked.setOptions({ breaks: true, gfm: true });

function renderMarkdown(text: string): string {
  const html = marked.parse(text || "") as string;
  return DOMPurify.sanitize(html);
}

function truncate(text: string, n: number): string {
  return text.length > n ? `${text.slice(0, n)}…` : text;
}

function shortName(model: string | undefined): string {
  return model ? model.split("/").pop() || model : "";
}

async function scrollToBottom() {
  await nextTick();
  if (chatBox.value) {
    chatBox.value.scrollTop = chatBox.value.scrollHeight;
  }
}

// 流式时每个 token 都滚一次会把主线程压满：合并到每帧最多一次。
let scrollScheduled = false;
function scheduleScroll() {
  if (scrollScheduled) return;
  scrollScheduled = true;
  requestAnimationFrame(() => {
    scrollScheduled = false;
    void scrollToBottom();
  });
}

function autoGrow(e?: Event) {
  const el = (e?.target as HTMLTextAreaElement) || null;
  const ta = el || (document.querySelector(".composer textarea") as HTMLTextAreaElement | null);
  if (!ta) return;
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
}

async function refreshStatus() {
  status.value = await fetchStatus();
}

async function loadHistory() {
  const data = await fetchHistory();
  messages.value = [];
  for (const turn of data.history || []) {
    messages.value.push({ role: "user", content: turn.question });
    messages.value.push({ role: "assistant", content: turn.answer });
  }
  await scrollToBottom();
}

async function onUpload(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  busy.value = true;
  error.value = "";
  uploadTip.value = "上传解析中…";
  uploadOk.value = false;
  try {
    const result = await uploadDocument(file);
    uploadOk.value = true;
    uploadTip.value = `已入库：${result.filename}（${result.chunk_count} 段）`;
    await refreshStatus();
  } catch (e) {
    uploadOk.value = false;
    uploadTip.value = e instanceof Error ? e.message : String(e);
  } finally {
    busy.value = false;
    input.value = "";
  }
}

function onDrop(event: DragEvent) {
  dragOver.value = false;
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  if (fileInput.value) fileInput.value.files = dt.files;
  const input = fileInput.value;
  if (input) {
    busy.value = true;
    error.value = "";
    uploadTip.value = "上传解析中…";
    uploadOk.value = false;
    uploadDocument(file)
      .then((result) => {
        uploadOk.value = true;
        uploadTip.value = `已入库：${result.filename}（${result.chunk_count} 段）`;
        return refreshStatus();
      })
      .catch((e) => {
        uploadOk.value = false;
        uploadTip.value = e instanceof Error ? e.message : String(e);
      })
      .finally(() => {
        busy.value = false;
        input.value = "";
      });
  }
}

async function onAsk() {
  const question = draft.value.trim();
  if (!question) return;
  busy.value = true;
  asking.value = true;
  error.value = "";
  messages.value.push({ role: "user", content: question });
  draft.value = "";
  // 强制同步清空输入框 DOM，防止输入法缓冲/默认行为把文字回填
  const ta = document.querySelector(".composer textarea") as HTMLTextAreaElement | null;
  if (ta) ta.value = "";
  await nextTick();
  autoGrow();
  await scrollToBottom();
  controller.value = new AbortController();

  // 先把空气泡放上去，后面每个 delta 直接往里追加——
  // 这是"打字机效果"的关键：用户看到的是逐步长出来的文字，
  // 而不是等十几秒后整段蹦出来。
  const reply: ChatMessage = {
    role: "assistant",
    content: "",
    sources: [],
    agents: [],
    streaming: true,
  };
  messages.value.push(reply);

  try {
    streaming.value = true;
    await streamAsk(question, mode.value, controller.value.signal, {
      onSources: (sources) => {
        reply.sources = sources;
      },
      onDelta: (text) => {
        reply.content += text;
        scheduleScroll();
      },
      onDone: (payload) => {
        reply.content = payload.answer;
        reply.sources = payload.sources;
        reply.agents = payload.agent_trace;
        reply.streaming = false;
      },
      onError: (message) => {
        error.value = message;
        reply.content = reply.content || `请求失败：${message}`;
        reply.streaming = false;
      },
    });
    await refreshStatus();
  } catch (e) {
    if ((e as Error)?.name === "AbortError") {
      // 主动停止：已经吐出来的内容保留（服务端也把这部分落库了），
      // 只是补一个"已停止"的标记，不让用户以为回答就这么长。
      reply.content = reply.content ? `${reply.content}\n\n_（已停止生成）_` : "已停止生成。";
    } else {
      error.value = e instanceof Error ? e.message : String(e);
      reply.content = reply.content || `请求失败：${error.value}`;
    }
    reply.streaming = false;
  } finally {
    streaming.value = false;
    busy.value = false;
    asking.value = false;
    controller.value = null;
    await scrollToBottom();
  }
}

function stopGeneration() {
  controller.value?.abort();
}

function onKeydown(event: KeyboardEvent) {
  // 中文输入法组合中（拼音/候选词未上屏）按 Enter 是选词确认，不是发送
  if (event.isComposing || event.keyCode === 229) return;
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void onAsk();
  }
}

async function onClearHistory() {
  busy.value = true;
  try {
    await clearHistory();
    messages.value = [];
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    busy.value = false;
  }
}

async function onResetKb() {
  if (!confirm("确认清空知识库向量索引？此操作不可恢复！")) return;
  busy.value = true;
  try {
    await resetKnowledgeBase();
    uploadTip.value = "知识库已清空";
    uploadOk.value = true;
    await refreshStatus();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    busy.value = false;
  }
}

onMounted(async () => {
  try {
    await refreshStatus();
    await loadHistory();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
  autoGrow();
});
</script>
