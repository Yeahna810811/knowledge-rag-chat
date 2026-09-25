export type ChatMode = "rag" | "chat";

export interface SourceItem {
  content: string;
  metadata: Record<string, unknown>;
}

export interface AskResponse {
  question: string;
  answer: string;
  sources: SourceItem[];
  mode: ChatMode;
  agent_trace?: string[];
  session_id?: string;
  history_turns?: number;
}

export interface HistoryTurn {
  question: string;
  answer: string;
}

export interface StatusResponse {
  vector_store_ready: boolean;
  embedding_model: string;
  chat_model: string;
  dashscope_configured: boolean;
  langsmith_enabled: boolean;
  active_sessions: number;
  agents: string[];
  modes: string[];
}

function getSessionId(): string {
  const key = "rag_session_id";
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const id = crypto.randomUUID();
  localStorage.setItem(key, id);
  return id;
}

export const sessionId = getSessionId();

async function readError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data.detail || data.message || res.statusText;
  } catch {
    return res.statusText;
  }
}

export async function fetchStatus(): Promise<StatusResponse> {
  const res = await fetch("/api/status");
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function uploadDocument(file: File): Promise<Record<string, unknown>> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: form });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function askQuestion(
  question: string,
  mode: ChatMode,
  signal?: AbortSignal,
): Promise<AskResponse> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId, mode }),
    signal,
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export interface StreamHandlers {
  onMeta?: (data: { question: string; session_id: string; mode: ChatMode }) => void;
  onSources?: (sources: SourceItem[]) => void;
  onDelta: (text: string) => void;
  onDone?: (payload: AskResponse) => void;
  onError?: (message: string) => void;
}

/**
 * 逐个 SSE 帧分发给回调。
 *
 * 这里手写解析而不用 EventSource：EventSource 只能发 GET，
 * 既装不下长问题，也没法带 AbortSignal —— 而"生成一半点停止"
 * 恰恰是流式最刚需的交互。
 */
function dispatchFrame(raw: string, handlers: StreamHandlers): void {
  const frame = raw.trim();
  // 心跳是 `: ping` 注释帧，直接忽略；空帧同理
  if (!frame || frame.startsWith(":")) return;

  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }
  if (!dataLines.length) return;

  let payload: Record<string, unknown> = {};
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    return; // 半包或脏帧：宁可丢一帧也不要整条流崩掉
  }

  switch (event) {
    case "meta":
      handlers.onMeta?.(payload as { question: string; session_id: string; mode: ChatMode });
      break;
    case "sources":
      handlers.onSources?.((payload.sources as SourceItem[]) || []);
      break;
    case "delta":
      handlers.onDelta((payload.text as string) || "");
      break;
    case "done":
      handlers.onDone?.(payload as unknown as AskResponse);
      break;
    case "error":
      handlers.onError?.((payload.message as string) || "流式回答失败");
      break;
  }
}

export async function streamAsk(
  question: string,
  mode: ChatMode,
  signal: AbortSignal | undefined,
  handlers: StreamHandlers,
): Promise<void> {
  const res = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, session_id: sessionId, mode }),
    signal,
  });
  // 限流（429）等非 2xx 在这里就被拦下：此时响应体是普通 JSON，
  // 不是 SSE，不能交给下面的帧解析器。
  if (!res.ok) throw new Error(await readError(res));
  if (!res.body) throw new Error("当前浏览器不支持流式响应（ReadableStream）");

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // 一个 chunk 可能包含多个帧，也可能只有半个帧：
      // 只处理以 \n\n 结尾的完整帧，剩下的留在 buffer 里等下一块。
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const raw = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        dispatchFrame(raw, handlers);
        boundary = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.cancel().catch(() => undefined);
  }
}

export async function fetchHistory(): Promise<{ history: HistoryTurn[] }> {
  const res = await fetch(`/api/history?session_id=${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function clearHistory(): Promise<void> {
  const res = await fetch("/api/clear_history", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await readError(res));
}

export async function resetKnowledgeBase(): Promise<void> {
  const res = await fetch("/api/reset", { method: "DELETE" });
  if (!res.ok) throw new Error(await readError(res));
}
