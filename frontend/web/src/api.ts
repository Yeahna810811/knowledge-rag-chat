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

export async function askQuestion(question: string, mode: ChatMode): Promise<AskResponse> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId, mode }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
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
