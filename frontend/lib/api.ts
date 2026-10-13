// 统一的 API 客户端, 自动带 token (HttpOnly cookie + credentials: include)
// 防御性 trim：NEXT_PUBLIC_API_BASE 若被复制粘贴带了尾随空格 / 换行，模板字面量拼出
// "http://localhost:8000 /api/v1/..." 会让 fetch 抛 TypeError。
// 同时去掉尾随 /，避免 "http://localhost:8000//api/v1/..." 这种双斜杠。
const API = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000")
  .trim()
  .replace(/\/+$/, "");

// 本地演示默认知识库（已预置示例文档，免登录即可对话）
// 可用 NEXT_PUBLIC_DEMO_KB 覆盖: 录屏 / 多套环境演示时指向独立知识库,
// 避免把开发时随手上传的文档(个人简历等)混进演示画面。
export const DEMO_KB = (process.env.NEXT_PUBLIC_DEMO_KB || "kb-demo")
  .trim();

export type Source = {
  doc_id?: string;
  chunk_id: string;
  content: string;
  score: number;
  metadata?: Record<string, unknown>;
};

export type ChatMode = "rag" | "agent";

export type SessionMeta = {
  session_id: string;
  title: string;
  kb_id: string;
  mode: string;
  created_at: number;
  updated_at: number;
  message_count: number;
};

export type HistoryMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  ts: number;
};

export type SessionDetail = SessionMeta & {
  messages: HistoryMessage[];
};

// P2-11: 优先用 HttpOnly cookie 传输 JWT; 同时保留 Bearer header 兼容.
// fetch 设 credentials: 'include', 让浏览器自动带上后端写的 cookie.
// 对已有 token (例如从 localStorage 迁移 / dev-token 手动输入) 仍走 Authorization.
function authHeaders(token?: string): HeadersInit {
  const h: HeadersInit = {};
  if (token) {
    h["Authorization"] = `Bearer ${token}`;
  }
  return h;
}

const fetchInit: RequestInit = { credentials: "include" };

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ----- Auth -----
export async function register(username: string, email: string, password: string): Promise<{ access_token: string }> {
  return jsonOrThrow(await fetch(`${API}/api/v1/auth/register`, {
    ...fetchInit,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, email, password }),
  }));
}

export async function login(username: string, password: string): Promise<{ access_token: string }> {
  return jsonOrThrow(await fetch(`${API}/api/v1/auth/login`, {
    ...fetchInit,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  }));
}

export async function logout(): Promise<void> {
  await fetch(`${API}/api/v1/auth/logout`, {
    ...fetchInit,
    method: "POST",
  });
}

export async function devToken(): Promise<{ access_token: string }> {
  return jsonOrThrow(await fetch(`${API}/api/v1/auth/dev-token`, { ...fetchInit, method: "POST" }));
}

// ----- Chat (SSE 流式) -----
export async function* streamChat(params: {
  token?: string;
  question: string;
  session_id?: string;
  kb_id?: string;
  top_k?: number;
  mode?: ChatMode;
}): AsyncGenerator<{ type: string; data: unknown }> {
  const res = await fetch(`${API}/api/v1/chat/stream`, {
    ...fetchInit,
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(params.token) },
    body: JSON.stringify({
      question: params.question,
      session_id: params.session_id,
      kb_id: params.kb_id ?? DEMO_KB,
      top_k: params.top_k ?? 5,
      mode: params.mode ?? "rag",
    }),
  });
  if (!res.body) throw new Error("SSE: no body");
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n\n");
    buf = lines.pop() || "";
    for (const chunk of lines) {
      const line = chunk.trim();
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      try {
        yield JSON.parse(payload);
      } catch {
        // ignore malformed line
      }
    }
  }
}

// ----- Documents -----
export async function uploadDocument(token: string, file: File, kb_id = DEMO_KB) {
  const fd = new FormData();
  fd.append("file", file);
  return jsonOrThrow<{ doc_id: string; filename: string; chunks: number; status: string; message?: string }>(
    await fetch(`${API}/api/v1/documents/upload?kb_id=${encodeURIComponent(kb_id)}`, {
      ...fetchInit,
      method: "POST",
      headers: authHeaders(token),
      body: fd,
    }),
  );
}

export async function listDocuments(token: string, kb_id = DEMO_KB) {
  return jsonOrThrow<{ kb_id: string; bm25_count: number; vector_count: number }>(
    await fetch(`${API}/api/v1/documents/list?kb_id=${encodeURIComponent(kb_id)}`, {
      ...fetchInit,
      headers: authHeaders(token),
    }),
  );
}

// ----- History -----
export async function listSessions(token: string): Promise<SessionMeta[]> {
  return jsonOrThrow(
    await fetch(`${API}/api/v1/history`, { ...fetchInit, headers: authHeaders(token) }),
  );
}

export async function getSession(token: string, session_id: string): Promise<SessionDetail> {
  return jsonOrThrow(
    await fetch(`${API}/api/v1/history/${session_id}`, { ...fetchInit, headers: authHeaders(token) }),
  );
}

export async function createSession(token: string, payload: { title?: string; kb_id?: string; mode?: string } = {}) {
  return jsonOrThrow<{ session_id: string }>(
    await fetch(`${API}/api/v1/history`, {
      ...fetchInit,
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(token) },
      body: JSON.stringify(payload),
    }),
  );
}

export async function appendMessage(token: string, session_id: string, msg: HistoryMessage) {
  return jsonOrThrow<{ session_id: string; message_count: number }>(
    await fetch(`${API}/api/v1/history/${session_id}/append`, {
      ...fetchInit,
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(token) },
      body: JSON.stringify(msg),
    }),
  );
}

export async function deleteSession(token: string, session_id: string) {
  return jsonOrThrow(
    await fetch(`${API}/api/v1/history/${session_id}`, {
      ...fetchInit,
      method: "DELETE",
      headers: authHeaders(token),
    }),
  );
}

export const API_BASE = API;

// ----- 客户端 token 存储 (与 AuthBar 共享) -----
// 免登录演示: 直接把后端 /auth/dev-token 颁发的 token 存到 localStorage,
// 之后所有请求用 Authorization: Bearer 带上 (跨域下比 HttpOnly cookie 更可靠).
const USER_KEY = "unikb.user";
const TOKEN_KEY = "unikb.token";

export function getStoredToken(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function getStoredUser(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(USER_KEY) || "";
}

export function setStoredToken(token: string, user: string): void {
  if (typeof window === "undefined") return;
  if (token) localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, user);
  window.dispatchEvent(new Event("unikb:auth"));
}

export function clearStoredAuth(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(TOKEN_KEY);
  window.dispatchEvent(new Event("unikb:auth"));
  // cookie 由后端 /auth/logout 清除; 这里也尝试触发一下 (可选)
  fetch(`${API_BASE}/api/v1/auth/logout`, { ...fetchInit, method: "POST" }).catch(() => {});
}

// 免登录: 自动向后端拿 dev token 并缓存, 已缓存则直接返回. 失败返回空串.
export async function ensureDevToken(): Promise<string> {
  if (typeof window === "undefined") return "";
  const existing = localStorage.getItem(TOKEN_KEY);
  if (existing) return existing;
  try {
    const resp = await devToken();
    setStoredToken(resp.access_token, "dev-user");
    return resp.access_token;
  } catch (e) {
    console.error("[unikb] 自动获取 dev token 失败:", e);
    return "";
  }
}
