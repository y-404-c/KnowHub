"use client";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import {
  Source,
  appendMessage,
  createSession,
  streamChat,
  ensureDevToken,
  DEMO_KB,
} from "@/lib/api";
import Link from "next/link";

type Msg = {
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  /** 独立于 content 的错误信息; 非空时渲染成错误卡片而不是糊在正文里 */
  error?: string;
  /** 正在流式输出(用于显示光标) */
  streaming?: boolean;
};

// 快捷问题: 对齐预置示例文档 (`data/samples/面试演示产品方案.docx`) 的内容,
// 保证开箱即问就能拿到有引用的真实回答, 而不是"未找到相关信息".
// 最后一条会触发 calculator 工具, 顺带演示多 Agent 里的计算分支。
const SUGGESTIONS = [
  "知枢 KnowHub 的产品定位是什么？",
  "它支持上传哪些文档格式？",
  "混合检索是怎么工作的？",
  "企业版买 20 个账号多少钱？",
];

/** 只更新最后一条消息(流式增量写入的目标) */
function patchLast(msgs: Msg[], fn: (m: Msg) => Msg): Msg[] {
  if (msgs.length === 0) return msgs;
  const i = msgs.length - 1;
  return msgs.map((m, idx) => (idx === i ? fn(m) : m));
}

export default function ChatPage() {
  const [token, setToken] = useState("");
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<"rag" | "agent">("rag");
  const [topK, setTopK] = useState(5);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [loading, setLoading] = useState(false);
  const [ready, setReady] = useState(false);
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ensureDevToken().then((t) => {
      setToken(t);
      setReady(true);
    });
  }, []);

  // 新消息到达时滚到底部
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  async function ensureSession() {
    if (sessionId) return sessionId;
    if (!token) return null;
    // 显式带 kb_id: 否则后端会落成 "default", 与实际检索用的知识库不一致,
    // 历史页里看到的库名是错的。
    const r = await createSession(token, { title: "新会话", mode, kb_id: DEMO_KB });
    setSessionId(r.session_id);
    return r.session_id;
  }

  /**
   * 发一轮问答。
   * @param addUser 重新生成时为 false —— 用户消息已在历史里, 只需重跑助手那一轮
   */
  async function runAsk(text: string, addUser: boolean) {
    const trimmed = text.trim();
    if (!trimmed || !token) return;

    const sid = await ensureSession();
    if (!sid) return;

    if (addUser) {
      setMessages((m) => [...m, { role: "user", content: trimmed, sources: [] }]);
    }
    setMessages((m) => [
      ...m,
      { role: "assistant", content: "", sources: [], streaming: true },
    ]);
    setLoading(true);

    if (addUser) {
      try {
        await appendMessage(token, sid, {
          role: "user",
          content: trimmed,
          ts: Date.now() / 1000,
        });
      } catch {
        /* 落库失败不影响对话本身 */
      }
    }

    let collected = "";
    // 用局部变量收集引用: 原实现读 asstMsg.sources, 而那是个初始空数组常量,
    // 流式写回的是 state 里的对象, 结果入库的 sources 永远为空。
    const collectedSources: Source[] = [];
    let errMsg = "";

    try {
      for await (const evt of streamChat({
        token,
        question: trimmed,
        session_id: sid,
        mode,
        top_k: topK,
      })) {
        if (evt.type === "token") {
          const t = (evt.data as any)?.text ?? "";
          collected += t;
          setMessages((m) => patchLast(m, (it) => ({ ...it, content: it.content + t })));
        } else if (evt.type === "source") {
          const src = evt.data as Source;
          collectedSources.push(src);
          setMessages((m) => patchLast(m, (it) => ({ ...it, sources: [...it.sources, src] })));
        } else if (evt.type === "error") {
          errMsg = (evt.data as any)?.msg ?? "服务端返回错误";
        }
      }
    } catch (e: any) {
      errMsg = e?.message ?? String(e);
    } finally {
      setLoading(false);
      setMessages((m) =>
        patchLast(m, (it) => ({
          ...it,
          streaming: false,
          error: errMsg || undefined,
        })),
      );
      // 只有成功产出内容才落库, 避免把错误文本写进历史
      if (!errMsg && collected) {
        try {
          await appendMessage(token, sid, {
            role: "assistant",
            content: collected,
            sources: collectedSources,
            ts: Date.now() / 1000,
          });
        } catch {
          /* ignore */
        }
      }
    }
  }

  function ask() {
    const text = q;
    if (!text.trim()) return;
    setQ("");
    runAsk(text, true);
  }

  /** 重新生成: 回溯到该助手消息之前最近的用户提问, 截断后重跑 */
  function regenerate(idx: number) {
    let question = "";
    for (let i = idx - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        question = messages[i].content;
        break;
      }
    }
    if (!question) return;
    setMessages((m) => m.slice(0, idx));
    runAsk(question, false);
  }

  async function copyText(text: string, idx: number) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIdx(idx);
      setTimeout(() => setCopiedIdx(null), 1500);
    } catch {
      /* 剪贴板不可用(如非安全上下文)时静默失败 */
    }
  }

  function newChat() {
    setSessionId(null);
    setMessages([]);
  }

  // 初始化阶段用骨架屏, 比一行"正在准备"更接近成品观感
  if (!ready) {
    return (
      <div className="chat-page">
        <div className="card">
          <div className="skeleton skeleton-line w-45" />
          <div className="skeleton skeleton-line w-90" />
          <div className="skeleton skeleton-line w-70" />
        </div>
      </div>
    );
  }

  return (
    <div className="chat-page">
      <div className="row chat-toolbar" style={{ justifyContent: "space-between" }}>
        <div className="row">
          <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
            <option value="rag">RAG 模式</option>
            <option value="agent">多 Agent</option>
          </select>
          <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
            {[3, 5, 8, 10].map((n) => (
              <option key={n} value={n}>
                top_k={n}
              </option>
            ))}
          </select>
          <span className="muted mono">session: {sessionId ?? "(新会话)"}</span>
        </div>
        <button className="ghost" onClick={newChat}>
          新会话
        </button>
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div className="chat-hero">
            <div className="hero-badge">Multi-Agent · Hybrid Search · Rerank</div>
            <h1 className="hero-title">
              Hi，我是 <span className="grad">KnowHub</span> 助手
            </h1>
            <p className="hero-sub">
              基于你的知识库完成检索、重排与生成，引用可溯源。免登录，直接开始提问。
            </p>
            <div className="suggs">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="sugg" onClick={() => setQ(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={"bubble " + m.role + (m.error ? " msg-error" : "")}
          >
            <div className="role">{m.role === "user" ? "你" : "助手"}</div>

            <div className="content">
              {m.role === "assistant" ? (
                <div className="md">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[[rehypeHighlight, { ignoreMissing: true }]]}
                    components={{
                      a: ({ ...props }) => (
                        <a {...props} target="_blank" rel="noopener noreferrer" />
                      ),
                    }}
                  >
                    {m.content}
                  </ReactMarkdown>
                  {m.streaming && <span className="stream-cursor" />}
                </div>
              ) : (
                <>
                  {m.content}
                  {m.streaming && <span className="stream-cursor" />}
                </>
              )}
              {/* 流式尚未产出首字时显示三点, 避免看起来"卡住了" */}
              {m.content === "" && !m.error && (
                <span className="typing">
                  <i />
                  <i />
                  <i />
                </span>
              )}
            </div>

            {m.error && (
              <div className="error-box">
                <span className="e-icon">⚠</span>
                <div className="e-body">
                  <div className="e-msg">{m.error}</div>
                  <div className="e-sub">
                    可能是模型服务未配置或网络异常，可重试一次。
                  </div>
                </div>
                <button
                  className="ghost"
                  style={{ padding: "5px 12px", fontSize: 12 }}
                  onClick={() => regenerate(i)}
                >
                  重试
                </button>
              </div>
            )}

            {m.sources.length > 0 && (
              <details className="sources">
                <summary>引用 {m.sources.length} 条</summary>
                {m.sources.map((s, j) => (
                  <div key={j} className="source">
                    <span className="score">{Number(s.score).toFixed(3)}</span>{" "}
                    <span className="text">{s.content}</span>
                  </div>
                ))}
              </details>
            )}

            {/* 消息操作: 悬停显形 */}
            {!m.streaming && m.content && (
              <div className="msg-actions">
                <button
                  className={"icon-btn" + (copiedIdx === i ? " copied" : "")}
                  title="复制"
                  onClick={() => copyText(m.content, i)}
                >
                  {copiedIdx === i ? "✓" : "⧉"}
                </button>
                {m.role === "assistant" && (
                  <button
                    className="icon-btn"
                    title="重新生成"
                    disabled={loading}
                    onClick={() => regenerate(i)}
                  >
                    ⟳
                  </button>
                )}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="composer">
        <textarea
          rows={2}
          placeholder="输入问题..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
          }}
        />
        <button onClick={ask} disabled={loading || !q.trim()}>
          {loading ? (
            <>
              <span className="spinner" />
              生成中
            </>
          ) : (
            "发送"
          )}
        </button>
      </div>

      <div className="muted" style={{ marginTop: 8, fontSize: 11 }}>
        提示: ⌘/Ctrl + Enter 发送。历史会话在 <Link href="/history">History</Link>。
      </div>
    </div>
  );
}
