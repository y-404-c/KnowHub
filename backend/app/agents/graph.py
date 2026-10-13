# -*- coding: utf-8 -*-
"""LangGraph 多 Agent 协作：Planner -> Retriever -> Coder -> Reviewer.

Reviewer 节点会以显式条件边判定 pass / retry:

    Planner -> Retriever -> Coder -> Reviewer -> [retry -> Retriever] -> ... -> END

`add_conditional_edges("reviewer", route_review, {"retry": "retriever", "pass": END})`
是真正的图回环, 不是 reviewer 节点里“就地调用 LLM 改写”。同时有最大重试次数
(`MAX_REVIEWER_RETRIES`, 默认 2) 防止死循环。

`kb_id` 通过 `AgentState` 显式传入, retriever 节点用 `state["kb_id"]` 调
`build_tools(kb_id=...)`, 不再默认走 "default", 配合 kb_registry 的 ACL。

`AgentState` 里还新增了 `retry_count`, 在图回环时单调累加, 超过阈值强制
走到 END (带 reason), 即使 LLM 一直判 retry 也不会无限循环。

注: 当前实现 **没有** 接入 `interrupt_before` 的 human-in-the-loop -- 这
条留在 docs/why-this-stack.md 之外的 Roadmap, 等真有人审需要时再接。
"""
from __future__ import annotations
import json
import re
from functools import lru_cache
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.agents.llm_router import get_llm
from app.agents.tools import build_tools
from app.core.config import settings
from app.core.text_utils import strip_think


# 最大 reviewer 回环次数 (这是 retry 总数, 含第 0 次首跑). 超过强制结束.
MAX_REVIEWER_RETRIES = 2


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "对话消息"]
    kb_id: Annotated[str, "目标知识库 ID; 配合 kb_registry 的 ACL"]
    plan: str
    retrieved: str
    draft: str
    final: str
    trace: List[dict]
    # retry 计数: reviewer 决定 retry 时递增, 超过 MAX_REVIEWER_RETRIES 强制 END.
    retry_count: int
    # 最后一次 reviewer 决策, 便于终止时返回原因.
    last_reviewer_reason: str


PLANNER_SYS = (
    "你是一个任务规划 Agent。给定用户问题，必须严格按以下 JSON 格式输出, 不要输出任何额外文字或 markdown 代码块:\n"
    '{"needs_calculator": true|false, "needs_retrieval": true|false, "needs_date": true|false, "keywords": "<检索关键词, 短句>"}\n'
    "判定规则:\n"
    "- 涉及算术/公式/数学函数 -> needs_calculator=true.\n"
    "- 涉及当前日期/时间 -> needs_date=true.\n"
    "- 需要查文档才能回答 -> needs_retrieval=true.\n"
    "- keywords 给 retriever 用, 短句即可 (例如 \"保修期 退换货\").\n"
)

CODER_SYS = "你是回答生成 Agent。基于【检索结果】直接回答用户问题。\n要求：\n- 用中文\n- 简洁、结构化\n- 必须引用检索结果中标注的 [1] [2] 等\n- 不确定就说「未找到相关信息」\n"

# Reviewer 现在要求严格 JSON 输出, 避免以前用 `if "通过" in out.content`
# 字符串匹配被误判 (例如 "审核通过率" 这种 token 也会触发).
REVIEWER_SYS = (
    "你是质量审查 Agent。检查【草稿】是否：\n"
    "1. 直接回答了用户问题\n"
    "2. 是否引用了检索结果\n"
    "3. 是否有明显事实错误\n"
    "请只输出一个 JSON 对象 (不要输出任何额外文字 / markdown 代码块):\n"
    '{"pass": true|false, "reason": "如果 pass=false 简要说明原因, 不超过 60 字"}\n'
    "如果合格, pass 必须是 true; 否则 false."
)


def _append_trace(state, role, content):
    trace = state.get("trace", [])
    trace.append({"role": role, "content": content})
    return trace


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\"pass\"[^{}]*\"reason\"[^{}]*\}", re.DOTALL)


def _parse_reviewer_decision(raw: str) -> dict:
    """从 LLM 输出里挑出 JSON 决策.

    容错策略:
    1. 尝试整体 json.loads.
    2. 失败 -> 去掉 ```json ... ``` 代码块再 parse.
    3. 再失败 -> 正则抓第一个 {"pass": ...} 块.
    4. 全失败 -> fallback {"pass": True, "reason": "(parse failed, 默认通过)"},
       避免 LLM 输出格式问题导致整条链路死循环 (reviewer 不通过会触发重写).
    """
    s = (raw or "").strip()
    if not s:
        return {"pass": True, "reason": "(empty reviewer output)"}
    # 1) 直接
    try:
        d = json.loads(s)
        if isinstance(d, dict) and "pass" in d:
            return _normalize_decision(d)
    except Exception:
        pass
    # 2) 去掉代码块
    if "```" in s:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(1))
                if isinstance(d, dict) and "pass" in d:
                    return _normalize_decision(d)
            except Exception:
                pass
    # 3) 正则抓 {"pass":..., "reason":...}
    m = _JSON_BLOCK_RE.search(s)
    if m:
        try:
            d = json.loads(m.group(0))
            return _normalize_decision(d)
        except Exception:
            pass
    # 4) fallback
    return {"pass": True, "reason": "(unparseable reviewer output, 默认通过)"}


def _normalize_decision(d: dict) -> dict:
    """把 LLM 输出 normalize 成统一 schema."""
    p = d.get("pass")
    # 兼容 true/false/"true"/"false"/1/0/"yes" 等
    if isinstance(p, str):
        pl = p.strip().lower()
        p = pl in ("true", "yes", "1", "pass", "通过")
    elif isinstance(p, int):
        p = bool(p)
    else:
        p = bool(p)
    reason = str(d.get("reason") or "").strip()
    if len(reason) > 200:
        reason = reason[:200]
    return {"pass": p, "reason": reason}


# ---------- planner 结构化解析 (与 reviewer 解析复用同一套容错思路) ----------

_PLANNER_BLOCK_RE = re.compile(r"\{[^{}]*\"needs_calculator\"[^{}]*\}", re.DOTALL)


def _parse_planner_decision(raw: str) -> dict:
    """解析 Planner 输出: {"needs_calculator", "needs_retrieval", "needs_date", "keywords"}.


    容错策略同 _parse_reviewer_decision:
    1. 整体 json.loads.
    2. 去掉 ```json ... ``` 代码块再 parse.
    3. 正则抓 {"needs_calculator": ...} 块.
    4. fallback 都 False / keywords="" (默认走 retrieval + 不用其他工具).

    Returns: dict, 键齐全, 都是合法类型.
    """
    s = (raw or "").strip()
    fallback = {
        "needs_calculator": False,
        "needs_retrieval": True,
        "needs_date": False,
        "keywords": "",
    }
    if not s:
        return {**fallback, "_fallback_reason": "empty planner output"}
    # 1) 直接
    try:
        d = json.loads(s)
        if isinstance(d, dict) and (
            "needs_calculator" in d or "needs_retrieval" in d or "keywords" in d
        ):
            return _normalize_planner(d)
    except Exception:
        pass
    # 2) 去掉代码块
    if "```" in s:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(1))
                if isinstance(d, dict):
                    return _normalize_planner(d)
            except Exception:
                pass
    # 3) 正则抓首块
    m = _PLANNER_BLOCK_RE.search(s)
    if m:
        try:
            d = json.loads(m.group(0))
            return _normalize_planner(d)
        except Exception:
            pass
    # 4) fallback
    return {**fallback, "_fallback_reason": "unparseable planner output"}


def _normalize_planner(d: dict) -> dict:
    def _b(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1", "y", "是")
        return False

    kw = str(d.get("keywords") or "").strip()
    if len(kw) > 200:
        kw = kw[:200]
    return {
        "needs_calculator": _b(d.get("needs_calculator")),
        "needs_retrieval": _b(d.get("needs_retrieval", True)),
        "needs_date": _b(d.get("needs_date")),
        "keywords": kw,
    }


# ---------- 节点实现 ----------

def planner_node(state):
    llm = get_llm()
    msgs = [SystemMessage(content=PLANNER_SYS), state["messages"][-1]]
    out = llm.invoke(msgs)
    # 推理模型可能把 <think>...</think> 思考过程混进 content, 会让下面的 JSON
    # 解析直接失败 (表现为"永远走不到工具分支"), 所以先剥掉再解析.
    raw = strip_think(out.content if hasattr(out, "content") else str(out))
    decision = _parse_planner_decision(raw)
    state["plan"] = raw
    state["trace"] = _append_trace(
        state, "planner",
        {"raw": raw, "decision": decision},
    )
    return state


def _extract_calc_expression(text: str) -> str:
    """从用户问句里尽力抠出算术表达式. 抠不到就把整句交给 calculator,
    让 calculator 报错也能在 trace 留下调用记录."""
    import re as _re
    m = _re.search(r"[\d\.\(\)\+\-\*/\s]+", text or "")
    expr = (m.group(0).strip() if m else (text or "")).strip()
    return expr or "0"


def retriever_node(state):
    """根据 planner 结构化决策路由.

    needs_calculator -> 真的去 invoke `calculator` LangChain 工具, 让 LLM 之外
                        的 safe_eval 实际执行, 避免长链 LLM 算错.
    needs_date       -> invoke `current_date` 工具.
    needs_retrieval  -> invoke `hybrid_search` (主路径).

    这一步同时负责把 `kb_id` 传给 build_tools, 走默认 kb 的 bug 已修复.
    """
    # 从 trace 末条拿解析后的 decision (planner_node 已经写过).
    trace = state.get("trace") or []
    decision = {}
    for t in reversed(trace):
        if t.get("role") == "planner" and isinstance(t.get("content"), dict):
            decision = t["content"].get("decision") or {}
            break
    if not decision:
        # graph 直接 invoke 时 trace 里还没有 planner (比如测试从 retriever_node
        # 直接调起), 用 plan 字符串再 parse 一次.
        decision = _parse_planner_decision(state.get("plan") or "")

    kb_id = state.get("kb_id") or "default"
    tools = build_tools(kb_id=kb_id)
    tool_map = {t.name: t for t in tools}

    parts = []
    if decision.get("needs_calculator"):
        calc = tool_map.get("calculator")
        if calc is not None:
            user_q = state["messages"][-1].content
            expression = _extract_calc_expression(user_q)
            calc_text = calc.invoke(expression)
            parts.append(f"calculator: {calc_text}")
            state["trace"] = _append_trace(state, "tool:calculator", calc_text)

    if decision.get("needs_date"):
        date_t = tool_map.get("current_date")
        if date_t is not None:
            date_text = date_t.invoke({})
            parts.append(f"current_date: {date_text}")
            state["trace"] = _append_trace(state, "tool:current_date", date_text)

    if decision.get("needs_retrieval", True):
        hybrid = tool_map.get("hybrid_search")
        if hybrid is not None:
            keywords = (decision.get("keywords") or state["messages"][-1].content)[:512]
            # 用配置的 top_k_final 而不是写死 5, 否则 RAG 主链路调了参数、
            # Agent 链路依旧固定召回 5 条, 两条路径行为不一致.
            res = hybrid.invoke({"query": keywords, "top_k": settings.top_k_final})
            parts.append(res)
            snippet = res[:500] if isinstance(res, str) else str(res)[:500]
            state["trace"] = _append_trace(state, "retriever", snippet)
    elif not parts:
        # 完全不要检索也不要工具, 给个空提示, 防止 coder 拿到空 context.
        parts.append("(no tool selected)")

    state["retrieved"] = "\n\n".join(parts)
    return state


def coder_node(state):
    llm = get_llm()
    user_q = state["messages"][-1].content
    retrieved = state.get("retrieved", "")
    prompt = "用户问题：" + user_q + "\n\n检索/工具结果：\n" + retrieved + "\n\n请基于上述结果给出最终回答。"
    msgs = [SystemMessage(content=CODER_SYS), HumanMessage(content=prompt)]
    out = llm.invoke(msgs)
    # 剥掉思考块, 否则最终回答正文会带上模型的英文独白.
    state["draft"] = strip_think(out.content if hasattr(out, "content") else str(out))
    state["trace"] = _append_trace(state, "coder", state["draft"])
    return state


def reviewer_node(state):
    """Reviewer 节点: 解析 JSON 决策, 递增 retry_count, 写到 trace.

    关键: **retry 计数必须在节点里递增**, 不能放在 `route_review` 里.

    LangGraph 只持久化"节点返回值"构成的 state delta; conditional edge (branch)
    函数对 state 的就地修改属于未定义行为, 不会被写回 checkpoint. 之前把
    `state["retry_count"] = retries + 1` 写在 route_review 里, 单测直接调函数能
    看到 dict 被改, 但真实图执行中该写入被丢弃 -> retry_count 恒为 0 ->
    `MAX_REVIEWER_RETRIES` 判据永远不成立 -> Reviewer 一旦持续判 false 就会
    无限回环, 直到 LangGraph 的 recursion_limit 熔断 (实测 reviewer 被调用 13 次,
    一次请求烧掉 26+ 次 LLM 调用后抛 500).
    """
    llm = get_llm()
    user_q = state["messages"][-1].content
    draft = state.get("draft", "")
    prompt = "用户问题：" + user_q + "\n\n草稿：\n" + draft + "\n\n请审查并按 JSON 格式输出决策."
    msgs = [SystemMessage(content=REVIEWER_SYS), HumanMessage(content=prompt)]
    out = llm.invoke(msgs)
    # 同 planner: reviewer 要输出严格 JSON, 思考块会污染解析.
    raw = strip_think(out.content if hasattr(out, "content") else str(out))
    decision = _parse_reviewer_decision(raw)

    passed = bool(decision.get("pass"))
    retries = int(state.get("retry_count", 0) or 0)
    if not passed:
        retries += 1
    state["retry_count"] = retries
    # 最终输出默认等于 draft, 由 route_review 决定要不要回到 retriever.
    state["final"] = draft
    state["last_reviewer_reason"] = decision.get("reason", "")

    # 重试次数耗尽且仍未通过: 把理由拼进 final, 让用户看得见.
    # 这段同样必须在节点内完成, 写在 edge 里的版本会被丢弃.
    if not passed and retries >= MAX_REVIEWER_RETRIES:
        reason = decision.get("reason") or "reviewer reached max retries"
        state["final"] = (draft + "\n\n[reviewer-retry-exhausted] " + reason).strip()

    # trace 同时记录 raw + parsed, 方便后续审计 LLM 决策是否被 parse.
    state["trace"] = _append_trace(
        state, "reviewer",
        {
            "raw": raw,
            "decision": decision,
            "retry_count": retries,
        },
    )
    return state


def route_review(state) -> str:
    """Conditional edge: 返回 "pass" / "retry" 两个标签, 由 add_conditional_edges 的
    path_map 映射到目标节点 (END / "retriever").

    **纯函数**: 只读 state, 不做任何写操作 -- branch 函数的副作用不会落盘,
    在这里改 state 是无效的 (见 reviewer_node 的 docstring). retry_count 已由
    reviewer_node 递增并持久化为 state delta.

    - pass=True 或 reviewer 决策缺失: 返回 "pass" -> END.
    - pass=False 且 retry_count < MAX_REVIEWER_RETRIES: 返回 "retry" -> "retriever".
    - 超过阈值: 返回 "pass" -> END (final 上的 exhausted 标记由 reviewer_node 写入).
    """
    trace = state.get("trace") or []
    decision = None
    for t in reversed(trace):
        if t.get("role") == "reviewer" and isinstance(t.get("content"), dict):
            decision = t["content"].get("decision") or {}
            break
    if not decision:
        # 拿不到 reviewer 决策, 安全起见结束.
        return "pass"
    if decision.get("pass"):
        return "pass"
    if int(state.get("retry_count", 0) or 0) >= MAX_REVIEWER_RETRIES:
        return "pass"
    return "retry"


def new_agent_state(question: str, kb_id: str = "default") -> AgentState:
    """构造一次 agent 运行的初始 state.

    抽成工厂是为了消除调用点上的复制粘贴 (chat 两处 + pipeline 一处各写了一遍
    这个 9 键 dict)。TypedDict 不提供运行时默认值, 少一个键 LangGraph 就会在
    节点读 `state["xxx"]` 时抛 KeyError —— 新增 state 字段时如果漏改某个调用点,
    只有在真正跑到那条链路时才会炸。集中在工厂里, 改一次即可。

    Args:
        question: 用户问题, 包成 HumanMessage 放进 messages.
        kb_id: 目标知识库; 必须显式传, 否则 retriever_node 会静默走 "default",
               架空私有 kb 的 ACL (P0-2 的历史 bug).
    """
    return AgentState(
        messages=[HumanMessage(content=question)],
        kb_id=kb_id or "default",
        plan="",
        retrieved="",
        draft="",
        final="",
        trace=[],
        retry_count=0,
        last_reviewer_reason="",
    )


def build_agent_graph():
    """获取编译好的 agent 图 (进程内复用, 不重复编译).

    为什么可以缓存:
        图的结构完全静态 —— 节点函数、边、条件边都是模块级常量, 不依赖任何
        请求参数; 一次运行的全部可变数据都通过 `invoke(state)` / `stream(state)`
        传入。编译产物 (Pregel) 本身是不可变的计算图描述, 并发 invoke 之间
        不共享可变状态, 所以跨请求复用是安全的。

    为什么要缓存:
        `StateGraph.compile()` 会做 schema 校验、channel 构建、节点/边注册,
        虽然远不如加载一个 CrossEncoder 昂贵, 但它跟请求量成正比地重复发生,
        且拿不到任何新信息 —— 属于纯浪费。原先 chat / chat_stream / pipeline
        三个调用点每次请求各编译一次。

    保留成 (而不是直接导出 `_compile_agent_graph`) 的理由:
        测试需要 monkeypatch 整个函数替换成 fake graph
        (见 tests/test_integration_retrieval.py::test_run_rag_agent_mode_invokes_graph)。

    注意 (演进约束):
        一旦给这张图接上 checkpointer (MemorySaver / SqliteSaver) 做多轮记忆,
        编译产物就会持有会话状态, 共享实例会造成**跨用户会话串扰**。届时必须
        改成 per-session 编译, 或改走 LangGraph 官方的 thread_id 隔离。现阶段
        无 checkpointer, 共享是安全的。
    """
    return _compile_agent_graph()


@lru_cache(maxsize=1)
def _compile_agent_graph():
    """真正建图 + compile 的地方, 被 `build_agent_graph` 缓存包装."""
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("retriever", retriever_node)
    g.add_node("coder", coder_node)
    g.add_node("reviewer", reviewer_node)
    g.set_entry_point("planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "coder")
    g.add_edge("coder", "reviewer")
    # 显式 conditional edge: route_review 返回 "pass" -> END, "retry" -> retriever.
    # 这是真正的图回环, 配合 MAX_REVIEWER_RETRIES 防死循环.
    g.add_conditional_edges(
        "reviewer",
        route_review,
        {"retry": "retriever", "pass": END},
    )
    return g.compile()


def reset_agent_graph_cache() -> None:
    """仅供测试: 让下一次 build_agent_graph() 重新编译.

    用于给节点函数打 monkeypatch 之后强制重建图 (已编译的图持有的是编译那一刻
    的函数对象引用, patch 模块属性不会影响已编译的图 —— 注意 `build_tools`
    是个例外, 它在 retriever_node 函数体内按名字查找, patch 后即时生效)。
    """
    _compile_agent_graph.cache_clear()
