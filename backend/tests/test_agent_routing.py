"""测试 LangGraph Agent 节点 + 图回环的新行为.

覆盖:
1. _parse_planner_decision 容错 (纯 JSON / 代码块 / 混杂 / fallback).
2. planner_node 把结构化决策写到 trace.
3. retriever_node 根据 needs_calculator 走 calculator, needs_retrieval 走 hybrid_search.
4. retriever_node 接受 state['kb_id'] 而非默认 "default".
5. build_agent_graph() 用的是 add_conditional_edges 而不是 add_edge.
6. route_review 在 pass=False / retries < MAX 时选 "retriever", 否则 END.
7. 完整 graph.invoke 端到端: 通过图回环后, retry_count 会递增.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

# LangGraph 的 END 常量现在等于 "__end__", 跟字面量 "END" 不一样 -- 直接 import 用.
from app.agents.graph import (
    MAX_REVIEWER_RETRIES,
    AgentState,
    _parse_planner_decision,
    build_agent_graph,
    planner_node,
    retriever_node,
    reviewer_node,
    route_review,
)

# ---------- _parse_planner_decision 解析 ----------

class TestParsePlannerDecision:
    def test_pure_json(self):
        raw = '{"needs_calculator": true, "needs_retrieval": false, "needs_date": false, "keywords": "1+1"}'
        d = _parse_planner_decision(raw)
        assert d["needs_calculator"] is True
        assert d["needs_retrieval"] is False
        assert d["keywords"] == "1+1"

    def test_code_block(self):
        raw = '```json\n{"needs_calculator": true, "needs_retrieval": true, "needs_date": false, "keywords": "k"}\n```'
        d = _parse_planner_decision(raw)
        assert d["needs_calculator"] is True
        assert d["needs_retrieval"] is True

    def test_string_bool(self):
        d = _parse_planner_decision('{"needs_calculator": "yes", "needs_retrieval": "0"}')
        assert d["needs_calculator"] is True
        assert d["needs_retrieval"] is False

    def test_unparseable_fallback(self):
        d = _parse_planner_decision("LLM 输出乱码")
        # 默认 fallback: needs_retrieval=True (默认走 hybrid_search), 其它 False.
        assert d["needs_retrieval"] is True
        assert d["needs_calculator"] is False


# ---------- planner / retriever 节点契约 ----------

class _PlannerFakeLLM:
    def __init__(self, raw):
        self._raw = raw

    def invoke(self, _msgs):
        return AIMessage(content=self._raw)


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "messages": [HumanMessage(content="123*456是多少?")],
        "kb_id": "default",
        "plan": "",
        "retrieved": "",
        "draft": "",
        "final": "",
        "trace": [],
        "retry_count": 0,
        "last_reviewer_reason": "",
    }
    state.update(overrides)
    return state


def test_planner_node_parses_decision(monkeypatch):
    monkeypatch.setattr(
        "app.agents.graph.get_llm",
        lambda: _PlannerFakeLLM(
            json.dumps({
                "needs_calculator": True,
                "needs_retrieval": False,
                "needs_date": False,
                "keywords": "123*456",
            })
        ),
    )
    state = _base_state()
    out = planner_node(state)
    planner_traces = [t for t in out["trace"] if t["role"] == "planner"]
    assert planner_traces, "planner 节点必须写 trace"
    decision = planner_traces[0]["content"]["decision"]
    assert decision["needs_calculator"] is True
    assert decision["needs_retrieval"] is False


def test_retriever_node_invokes_calculator_for_math_question(monkeypatch):
    """当 planner 决定 needs_calculator=true 时, retriever_node 真的会调 calculator."""
    state = _base_state()
    # 先跑一次 planner_node 把 decision 写进 trace.
    monkeypatch.setattr(
        "app.agents.graph.get_llm",
        lambda: _PlannerFakeLLM(
            json.dumps({
                "needs_calculator": True,
                "needs_retrieval": False,
                "needs_date": False,
                "keywords": "",
            })
        ),
    )
    state = planner_node(state)
    # 现在跑 retriever_node, 它会从 trace 里读 planner decision.
    # monkey-patch build_tools 让 calculator / hybrid_search 都是 stub, 验证调用.
    calc_called = {"n": 0}
    hybrid_called = {"n": 0}

    def _calc_stub(expression: str) -> str:
        calc_called["n"] += 1
        return f"calc({expression})=56088"

    def _hybrid_stub(query: str, top_k: int = 5) -> str:
        hybrid_called["n"] += 1
        return f"hybrid({query})"

    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        "calc"
        return _calc_stub(expression)

    @tool
    def hybrid_search(query: str, top_k: int = 5) -> str:
        "hybrid"
        return _hybrid_stub(query, top_k)

    monkeypatch.setattr(
        "app.agents.graph.build_tools",
        lambda kb_id="default": [hybrid_search, calculator],
    )

    out = retriever_node(state)
    assert calc_called["n"] >= 1, "calculator 工具必须被调用"
    # hybrid 不应该被调 (因为 needs_retrieval=False).
    assert hybrid_called["n"] == 0
    # trace 里有 calculator 记录.
    assert any(t["role"] == "tool:calculator" for t in out["trace"])


def test_retriever_node_invokes_hybrid_by_default(monkeypatch):
    """默认 (fallback 决策) 下, retriever_node 调 hybrid_search."""
    state = _base_state()
    # 不跑 planner, 让 retriever_node 自己解析空字符串 (走 fallback needs_retrieval=True).
    hybrid_called = {"n": 0}

    def _hybrid_stub(query: str, top_k: int = 5) -> str:
        hybrid_called["n"] += 1
        return "ctx"

    from langchain_core.tools import tool

    @tool
    def hybrid_search(query: str, top_k: int = 5) -> str:
        "hybrid"
        return _hybrid_stub(query, top_k)

    monkeypatch.setattr(
        "app.agents.graph.build_tools",
        lambda kb_id="default": [hybrid_search],
    )
    retriever_node(state)
    assert hybrid_called["n"] == 1


# ---------- retriever_node 真的把 kb_id 传进去 ----------

def test_retriever_node_propagates_kb_id(monkeypatch):
    """retriever_node 必须把 state['kb_id'] 传给 build_tools(kb_id=...)."""
    captured = {"kb_id": None}

    def _fake_build_tools(kb_id="default"):
        captured["kb_id"] = kb_id
        from langchain_core.tools import tool

        @tool
        def hybrid_search(query: str, top_k: int = 5) -> str:
            "stub"
            return "ctx"

        return [hybrid_search]

    monkeypatch.setattr("app.agents.graph.build_tools", _fake_build_tools)
    state = _base_state(kb_id="alice_private_kb")
    # trace 里给一个 planner decision 让 retriever 走 hybrid_search 路径.
    state["trace"].append({
        "role": "planner",
        "content": {
            "raw": "{}",
            "decision": {
                "needs_calculator": False,
                "needs_retrieval": True,
                "needs_date": False,
                "keywords": "x",
            },
        },
    })
    retriever_node(state)
    assert captured["kb_id"] == "alice_private_kb", (
        f"retriever_node 没把 kb_id 传给 build_tools, 实际 {captured['kb_id']!r}"
    )


# ---------- reviewer_node + route_review ----------

class _ReviewerFakeLLM:
    """LLM 第一次返回 pass, 第二次返回 retry; 每次 invoke 返回对应 raw."""

    def __init__(self, raw):
        self._raw = raw

    def invoke(self, _msgs):
        return AIMessage(content=self._raw)


def test_route_review_pass_returns_END_label(monkeypatch):
    """pass=True 时返回 "pass" 标签 (path_map 会映射到 END)."""
    state = _base_state()
    state["trace"].append({
        "role": "reviewer",
        "content": {"decision": {"pass": True, "reason": "ok"}, "raw": ""},
    })
    assert route_review(state) == "pass"


def test_route_review_retry_within_limit_returns_retry_label(monkeypatch):
    """pass=False 且 retry_count 未达上限 -> "retry".

    Regression: route_review 必须是**纯函数**。之前它在这里顺手递增 retry_count,
    但 conditional edge 的副作用不会写回 checkpoint, 导致 MAX_REVIEWER_RETRIES
    在真实图执行中永不生效 (Reviewer 持续判 false 会无限回环, 直到
    GraphRecursionError)。计数递增已移到 reviewer_node。
    """
    state = _base_state(retry_count=0)
    state["trace"].append({
        "role": "reviewer",
        "content": {"decision": {"pass": False, "reason": "需要引用"}, "raw": ""},
    })
    assert route_review(state) == "retry"
    assert state["retry_count"] == 0, "route_review 不得修改 state (branch 副作用不落盘)"


def test_route_review_retry_exhausted_returns_pass_label():
    """retry_count 达上限 -> "pass" (结束)。

    exhausted 标记由 reviewer_node 写进 final, edge 只负责路由。
    """
    state = _base_state(retry_count=MAX_REVIEWER_RETRIES)
    state["trace"].append({
        "role": "reviewer",
        "content": {"decision": {"pass": False, "reason": "还是不行"}, "raw": ""},
    })
    assert route_review(state) == "pass"
    assert state["final"] == "", "route_review 不得修改 final"


def test_reviewer_node_increments_retry_count(monkeypatch):
    """reviewer_node 在 pass=false 时递增 retry_count (唯一可靠的写入点)."""
    state = _base_state(retry_count=0, draft="草稿 A")
    monkeypatch.setattr(
        "app.agents.graph.get_llm",
        lambda: _ReviewerFakeLLM('{"pass": false, "reason": "缺少引用"}'),
    )
    out = reviewer_node(state)
    assert out["retry_count"] == 1, "pass=false 必须递增 retry_count"
    assert out["last_reviewer_reason"] == "缺少引用"
    # 未达上限时 final 保持 draft, 不加 exhausted 标记
    assert out["final"] == "草稿 A"


def test_reviewer_node_does_not_increment_on_pass(monkeypatch):
    """pass=true 时不递增计数."""
    state = _base_state(retry_count=1, draft="草稿 B")
    monkeypatch.setattr(
        "app.agents.graph.get_llm",
        lambda: _ReviewerFakeLLM('{"pass": true, "reason": "ok"}'),
    )
    out = reviewer_node(state)
    assert out["retry_count"] == 1


def test_reviewer_node_marks_exhausted_when_limit_reached(monkeypatch):
    """达到 MAX_REVIEWER_RETRIES 时, final 上追加 exhausted 标记."""
    state = _base_state(retry_count=MAX_REVIEWER_RETRIES - 1, draft="草稿 C")
    monkeypatch.setattr(
        "app.agents.graph.get_llm",
        lambda: _ReviewerFakeLLM('{"pass": false, "reason": "还是不行"}'),
    )
    out = reviewer_node(state)
    assert out["retry_count"] == MAX_REVIEWER_RETRIES
    assert "reviewer-retry-exhausted" in out["final"], (
        f"final 应标记 exhausted, 实际 {out['final']!r}"
    )


def test_agent_graph_terminates_when_reviewer_always_rejects(monkeypatch):
    """端到端回归: Reviewer 永远判 false 时, 图必须在 MAX_REVIEWER_RETRIES 内终止.

    这是本文件最重要的测试。修复前 reviewer 会被调用 13 次直到 LangGraph 的
    recursion_limit 熔断抛 GraphRecursionError (一次请求烧掉 26+ 次 LLM 调用)。
    recursion_limit 设得足够大, 确保防死循环一旦再次失效, 这里必然失败。
    """
    from langchain_core.tools import tool as _tool

    planner_json = json.dumps({
        "needs_calculator": False,
        "needs_retrieval": True,
        "needs_date": False,
        "keywords": "q",
    })
    reject = json.dumps({"pass": False, "reason": "永远不通过"})
    calls = {"reviewer": 0}

    class RouterLLM:
        def invoke(self, msgs):
            head = msgs[0].content if msgs else ""
            if "任务规划" in head:
                return AIMessage(content=planner_json)
            if "质量审查" in head:
                calls["reviewer"] += 1
                return AIMessage(content=reject)
            return AIMessage(content="草稿答案")

    monkeypatch.setattr("app.agents.graph.get_llm", lambda: RouterLLM())

    @_tool
    def hybrid_search(query: str, top_k: int = 5) -> str:
        "stub"
        return "ctx: relevant info"

    monkeypatch.setattr(
        "app.agents.graph.build_tools", lambda kb_id="default": [hybrid_search]
    )

    graph = build_agent_graph()
    out = graph.invoke(
        {
            "messages": [HumanMessage(content="q")],
            "kb_id": "default",
            "plan": "", "retrieved": "", "draft": "", "final": "",
            "trace": [], "retry_count": 0, "last_reviewer_reason": "",
        },
        config={"recursion_limit": 40},
    )
    assert calls["reviewer"] == MAX_REVIEWER_RETRIES, (
        f"reviewer 应恰好调用 {MAX_REVIEWER_RETRIES} 次, 实际 {calls['reviewer']} 次 "
        f"(防死循环失效时会远大于此值)"
    )
    assert out["retry_count"] == MAX_REVIEWER_RETRIES
    assert "reviewer-retry-exhausted" in (out.get("final") or "")


def test_route_review_missing_decision_returns_pass_label():
    state = _base_state()
    # trace 里没有 reviewer 记录 -> 安全结束.
    assert route_review(state) == "pass"


# ---------- build_agent_graph 是真 conditional edge ----------

def test_build_agent_graph_uses_conditional_edges(monkeypatch):
    """直接构造图, 跑一次端到端: planner -> retriever -> coder -> reviewer -> END."""
    monkeypatch.setattr(
        "app.agents.graph.get_llm",
        lambda: _PlannerFakeLLM(json.dumps({
            "needs_calculator": False,
            "needs_retrieval": True,
            "needs_date": False,
            "keywords": "q",
        })),
    )

    captured = {"kb_id": None}

    def _fake_build_tools(kb_id="default"):
        captured["kb_id"] = kb_id
        from langchain_core.tools import tool

        @tool
        def hybrid_search(query: str, top_k: int = 5) -> str:
            "stub"
            return "ctx: relevant info"

        @tool
        def calculator(expression: str) -> str:
            "stub"
            return "0"

        return [hybrid_search, calculator]

    monkeypatch.setattr("app.agents.graph.build_tools", _fake_build_tools)

    g = build_agent_graph()
    out = g.invoke({
        "messages": [HumanMessage(content="q")],
        "kb_id": "my_kb",
        "plan": "",
        "retrieved": "",
        "draft": "",
        "final": "",
        "trace": [],
        "retry_count": 0,
        "last_reviewer_reason": "",
    })
    # kb_id 必须传递到 build_tools.
    assert captured["kb_id"] == "my_kb"
    # trace 至少包含 4 个节点 (planner / retriever / coder / reviewer).
    roles = [t["role"] for t in out["trace"]]
    assert "planner" in roles
    assert "retriever" in roles
    assert "coder" in roles
    assert "reviewer" in roles