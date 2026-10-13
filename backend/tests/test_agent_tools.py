# -*- coding: utf-8 -*-
"""Agent 工具的上下文截断行为。

回归背景: `hybrid_search` 早年把每条召回结果写死截断到 300 字符, 比默认
`chunk_size=500` 还短, 导致 chunk 后半段(定价 / 结论这类常放在文末的内容)
在 Agent 链路被静默丢掉 —— 表现为 Agent 模式答"未找到相关信息", 而 RAG
主链路(用全文)答得对, 两条路径行为不一致。
"""
from __future__ import annotations

import pytest

from app.agents import tools as tools_mod
from app.core.config import settings


class _FakeRetriever:
    def __init__(self, doc: str):
        self._doc = doc

    def retrieve(self, query, top_k=5):
        return [{"id": "c0", "document": self._doc, "score": 1.0}]


class _FakeReranker:
    def rerank(self, query, candidates, top_k=5):
        return candidates[:top_k]


@pytest.fixture
def _patch_stores(monkeypatch):
    """把 retriever / reranker 换成假的, 只测截断逻辑本身。"""
    made = {}

    def _get_retriever(kb_id="default"):
        return made["retriever"]

    def _get_reranker():
        return _FakeReranker()

    monkeypatch.setattr(tools_mod, "get_retriever", _get_retriever)
    monkeypatch.setattr(tools_mod, "get_reranker", _get_reranker)
    return made


def _run(doc: str):
    tools = tools_mod.build_tools(kb_id="kb-test")
    hybrid = {t.name: t for t in tools}["hybrid_search"]
    return hybrid.invoke({"query": "任意问题", "top_k": 3})


def test_hybrid_search_keeps_document_tail(_patch_stores):
    """文档末尾的内容必须能被召回上下文带到 (默认配置下)。"""
    # 造一篇 800 字的文档, 把"关键信息"放在最后。
    body = "产品定位与核心能力的说明。" * 30       # 约 420 字
    tail = "企业版年费为 1280 元，年度促销按原价 85% 收费。"
    doc = body + tail

    _patch_stores["retriever"] = _FakeRetriever(doc)
    out = _run(doc)

    assert "1280" in out, "文档尾部的关键信息被截断了"
    assert "85%" in out


def test_hybrid_search_respects_configured_limit(_patch_stores, monkeypatch):
    """截断上限应来自 settings.context_snippet_chars, 而不是写死的常量。"""
    doc = "甲" * 1000 + "乙" * 1000
    _patch_stores["retriever"] = _FakeRetriever(doc)

    monkeypatch.setattr(settings, "context_snippet_chars", 1200, raising=False)
    out = _run(doc)
    assert "乙" in out, "配置放宽后应能看到 1200 字符处的内容"
    # 1200 之后的第二段的确不该出现, 证明上限真的生效了 (而不是干脆不截断)。
    body = out.split(") ", 1)[-1]
    assert len(body) == 1200

    monkeypatch.setattr(settings, "context_snippet_chars", 500, raising=False)
    out = _run(doc)
    assert "乙" not in out, "配置收窄后应重新截断"


def test_hybrid_search_outputs_index_and_score(_patch_stores):
    _patch_stores["retriever"] = _FakeRetriever("简短文档内容。")
    out = _run("简短文档内容。")
    assert out.startswith("[1] ")
    assert "score=" in out


def test_hybrid_search_empty_corpus(_patch_stores):
    class _Empty:
        def retrieve(self, query, top_k=5):
            return []

    _patch_stores["retriever"] = _Empty()
    tools = tools_mod.build_tools(kb_id="kb-test")
    hybrid = {t.name: t for t in tools}["hybrid_search"]
    assert hybrid.invoke({"query": "x", "top_k": 3}) == "未检索到相关文档。"


def test_calculator_and_date_tools_exist():
    names = {t.name for t in tools_mod.build_tools(kb_id="kb-test")}
    assert {"hybrid_search", "calculator", "current_date"} <= names
