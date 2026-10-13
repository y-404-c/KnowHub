# -*- coding: utf-8 -*-
"""RAG 主链路的上下文截断上限 (`settings.context_snippet_chars`)。

回归背景: `_build_prompt` 早年把每条上下文写死截断到 600 字符, 比默认
`chunk_size=500` / `adaptive_chunk_size=5500` 都短。长 chunk 的文末内容
(这篇示例文档恰好把"定价规则"放在最后) 会被静默切掉, 模型看到半句话后
不会报错, 而是在回答里含糊地说"原文此处似乎不完整 / 被截断" —— 排查时
链路全是 200 OK, 极易误判成模型能力问题。

三条路径 (API 主链路 / 评估链路 / Agent 工具) 必须共用同一个上限,
否则"同一份文档, RAG 答得出来、Agent 答不出来"。
"""
from __future__ import annotations

import pytest

from app.api.chat import _build_prompt
from app.core.config import settings


LONG_DOC = "产品定位与核心能力的详细说明。" * 60      # 约 900 字
PRICING_TAIL = "企业版年费为 1280 元，年度促销按原价 85% 收费，20 个账号按折后单价乘以数量。"


def _contexts(doc):
    return [{"document": doc, "id": "c0", "score": 1.0}]


def _ctx_body(prompt: str) -> str:
    """取出 prompt 里【检索结果】到【用户问题】之间的正文。
    注意不能 split("[1] ") —— 系统提示词里也写了 "[1] [2] 等标注引用来源"。
    """
    return prompt.split("【检索结果】\n", 1)[1].split("\n\n【用户问题】", 1)[0]


def test_prompt_keeps_chunk_tail_by_default():
    """默认配置下, chunk 末尾的内容必须完整进入 prompt。"""
    prompt = _build_prompt("定价规则是什么？", _contexts(LONG_DOC + PRICING_TAIL))
    assert "1280 元" in prompt
    assert "85%" in prompt


def test_prompt_respects_configured_limit(monkeypatch):
    """截断长度应来自配置, 而不是写死的 600。"""
    doc = "甲" * 5000
    monkeypatch.setattr(settings, "context_snippet_chars", 3000, raising=False)
    prompt = _build_prompt("问题", _contexts(doc))
    # "[1] " 前缀 (4 字符) 之后应恰好是 3000 个字符。
    assert len(_ctx_body(prompt)) == len("[1] ") + 3000

    monkeypatch.setattr(settings, "context_snippet_chars", 500, raising=False)
    prompt = _build_prompt("问题", _contexts(doc))
    assert len(_ctx_body(prompt)) == len("[1] ") + 500


def test_prompt_numbers_contexts_and_keeps_question():
    prompt = _build_prompt("我的问题", _contexts("文档甲") + _contexts("文档乙"))
    assert "[1] 文档甲" in prompt
    assert "[2] 文档乙" in prompt
    assert "【用户问题】我的问题" in prompt


@pytest.mark.parametrize("doc", ["", None])
def test_prompt_handles_empty_document(doc):
    prompt = _build_prompt("问题", [{"document": doc, "id": "c0"}])
    assert "[1] " in prompt
