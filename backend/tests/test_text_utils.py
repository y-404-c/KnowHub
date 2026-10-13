# -*- coding: utf-8 -*-
"""app.core.text_utils 单测: 思考块清洗。

覆盖点:
1. 完整 `<think>...</think>` 被整块剥离。
2. 未闭合 `<think>` (模型吐一半就结束) 从标签处截断。
3. 残留的 `</think>` / 裸 `<thinking>` 标签被清掉。
4. 流式: 标签被切分到多个 chunk 时也不能漏到用户可见文本里。
5. 流式: 流结束仍停在思考态时, 剩余内容全部丢弃。
6. 普通文本不被误伤。
"""
from __future__ import annotations

import pytest

from app.core.text_utils import ThinkBlockFilter, strip_think


# ---------- strip_think (一次性清洗) ----------

def test_strip_complete_think_block():
    raw = "<think>The user asks about X. Let me check [1].</think>\n\n答案是 42。"
    assert strip_think(raw) == "答案是 42。"


def test_strip_multiline_think_block():
    raw = "<think>\nline1\nline2\n</think>正文在这里"
    assert strip_think(raw) == "正文在这里"


def test_strip_unclosed_think_block():
    # 模型吐了开头但没闭合 (流式中途断 / 被截断): 从标签处到结尾全丢弃。
    raw = "前缀保留。<think>thinking forever without closing"
    assert strip_think(raw) == "前缀保留。"


def test_strip_stray_close_tag():
    raw = "正文 A</think>正文 B"
    assert strip_think(raw).replace("\n", "") == "正文 A正文 B"


def test_strip_other_vendor_tags():
    raw = "<thinking>内部</thinking>可见内容"
    assert strip_think(raw) == "可见内容"


def test_strip_case_insensitive():
    raw = "<THINK>hidden</THINK>shown"
    assert strip_think(raw) == "shown"


def test_strip_leaves_normal_text_untouched():
    raw = "## 标题\n- 列表项 [1]\n\n普通回答。"
    assert strip_think(raw) == raw


def test_strip_handles_empty_and_non_str():
    assert strip_think("") == ""
    assert strip_think(None) is None
    assert strip_think(123) == 123


# ---------- ThinkBlockFilter (流式清洗) ----------

def _feed_all(chunks):
    f = ThinkBlockFilter()
    out = "".join(f.feed(c) for c in chunks)
    out += f.flush()
    return out


def test_stream_whole_block_in_one_chunk():
    out = _feed_all(["<think>secret</think>", "可见"])
    assert out == "可见"
    assert "secret" not in out


def test_stream_open_tag_split_across_chunks():
    # `<think>` 被切成 `<th` + `ink>`: 逐 chunk replace 会漏, 状态机不会。
    out = _feed_all(["前面", "<th", "ink>secret</think>", "后面"])
    assert out == "前面后面"
    assert "secret" not in out


def test_stream_close_tag_split_across_chunks():
    out = _feed_all(["<think>secret</thi", "nk>", "可见文本"])
    assert out == "可见文本"
    assert "secret" not in out


def test_stream_char_by_char():
    # 最极端: 一个字符一个 chunk。
    raw = "A<think>secret</think>B"
    out = _feed_all(list(raw))
    assert out == "AB"
    assert "secret" not in out


def test_stream_prefix_emitted_before_think_starts():
    # 思考块之前的正文必须正常放行, 且思考开始后立刻掐断。
    out = _feed_all(["好的，", "<think>hmm", " more", "</think>", "答案是 42"])
    assert out == "好的，答案是 42"


def test_stream_unclosed_think_at_end_discards_tail():
    out = _feed_all(["正文", "<think>thinking forever"])
    assert out == "正文"


def test_stream_no_think_passthrough():
    chunks = ["KnowHub", " 是一个", " RAG 平台", "。"]
    assert _feed_all(chunks) == "KnowHub 是一个 RAG 平台。"


@pytest.mark.parametrize("raw", [
    "<think>a</think>b",
    "b<think>a</think>",
    "<think>a</think><think>b</think>c",
    "no think here",
])
def test_stream_matches_strip_think(raw):
    """流式过滤器的结果应与一次性清洗一致 (同一份输入)。"""
    assert _feed_all([raw]) == strip_think(raw)
