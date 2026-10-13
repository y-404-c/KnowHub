# -*- coding: utf-8 -*-
"""模型输出清洗工具。

背景: 部分推理模型 (如 MiniMax-M3 / DeepSeek-R1 系列) 会把内部思考过程以
`<think>...</think>` 块的形式吐在 `content` 里, 而不是放进独立的
`reasoning_content` 字段。如果原样透传, 用户会在回答正文开头看到一大段英文
内心独白, 录屏演示里尤其明显。

`OPENAI_EXTRA_BODY` 可以在部分平台上关掉思考, 但:
1. 不是所有 provider / 网关都支持该字段;
2. 字段拼错或被 shell 吃掉引号时静默失效;
3. 换成别的 provider 又要重配一遍。

所以这里做**兜底清洗**: 不管上游有没有关掉, 出网前统一剥掉思考块。

支持的标签 (各厂商叫法不一, 统一按"块"处理, 连内容一起删):
`<think>` / `<thinking>` / `<reasoning>` / `<reflection>`

两个入口:
- `strip_think(text)`: 一次性清洗 (非流式 / 落库前)。
- `ThinkBlockFilter`: 流式逐 token 清洗, 能处理标签被切分到多个 chunk
  的情况 (例如先收到 `<th`, 再收到 `ink>`)。
"""
from __future__ import annotations

import re

# 各厂商的思考标签名 (不带尖括号)。
THINK_TAGS = ("think", "thinking", "reasoning", "reflection")

# 完整块 `<tag ...>...</tag>`: 连内容一起删除。
_BLOCK_RES = tuple(
    re.compile(rf"<{t}\b[^>]*>.*?</{t}\s*>", re.IGNORECASE | re.DOTALL) for t in THINK_TAGS
)
# 未闭合块 `<tag ...>...`: 模型吐一半就结束 / 被截断, 从标签处到结尾全删。
_OPEN_RES = tuple(
    re.compile(rf"<{t}\b[^>]*>.*", re.IGNORECASE | re.DOTALL) for t in THINK_TAGS
)
# 残留的结束标签 / 自闭合标签 `<thinking/>`: 只删标签本身。
_STRAY_RE = re.compile(rf"</?(?:{'|'.join(THINK_TAGS)})\b[^>]*>", re.IGNORECASE)


def strip_think(text: str) -> str:
    """剥离模型输出里的思考块, 返回清洗后的正文。

    - 完整 `<think>...</think>`: 标签连同内容整块移除。
    - 未闭合的 `<think ...>`: 从该标签到结尾全部移除。
    - 残留的 `</think>` / 自闭合 `<thinking/>`: 移除标签本身。
    - 非字符串输入: 原样返回, 不抛异常 (调用点大多是宽松透传)。
    """
    if not isinstance(text, str) or not text:
        return text
    out = text
    for regex in _BLOCK_RES:
        out = regex.sub("", out)
    for regex in _OPEN_RES:
        out = regex.sub("", out)
    out = _STRAY_RE.sub("", out)
    return out.strip()


class ThinkBlockFilter:
    """流式思考块过滤器。

    用法::

        f = ThinkBlockFilter()
        for chunk in stream:
            visible = f.feed(getattr(chunk, "content", "") or "")
            if visible:
                yield visible
        tail = f.flush()
        if tail:
            yield tail

    状态机:
    - 正文态: 正常放行文本, 遇到任一开标签则切到思考态并丢弃其后内容。
    - 思考态: 丢弃所有文本, 直到遇到对应的闭标签才切回正文态。
    - 因为开/闭标签可能被切到两个 chunk 之间, 每次只消费"确定安全"的部分,
      把可能为标签前缀的尾部留在 buffer 里等下一个 chunk。
    """

    # 最长的闭标签是 `</reflection>` (13 字符); 多留一点余量。
    _RESERVE = 16

    def __init__(self, tags=THINK_TAGS) -> None:
        self._open_tokens = tuple("<" + t for t in tags)
        self._close_tokens = tuple("</" + t for t in tags)
        self._buf = ""
        self._in_think = False

    def _safe_len(self, text: str) -> int:
        """返回 buffer 中可以安全消费的长度 (保留疑似标签前缀的尾巴)。"""
        if len(text) <= self._RESERVE:
            return 0
        return len(text) - self._RESERVE

    @staticmethod
    def _find_earliest(text: str, tokens):
        """找出 tokens 中最早出现的位置, 返回 (index, matched_token); 没有则 (-1, None)。"""
        best_idx, best_tok = -1, None
        lowered = text.lower()
        for tok in tokens:
            idx = lowered.find(tok)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx, best_tok = idx, tok
        return best_idx, best_tok

    def feed(self, chunk: str) -> str:
        """喂入一个片段, 返回应当展示给用户的文本。"""
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []

        while self._buf:
            if not self._in_think:
                idx, tok = self._find_earliest(self._buf, self._open_tokens)
                if idx == -1:
                    # 没有开标签: 消费掉除尾部保留区以外的全部内容。
                    n = self._safe_len(self._buf)
                    if n == 0:
                        break
                    out.append(self._buf[:n])
                    self._buf = self._buf[n:]
                    continue
                # 找到开标签: 标签之前的正文放行, 之后进入思考态。
                if idx > 0:
                    out.append(self._buf[:idx])
                # 跳过开标签本身 (`>` 可能还没到, 到就跳过去, 没到就整个丢弃)。
                gt = self._buf.find(">", idx)
                self._buf = "" if gt == -1 else self._buf[gt + 1:]
                self._in_think = True
                continue

            # 思考态: 找闭标签。
            idx, tok = self._find_earliest(self._buf, self._close_tokens)
            if idx == -1:
                # 还没收到闭标签: 丢弃除尾部保留区以外的全部内容。
                n = self._safe_len(self._buf)
                if n == 0:
                    break
                self._buf = self._buf[n:]
                continue
            # 收到闭标签: 丢弃思考内容, 切回正文态。
            gt = self._buf.find(">", idx)
            if gt == -1:
                # 闭标签还没收全 (`</think` 但 `>` 在下一个 chunk): 丢掉它之前的
                # 思考内容, 把疑似闭标签的尾巴留在 buffer 里, 等收全再切状态.
                # 直接切回正文态会把后面那个 `>` 当成正文漏给用户。
                self._buf = self._buf[idx:]
                break
            self._buf = self._buf[gt + 1:]
            self._in_think = False

        return "".join(out)

    def flush(self) -> str:
        """流结束时调用: 吐出 buffer 里剩余的非思考文本。

        若流结束时仍停在思考态, 说明思考块没闭合, 剩余内容全部丢弃 —— 这与
        `strip_think` 对"未闭合块"的处理保持一致。
        """
        rest = "" if self._in_think else self._buf
        self._buf = ""
        self._in_think = False
        return rest
