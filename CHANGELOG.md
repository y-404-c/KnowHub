# Changelog

本文件记录每个版本值得单独说一嘴的变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [0.2.0] - 2026-08-29

### 🎬 演示

- **重制演示录屏**：14.5 秒 / 1440×900 / H.264，核心是**真实提问 → 真实流式回答**（两轮）。
  旧录屏 2 分 44 秒且**没有任何真实 LLM 回答**，只有 webm（Safari 放不了）。
- 新增 `frontend/record_demo.mjs`（Playwright 无头录制）与 `scripts/build_demo_assets.py`
  （webm → MP4 + GIF 转码），录制到发布全流程可一键复现。
- 产物 `docs/demo/demo.mp4`（0.74 MB）托管在 GitHub Release，README 内嵌 `demo.gif` 预览。
- 移除 `docs/demo/unikb_demo.webm`（5.3 MB）。

### 🐛 修复：三处静默丢内容的数据链路缺陷

这三个 bug 的共同点值得记一笔：**都不报错，链路全是 200 OK**，只在模型侧表现为
「答非所问 / 说原文被截断 / 未找到相关信息」，排查时极易误判成模型能力问题。

- **清洗器把行内百分号当成 LaTeX 注释删掉了**
  `_RE_LATEX_COMMENT` 原本是 `r"%.*$"`，会把行内**任意** `%` 到行尾整段删除。
  业务文档里百分比极其常见，于是
  「年度促销按原价的 `85%` 收费；如客户购买 20 个账号，则总价等于折后单价乘以账号数量。」
  入库时被削成「……按原价的 `85`」—— chunk 断在半句，模型只能回答「原文此处似乎不完整」。
  改为只认行首（可前置空格）的整行注释。

- **上下文截断在三处各写死一次，且都小于 chunk_size**
  `tools.py` 300 / `chat.py` 600 / `pipeline.py` 600，而默认 `chunk_size=500`、
  `adaptive_chunk_size` 上限 5500。长 chunk 的文末（定价、结论这类常放最后的内容）
  被静默切掉，后果是：Agent 链路答「未找到相关信息」、`calculator` 因为拿不到数字
  而永不触发，RAG 主链路却又抱怨「原文被截断」—— 同一份文档两条路径行为不一致。
  统一成 `settings.context_snippet_chars`（默认 2000）。

- **推理模型的 `<think>` 思考块混进了回答正文**
  MiniMax-M3 等模型把思考过程吐在 `content` 而不是 `reasoning_content`，用户会在
  回答开头看到一大段英文独白。`OPENAI_EXTRA_BODY` 能在部分平台关掉，但不是所有
  provider / 网关都支持，配错时还静默失效。新增 `app/core/text_utils.py` 做兜底：
  `strip_think()` 用于非流式与落库前，`ThinkBlockFilter` 状态机用于 SSE 流式
  （能处理 `<think>` 被切到两个 chunk 之间的情况，逐 chunk `replace` 会漏）。

### 🧪 测试

- 测试不再污染开发者数据目录。conftest 只重定向了 `*_PERSIST_DIR`，而
  `kb_registry.json` 直接挂在 `DATA_DIR` 上 —— 跑一次 pytest 就会把本地知识库
  注册表冲掉，表现为「重启后端后知识库不存在，但 bm25 / 向量文件还在磁盘上」。
  新增 `UNIKB_DATA_DIR` 环境变量，conftest 一并重定向。
- 新增 4 个测试文件、共 39 条回归用例，全部针对这次的三处缺陷：

  | 文件 | 用例数 | 覆盖 |
  |---|---|---|
  | `tests/test_cleaner_percent.py` | 10 | 行内 `%` 不被当 LaTeX 注释删掉 |
  | `tests/test_text_utils.py` | 19 | 思考块清洗，含流式逐字符喂入、标签被切断在 chunk 边界 |
  | `tests/test_context_snippet_limit.py` | 5 | 三处截断都读 `settings.context_snippet_chars` |
  | `tests/test_agent_tools.py` | 5 | `hybrid_search` 不再把文末内容切掉 |

- 全量 **378 passed**。

### 🔧 其它

- `DEMO_KB` 支持 `NEXT_PUBLIC_DEMO_KB` 覆盖：录屏可指向独立知识库，避免把开发时
  随手上传的文档混进演示画面。
- `createSession` 显式带 `kb_id`：原先不传，后端落成 `"default"`，与实际检索用的库
  不一致，历史页显示错误。
- 首页快捷问题对齐预置示例文档内容，保证开箱即问就有带引用的真实回答。

[0.2.0]: https://github.com/USTCCB/UniKB/releases/tag/v0.2.0
