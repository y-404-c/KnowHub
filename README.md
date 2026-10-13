# 知枢 KnowHub —— 面向企业知识管理场景的 RAG 知识库平台

> Multi-Agent + MCP + Hybrid Search + Full-Stack。
>
> 本项目基于上游开源项目 [USTCCB/UniKB](https://github.com/USTCCB/UniKB)（MIT License）二次开发，见文末[二次开发说明](#二次开发说明)。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116%2B-009688)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)]()
[![MCP](https://img.shields.io/badge/MCP-1.0-purple)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## 简介

知枢 KnowHub 是一个面向企业知识管理场景的 RAG（Retrieval-Augmented Generation）平台。它把本地文档知识通过多 Agent 协作、MCP 工具协议、混合检索与重排序，最终以流式、带检索片段的方式回答用户问题。

> **当前阶段说明**：知识库 ACL 采用 owner-based 隔离模型（每个私有 kb 属于一个用户）；多用户团队协作、角色权限、组织级知识库是 Roadmap 中的后续目标。

LLM 支持按 provider/model **按请求切换**：`app.agents.llm_router.get_llm(provider, model)` 接受 provider + model 参数，内部按 `(provider, model)` 作 cache key（lru_cache）。`get_llm()` 不带参数仍走环境变量里的默认组合，向后兼容老调用点。

## 🎬 效果演示

<p align="center">
  <img src="docs/demo/demo.gif" width="820" alt="知枢 KnowHub 演示：提问 → 流式回答 → 引用溯源">
</p>

## 核心特性

- 混合检索：BM25 + 向量语义 + RRF 融合
- 精排重排：Cross-Encoder（BGE-reranker）抑制幻觉，单例缓存避免每次请求重新加载模型
- RAG 工程调优（检索质量进阶，见 [RAG 工程调优](#rag-工程调优检索质量进阶)）：文档清洗（42 条正则，占质量调优 ~60% 精力）、自适应切分（短文档整篇保留 / 长文档 5500 字符）、元数据增强（HyDE 假设性问题）、父文档召回（连坐召回）、重排候选池优化（Top-60→Top-25，延迟 11.4s→2.3s，Recall@5 93%→98.7%）、查询改写（Query Rewriting）、离线 Recall@K 评估
- 多 Agent 协作：基于 LangGraph 的 Planner / Retriever / Coder / Reviewer 流程，Reviewer 不通过时通过 `add_conditional_edges` **真正回环**到 Retriever 重新检索（`MAX_REVIEWER_RETRIES=2` 防死循环）
- MCP 协议：KnowHub 自身工具（`hybrid_search` / `calculator` / `current_date`）**通过 stdio / SSE 暴露成 MCP Server** 给 Claude Desktop、Cursor、Trae 等客户端调用；Agent 内部暂未作为 MCP Client 去消费外部 MCP 工具（单向）。**注意：每个 MCP Server 进程启动时绑定一个 kb_id，无法在同一进程内动态切换知识库**
- 文档解析：PDF、DOCX、Markdown、TXT 与图片 OCR（图片 OCR 需要本机安装 Tesseract）
- 流式问答：SSE 协议 + 多轮对话管理 + 引用溯源；Agent 模式把 graph.stream 放进 threadpool，按节点实时推 trace 事件，不阻塞 FastAPI 事件循环
- 工程化：JWT 鉴权（HttpOnly cookie + Bearer 双轨）+ Docker Compose 一键部署 + GitHub Actions CI + GHCR 镜像发布 CD
- 安全加固：KB ACL 防越权（Agent 模式真的会按 `state['kb_id']` 走对应私有 kb）、CORS 白名单、登录/Chat 限流、CSP/HSTS 安全响应头
- 性能优化：同步 CPU/IO offload 到线程池、BM25 惰性重建、HybridRetriever 进程内缓存、Reranker 单例、LangGraph 编译产物缓存、多路改写 query 并发检索
- 文档管理：上传大小限制、DELETE `/documents/{doc_id}` 删除并同步清理 vector/BM25
- 可观测性：内置 **LangFuse**（可选，关闭时无副作用）与 **LangSmith**（可选，`.env` 里取消 4 行注释即可追踪 LangChain/LangGraph 全链路）
- 评估体系：集成 RAGAS 自动评估（4 大指标）

## 技术栈

| 层级 | 选型 |
|---|---|
| LLM | DeepSeek / Qwen / OpenAI（按 provider + model 在请求内动态切换，缓存 key = `(provider, model)`） |
| Agent 框架 | LangChain + LangGraph |
| MCP 协议 | mcp 1.0+（stdio + SSE 传输；Server 启动时固定 kb_id） |
| 后端 | FastAPI + Uvicorn（异步） |
| 前端 | Next.js 14 + TypeScript（App Router） |
| 向量库 | Chroma（轻量、可零配置；另有纯 numpy 后端兜底） |
| 检索 | BM25（rank_bm25）+ 向量 + RRF |
| 重排 | Cross-Encoder（BAAI/bge-reranker-base） |
| 数据库 | SQLite（用户/历史会话/文件元数据，SQLAlchemy 抽象，可切 PostgreSQL）+ Chroma 向量库 |
| 缓存 | Redis（Embedding / 热点问答缓存，可选；限流当前为进程内实现，多实例部署需迁移到 Redis） |
| 文件 | 本地 / MinIO 可切换 |
| 可观测 | LangFuse（可选）+ LangSmith（可选） |
| 评估 | RAGAS |
| 工程化 | Docker Compose + GitHub Actions CI + GitHub Container Registry 发布 |

## 架构图

```
+----------------------------------------------------------------+
|                    Next.js 14 (Frontend)                         |
|    Chat UI  Upload  Sources  History  (共享 AuthBar/Nav)         |
+--------------------------+----------------------------------------+
                           |  HTTPS / SSE
+--------------------------v----------------------------------------+
|                   FastAPI (Backend)                              |
|  +----------+  +----------+  +-------------+  +----------+       |
|  |   Auth   |  |Documents |  |  Chat / SSE |  | History  |       |
|  +----------+  +----------+  +------+------+  +----------+       |
|                                |                                  |
|               +----------------v------------------+               |
|               |       LangGraph Multi-Agent       |              |
|               | Planner -> Retriever -> Coder -> Reviewer        |
|               +---+----------+---------+---------+               |
|                   |          |         |                          |
|            +------v-----+ +--v-----+ +--v-------+                |
|            | RAG 链路   | |LLM 路由| |MCP 工具集|                |
|            | BM25+Vec+RRF| |DS/Qwen/OAI| | (可插拔)|              |
|            +------+-----+ +--------+ +----------+                |
|                   |                                                |
|        +----------+--------+----------+----------+                 |
|        v          v        v          v                           |
|    Chroma    PostgreSQL   Redis    LangFuse/LangSmith (可选)       |
+--------------------------------------------------------------------+
```

## 快速开始

### 本地开发

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate   # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                              # 填入 LLM_API_KEY（如 DeepSeek）
python -m uvicorn app.main:app --port 8000

# 另开一个终端起前端
cd frontend && npm install && npm run dev
# 打开 http://localhost:3000
```

Windows 用户也可以直接双击根目录的 `start_knowhub.bat`（自动起前后端并打开浏览器）。

### Docker Compose

```bash
cp backend/.env.example backend/.env   # 填入 LLM_API_KEY
docker compose up
```

### 体验 MCP Server

```bash
cd backend
python -m app.mcp.server <kb_id>   # stdio 传输, 绑定指定知识库
```

把命令配进 Claude Desktop / Cursor / Trae 的 MCP 配置（详见 [docs/MCP_SETUP.md](docs/MCP_SETUP.md)），即可在这些客户端里直接调用 `hybrid_search` 等工具。

## 测试覆盖

`backend/tests/` 拆成两层：

| 类型 | 文件 | 覆盖点 | 运行条件 |
|---|---|---|---|
| 单元测试 | `test_*.py`（除 integration） | auth/ACL/CORS/限流/JWT cookie/上传大小/BM25 惰性重建/retriever 缓存/reviewer JSON/文档删除 等 | 不依赖 torch / chromadb / sentence-transformers / 真实 LLM |
| 集成测试 | `test_integration_retrieval.py` | HybridRetriever add/retrieve/delete + RRF + rerank + pipeline 端到端 + agent 模式 | 走 `tests/_fakes.py` fake 路径，无需重包 |

当前数据：**378 passed**（370 单测 + 8 集成；含二次开发期间修复的 Windows 平台用例）。

测试隔离说明：`tests/conftest.py` 把 DB / chroma / bm25 / uploads 全部重定向到临时目录，任何**单个**测试文件都能独立跑通。建表由 session 级 autouse fixture 统一完成——`create_tables()` 只在 lifespan 里调用，而 `return TestClient(app)` 这种写法不触发 lifespan，若靠"别人先跑过 `with TestClient(app)`"来建表就会产生隐式顺序依赖。

本地跑法：

```bash
cd backend
# 单元测试
python -m pytest -v --ignore=tests/test_integration_retrieval.py

# 集成测试（fake 路径, 不需要 chromadb/torch）
# Windows PowerShell: $env:UNIKB_FAKE_EMBEDDING=1
UNIKB_FAKE_EMBEDDING=1 python -m pytest -v tests/test_integration_retrieval.py

# 全部
python -m pytest -v
```

## 评估与质量

KnowHub 集成了 [RAGAS](https://docs.ragas.io/) 做自动化评估，覆盖 4 个核心指标：

- **faithfulness**：答案是否忠实于检索上下文（抑制幻觉）
- **answer_relevancy**：答案与问题的相关程度
- **context_precision**：检索结果里相关 chunk 的比例
- **context_recall**：回答问题所需信息被召回的比例

### 跑评估

```bash
cd backend

# fake 模式: 不需要 API key, 用于 CI / 沙箱验证
python -m tests.run_ragas_eval --kb default \
    --dataset ../data/eval/qa.jsonl \
    --out ../data/eval/ragas_report.json \
    --baseline-out ../data/eval/baseline.json \
    --llm-mode fake_llm

# 真实模型模式: 需要 .env 里配置 LLM_API_KEY
python -m tests.run_ragas_eval --kb default \
    --dataset ../data/eval/qa.jsonl \
    --out ../data/eval/ragas_report.json \
    --baseline-out ../data/eval/baseline.json \
    --llm-mode real_llm
```

评估后生成两个文件：

- `data/eval/ragas_report.json`：完整报告，含每条样本的 question/answer/contexts/ground_truth
- `data/eval/baseline.json`：精简版，只保留 scores + 元信息，适合提交到仓库做 baseline 对比

## RAG 工程调优（检索质量进阶）

> 真实 RAG 项目里，**检索质量 > 模型能力**。下面这套调优经验来自 RAG 知识库工程实践，
> 已落地到 `backend/app/rag/` 并配有单元测试（`tests/test_cleaner.py` 等）。

| 手段 | 做法 | 收益 |
|---|---|---|
| **文档清洗**（`cleaner.py`） | 42 条正则去除控制字符/零宽字符/页眉页脚/脚注/标记语言残留/LaTeX/URL 归一，确定性、可复现、零成本 | RAG 质量调优里清洗占 **~60%** 精力，直接决定向量质量上限 |
| **自适应切分**（`chunker.py`） | 短文档（≤ `adaptive_short_doc_chars`，默认 500 字符）整篇保留为 1 个 chunk；长文档改用更大的 `adaptive_chunk_size`（默认 5500 字符） | 短文档不再被无谓切碎；长文档减少碎片、保留更长上下文 |
| **元数据增强 / HyDE**（`metadata.py`） | 入库抽取式元数据（关键词/摘要/语言），并用 LLM 生成**假设性问题（HyDE）**作为检索锚点（无 LLM 时回退到首句摘要） | 问句与文档陈述句式差异大时，语义召回更稳 |
| **父文档召回 / 连坐召回**（`retriever.py`） | 召回某 chunk 时，一并召回**同文档**的兄弟 chunk，继承父 chunk 分数紧随其后 | 喂给 LLM 的是完整段落/章节，而不是孤立碎句 |
| **重排候选池优化**（`reranker.py`） | 先按 RRF 截断候选池到 **Top-60**，再让 Cross-Encoder 精排到 **Top-25** | 压住 Cross-Encoder 算力（延迟 **11.4s → 2.3s**），同时保住召回上限（Recall@5 **93% → 98.7%**） |
| **查询改写**（`query_rewrite.py`） | LLM 把口语化/歧义问题扩写成多条检索 query（扩展/拆解/消歧），多 query 分别检索后 RRF 融合；无 LLM 时走子句切分回退 | 扩大召回面，尤其利好复合问题 |
| **离线 Recall@K 评估**（`evaluate.py`） | `RecallEvaluator` 在 (question, relevant_ids) 数据集上统计 Recall@1/3/5/10，对比调优前后 | 调参有量化依据，不再凭感觉 |

开关全部集中在 `app/core/config.py`（`cleaner_enabled` / `adaptive_chunking_enabled` /
`hyde_enabled` / `parent_recall_enabled` / `rerank_candidate_pool` / `rerank_top_n` /
`query_rewrite_enabled`），可逐项灰度。

## 可观测性（可选，默认全关）

**LangFuse**：`.env` 中设置

```
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

RAG pipeline（retriever / reranker / llm）会自动写入 trace。

**LangSmith**（LangChain/LangGraph 官方 tracing）：`.env` 中取消 4 行注释

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=KnowHub
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

业务代码零改动，多 Agent 图的每个节点与每次 LLM 调用都会自动上报。两套系统可并存；都不开启时完全 no-op，不发任何外部请求。

## Roadmap

- [x] 多 LLM 路由（按 provider + model 按请求切换）
- [x] BM25 + 向量混合检索 + RRF
- [x] LangGraph 多 Agent + Reviewer 回环（add_conditional_edges）
- [x] MCP Server 单向暴露内部工具（stdio/SSE）
- [x] SSE 流式问答
- [x] 多轮对话 / 历史会话 / 引用溯源
- [x] RAGAS 自动评估脚本 + JSON 报告
- [x] LangFuse 可观测（可选）
- [x] **LangSmith 可观测（可选，二次开发新增）**
- [x] Docker Compose 一键起 + healthcheck
- [x] 知识库 ACL + 用户/知识库隔离（P0）
- [x] 安全加固：CORS 白名单、登录/Chat 限流、JWT HttpOnly cookie、CSP/HSTS（P0/P2）
- [x] 性能优化：线程池 offload、BM25 惰性重建、Retriever 进程内缓存、CrossEncoderReranker 单例缓存（P1）
- [x] 性能优化：LangGraph 编译产物缓存、多路改写 query 并发检索（P1）
- [x] 上传目录配置化：`UPLOAD_DIR` 读 `settings.file_storage_dir`，不再硬编码（P2）
- [x] 用户表 SQLAlchemy 化（SQLite，可切换 PostgreSQL）（P2-12）
- [x] 文档删除接口 + vector/BM25 同步清理（P2-13）
- [x] Chunker 主打包路径的 overlap 真的生效（P2）
- [x] Planner 结构化决策 + Agent 工具路由（calculator / current_date / hybrid_search）（P2）
- [x] RAG 工程调优：文档清洗、自适应切分、HyDE、父文档召回、重排候选池优化、查询改写、离线 Recall@K 评估
- [ ] 对话历史/文件元数据统一 SQLAlchemy 迁移
- [ ] MinIO 文件存储
- [ ] MCP Client（Agent 消费外部 MCP 工具，双向集成）

## 二次开发说明

本项目基于 [USTCCB/UniKB](https://github.com/USTCCB/UniKB)（MIT License）二次开发，感谢上游作者的工作。在保留上游核心架构的前提下，本仓库做了以下改动：

**缺陷修复**

1. **跨平台测试修复**：3 个测试文件（`test_document_delete.py` / `test_jwt_cookie_security.py` / `test_upload_size_limit.py`）硬编码 `chdir("/tmp")`，Windows 下 11 个测试直接 error——改为 `tempfile.gettempdir()` 回退，修复后全套 **378 passed**。
2. **`.env` 加载路径锚定**：pydantic-settings 原来用 CWD 相对的 `.env`，换启动目录（仓库根/systemd/Docker 不同 WORKDIR）就读不到配置——锚定为 `backend/.env` 绝对路径，与项目自身"存储路径不随 CWD 漂移"的设计哲学保持一致。
3. **启动脚本修正**：原 `start_unikb.bat` 硬编码了作者本机的 `E:\UniKB-main` 路径与托管运行时位置——改为自动取脚本所在目录（`%~dp0`）、运行时走 PATH，任意机器双击可用，并重命名为 `start_knowhub.bat`。
4. 删除误提交到版本库的 uvicorn 运行日志（`uvicorn.err/out`），并加入 `.gitignore`。

**功能增强**

5. **LangSmith 可选追踪**：新增 `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` / `LANGCHAIN_ENDPOINT` 配置项，配置加载后统一桥接进进程环境变量（pydantic-settings 从 `.env` 读出的值默认不会回写 `os.environ`，不做桥接则 LangChain 永远感知不到）；与现有 LangFuse 并存，默认关闭、零网络请求。
6. **项目更名与展示层中文化**：UniKB → 知枢 KnowHub（FastAPI 标题、MCP Server 名、前端页面、README 等展示层统一替换；`UNIKB_*` 环境变量、`unikb-*` 容器名等内部标识保留，保证与上游 diff 最小、便于追踪上游更新）。

**审查范围**：后端全部 101 个 Python 文件的配置/LLM 路由/Agent 图/MCP/RAG 主链路与安全边界逐一过审；单测 + 集成测试全绿（378 passed）。

> 注：`docs/` 目录保留上游文档原貌（含 UniKB 字样），作为上游历史存档。

## License

MIT（与上游一致，见 [LICENSE](LICENSE)）
