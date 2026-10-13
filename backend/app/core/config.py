"""Application configuration loaded from environment variables.

.env.example 中列出的所有 key 都在这里集中映射, 业务代码通过 settings.xxx 访问。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ 目录 (本文件位于 backend/app/core/config.py).
BACKEND_ROOT = Path(__file__).resolve().parents[2]
# 所有运行时数据统一落在 backend/data/ 下.
#
# 允许用 UNIKB_DATA_DIR 整体覆盖: `kb_registry.json` 这类文件不走
# `*_PERSIST_DIR` 系列配置, 而是直接拼在 DATA_DIR 上, 所以只重定向那几个
# PERSIST_DIR 是拦不住的 —— 跑一次 pytest 就会把开发者本地的知识库注册表
# 冲掉 (表现为重启后端后"知识库不存在", 但 bm25/vector 文件还在磁盘上)。
DATA_DIR = Path(os.environ.get("UNIKB_DATA_DIR") or (BACKEND_ROOT / "data"))


def _to_abs_path(value: str) -> str:
    """把相对路径按 backend/ 根目录解析成绝对路径.

    `./data/chroma` 这类配置是相对**进程 CWD** 的: CWD 一变 (Docker WORKDIR 不同、
    systemd 启动、测试 monkeypatch.chdir), 数据就会落到意料之外的位置, 或者因为
    父目录不存在直接 IOError。这里统一锚定到 backend/, 让 CWD 不再影响存储位置。
    """
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((BACKEND_ROOT / p).resolve())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # 锚定到 backend/.env, 而不是 CWD 相对的 ".env": 与下方存储路径一样,
        # 从仓库根 / systemd / Docker 不同 WORKDIR 启动都应读到同一份配置文件。
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== App =====
    app_env: Literal["dev", "prod", "test"] = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"
    # 仅用于本地 / CI 演示，显式开启时不请求外部模型或加载 torch 模型。
    fake_llm: bool = Field(default=False, validation_alias="UNIKB_FAKE_LLM")
    fake_embedding: bool = Field(default=False, validation_alias="UNIKB_FAKE_EMBEDDING")

    # ===== LLM (三个 provider 可并存) =====
    llm_provider: Literal["deepseek", "qwen", "openai"] = "deepseek"

    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    qwen_api_key: Optional[str] = None
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    # 部分 OpenAI 兼容 provider (如 MiniMax) 需要在请求 body 里加私有字段
    # (例如 {"thinking": {"type": "disabled"}} 关掉 M3 的思考模式).
    # JSON 字符串, 解析失败或为空则不传. 与 provider 名无关, 避免在代码里写死分支.
    openai_extra_body: Optional[str] = None

    # 兼容旧字段
    @property
    def default_llm_provider(self) -> str:
        return self.llm_provider

    @property
    def default_llm_model(self) -> str:
        return {
            "deepseek": self.deepseek_model,
            "qwen": self.qwen_model,
            "openai": self.openai_model,
        }[self.llm_provider]

    # ===== Embedding / Rerank =====
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    rerank_model: str = "BAAI/bge-reranker-base"

    # ===== Storage =====
    database_url: str = "sqlite:///./data/unikb.db"
    redis_url: str = "redis://localhost:6379/0"
    chroma_persist_dir: str = "./data/chroma"
    # 上传文件落盘目录。默认 ./data/uploads —— 必须和上传链路历史上实际使用的
    # 目录保持一致, 否则改配置驱动的那一刻, 之前上传的文件会全部"找不到"
    # (原先 documents.py 里硬编码的就是 data/uploads)。改这个值前先迁移已有文件。
    file_storage_dir: str = "./data/uploads"
    # BM25 倒排索引的持久化目录 (文件名 bm25_{kb_id}.pkl).
    bm25_persist_dir: str = "./data"
    # 会话历史的持久化目录 (文件名 {user}.jsonl).
    history_storage_dir: str = "./data/history"

    # 存储类路径一律锚定到 backend/ 根目录, 避免随 CWD 漂移 (见 _to_abs_path).
    _normalize_dirs = field_validator(
        "chroma_persist_dir", "file_storage_dir", "bm25_persist_dir", "history_storage_dir"
    )(_to_abs_path)
    # 向量库后端: chroma (默认, 生产推荐) | numpy (纯 Python, 用于 chroma 在当前平台无法运行的兜底).
    # chromadb 在 Windows + Anaconda Python 3.12 上会 ACCESS_VIOLATION, 此时可临时切 numpy.
    vector_store_backend: Literal["chroma", "numpy"] = "chroma"

    # ===== Auth =====
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    # ===== Knowledge Base ACL =====
    # 公共只读知识库白名单. pydantic-settings 对 list 默认要求 JSON 格式,
    # 这里用 str, 由 kb_registry 自行解析 (支持逗号分隔或 JSON 数组).
    # 这些 kb_id 任何登录用户可读, 不可写入; 上传文档时若 kb 不存在会自动创建并归属当前用户.
    # 默认含 "default" 以兼容 demo.
    public_kb_ids: str = "default"

    # ===== CORS =====
    # 允许跨域的前端 origin 白名单 (逗号分隔 / JSON 数组).
    # 严禁 "*" + credentials=True (违反 CORS 规范, 也是明显的安全反模式).
    # 多个 origin 用逗号分隔, 例如 "http://localhost:3000,https://app.example.com"
    cors_allowed_origins: str = "http://localhost:3000"

    # ===== Rate Limit =====
    # 同一账号 login 失败 5 次 / 5 分钟, 锁定 15 分钟 (返回 429).
    rate_limit_login_window_sec: int = 300
    rate_limit_login_fail_max: int = 5
    rate_limit_login_lock_sec: int = 900
    # chat 接口每用户每分钟请求上限 (默认 20).
    rate_limit_chat_per_min: int = 20
    # 是否启用限流 (测试时可关掉以免 flaky)
    rate_limit_enabled: bool = True

    # ===== MCP =====
    mcp_enabled: bool = True
    mcp_servers_config: str = "./mcp_servers.json"
    mcp_transport: str = "stdio"
    mcp_server_name: str = "knowhub"

    # ===== Observability =====
    langfuse_enabled: bool = False
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # ---- LangSmith (LangChain / LangGraph 官方 tracing, 可选) ----
    # LangChain/LangGraph 在检测到环境变量 LANGCHAIN_TRACING_V2=true 时会
    # 自动把每条 chain / graph 执行上报到 LangSmith, 无需改业务代码。
    # 默认关闭: 不填 key、不开启就完全 no-op, 不发任何网络请求。
    langsmith_tracing_v2: bool = Field(default=False, validation_alias="LANGCHAIN_TRACING_V2")
    langsmith_api_key: Optional[str] = Field(default=None, validation_alias="LANGCHAIN_API_KEY")
    langsmith_project: str = Field(default="KnowHub", validation_alias="LANGCHAIN_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias="LANGCHAIN_ENDPOINT",
    )

    # ===== RAG 切分 / 检索参数 =====
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=80, ge=0, le=400)
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 5
    # 每条召回结果拼进 LLM 上下文的字符上限, RAG 主链路 / 评估链路 / Agent 工具
    # 三条路径共用, 保证行为一致。
    #
    # 这个值原先在三处各写死一次 (tools.py 300, chat.py 600, pipeline.py 600),
    # 都小于默认 adaptive_chunk_size, 于是长 chunk 的后半段 —— 很多文档正好把
    # "定价 / 结论 / 注意事项" 放在文末 —— 会被静默切掉。后果不是报错, 而是模型
    # 含糊其辞: Agent 链路答"未找到相关信息"、calculator 拿不到数字所以永不触发,
    # 而 RAG 主链路又会抱怨"原文此处被截断"。这类"答案似是而非"的问题排查成本
    # 极高, 因为链路全是 200 OK。
    context_snippet_chars: int = Field(default=2000, ge=200, le=20000)

    # ===== RAG 质量调优 (来自 RAG 知识库工程实践) =====
    # 1) 文档清洗: 占 RAG 质量调优 ~60% 精力, 42 条正则规则去除噪声.
    cleaner_enabled: bool = True
    # 2) 自适应切分: 短文档整篇保留为 1 个 chunk; 长文档按 adaptive_chunk_size 切.
    adaptive_chunking_enabled: bool = True
    adaptive_chunk_size: int = Field(default=5500, ge=500, le=8000)
    adaptive_short_doc_chars: int = Field(default=500, ge=100)
    # 3) 元数据增强 / HyDE: 入库存假设性问题 (hypothetical question) 提升召回.
    hyde_enabled: bool = True
    # 入库侧 HyDE 是"每 chunk 一次 LLM 调用", 大文档会把上传拖成分钟级并烧 token.
    # 因此只对前 N 个 chunk 走 LLM, 其余回退抽取式首句 (metadata 仍完整, 只是弱一些);
    # 设为 0 表示入库侧完全不调 LLM (纯离线入库).
    hyde_index_max_chunks: int = Field(default=20, ge=0, le=500)
    # 这 N 个 chunk 的 LLM 调用并发度, 避免串行等待也避免打爆 provider 限流.
    hyde_index_concurrency: int = Field(default=4, ge=1, le=32)
    # 4) 父文档召回 (连坐召回): 召回某 chunk 时一并召回同 doc 的兄弟 chunk.
    parent_recall_enabled: bool = True
    parent_recall_extra: int = Field(default=6, ge=0, le=32)
    # 5) 重排优化: 候选池 Top-60 -> 重排后 Top-25 (控制 Cross-Encoder 算力).
    rerank_candidate_pool: int = Field(default=60, ge=10, le=200)
    rerank_top_n: int = Field(default=25, ge=5, le=100)
    # 6) 查询改写 (Query Rewriting): LLM 改写/拆解问题提升召回.
    query_rewrite_enabled: bool = True

    # ===== Upload =====
    # 单文件最大字节数, 防止恶意上传把磁盘填爆或拖死 embedding.
    # 默认 25MB, 可通过环境变量调整. 超出时返回 413.
    upload_max_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)

    def get_llm_api_key(self, provider: str) -> Optional[str]:
        return {
            "deepseek": self.deepseek_api_key,
            "qwen": self.qwen_api_key,
            "openai": self.openai_api_key,
        }.get(provider)

    def get_llm_base_url(self, provider: str) -> str:
        return {
            "deepseek": self.deepseek_base_url,
            "qwen": self.qwen_base_url,
            "openai": self.openai_base_url,
        }.get(provider, self.openai_base_url)

    def get_llm_model(self, provider: str) -> str:
        return {
            "deepseek": self.deepseek_model,
            "qwen": self.qwen_model,
            "openai": self.openai_model,
        }.get(provider, self.openai_model)


def apply_langsmith_env(s: "Settings") -> None:
    """把 .env 里的 LangSmith 配置注入进程环境变量.

    LangChain / LangGraph 的自动 tracing 只认 **os.environ**: pydantic-settings
    从 .env 文件读出的值默认只存在 Settings 对象里, 不会回写环境变量 —— 于是
    ".env 里写了 LANGCHAIN_TRACING_V2=true 却看不到 trace"。这里在配置加载后
    统一桥接一次:

    - 显式开启且填了 key: 注入 4 个 LANGCHAIN_* 变量, 后续 LangChain 调用自动上报;
    - 否则: 显式置 LANGCHAIN_TRACING_V2=false, 保证行为只由本项目 .env 决定,
      不受机器上残留的全局环境变量影响, 离线运行零网络请求。
    """
    enabled = bool(s.langsmith_tracing_v2) and bool(s.langsmith_api_key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGCHAIN_API_KEY"] = s.langsmith_api_key  # type: ignore[assignment]
        os.environ["LANGCHAIN_PROJECT"] = s.langsmith_project or "KnowHub"
        os.environ["LANGCHAIN_ENDPOINT"] = s.langsmith_endpoint


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
# 必须在 langchain / langgraph 被 import 之前完成 (本模块通常也是应用最早
# import 的模块之一); 即使晚于 import, 二者的 tracing 开关也是运行时惰性读取。
apply_langsmith_env(settings)
