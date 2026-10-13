"""共享 fixtures: 把 app 加入 sys.path 让 import 跑得通."""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# 让 `from app.xxx import ...` 跑得通, 不依赖 src layout
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pytest_configure() 里建出来的临时数据目录, 会话结束时清理.
_TMP_DATA_ROOT: str | None = None


def _cleanup_tmp_data_root() -> None:
    global _TMP_DATA_ROOT
    if _TMP_DATA_ROOT and os.path.isdir(_TMP_DATA_ROOT):
        shutil.rmtree(_TMP_DATA_ROOT, ignore_errors=True)
    _TMP_DATA_ROOT = None


def pytest_configure(config):  # noqa: ARG001
    """把测试用到的持久化目录全部重定向到临时目录.

    为什么必须在 pytest_configure 里做:
      `app.core.config` 在模块级就 `settings = get_settings()`, 而
      `app.db.session` 又在模块级按 settings 建 engine. 两者都发生在
      **第一次 import app.* 时**, 也就是测试模块被收集/导入的那一刻.
      pytest_configure 早于收集, 所以这里是唯一能赶在 app 被 import 之前
      把环境变量铺好的钩子.

    不隔离的后果 (都是真实踩过的):
      1. 测试直接写开发者的 `backend/data/unikb.db`, 每次跑测试都往里塞
         alice/bob/... 这些测试账号;
      2. 反过来, 测试还依赖"库是干净的"——同一个文件第二次跑就会因为
         "用户已存在" 全线 400, 单测不可重复执行;
      3. bm25/chroma/uploads 的 pkl 和上传文件同样散落在 backend/data/.

    逃生口: 显式设置 UNIKB_TEST_DATA_DIR 或对应的 *_DIR / DATABASE_URL
    环境变量时, 尊重调用方的选择, 不覆盖.
    """
    global _TMP_DATA_ROOT

    override_root = os.environ.get("UNIKB_TEST_DATA_DIR")
    data_root = Path(override_root) if override_root else Path(tempfile.mkdtemp(prefix="unikb_testdata_"))
    data_root.mkdir(parents=True, exist_ok=True)

    if not override_root:
        _TMP_DATA_ROOT = str(data_root)
        atexit.register(_cleanup_tmp_data_root)

    def _setdefault(env_key: str, value: str) -> None:
        # 已显式指定的不动; 否则指到临时目录.
        os.environ.setdefault(env_key, value)

    # sqlite 用正斜杠, 避免 Windows 反斜杠在 URL 里被当转义.
    db_path = (data_root / "unikb_test.db").as_posix()
    _setdefault("DATABASE_URL", f"sqlite:///{db_path}")
    # 整棵数据目录一起搬走: kb_registry.json 不走下面的 *_PERSIST_DIR 配置,
    # 而是直接挂在 DATA_DIR 上。只搬那几个子目录的话, 测试仍会写开发者真实的
    # backend/data/kb_registry.json, 一次 pytest 就把本地知识库注册表冲掉。
    _setdefault("UNIKB_DATA_DIR", str(data_root))
    for key, sub in (
        ("FILE_STORAGE_DIR", "files"),
        ("CHROMA_PERSIST_DIR", "chroma"),
        ("BM25_PERSIST_DIR", "bm25"),
        ("HISTORY_STORAGE_DIR", "history"),
    ):
        _setdefault(key, (data_root / sub).as_posix())


def pytest_unconfigure(config):  # noqa: ARG001
    _cleanup_tmp_data_root()


@pytest.fixture(scope="session", autouse=True)
def _ensure_db_schema():
    """保证测试库的 schema 就绪, 让每个测试文件都能独立跑.

    为什么要这个 fixture:
        `create_tables()` 只在 FastAPI 的 lifespan 里被调用, 而 `TestClient(app)`
        **必须作为上下文管理器使用才会触发 lifespan** (`with TestClient(app) as c`)。
        写成 `return TestClient(app)` 的 fixture 不会建表 —— 这类测试之所以平时
        能过, 纯粹是因为同一次 pytest 会话里前面某个文件用过 `with TestClient(app)`,
        其副作用把表建好了。于是测试之间产生了隐式顺序依赖: 单独跑该文件、
        或调整文件收集顺序, 就会全线 `OperationalError: no such table: users`
        (test_upload_resilience.py 单独跑 4 个全挂, 全量跑只挂 3 个, 就是这么来的)。

        建 schema 属于测试基础设施职责, 不该依赖"某个测试恰好触发了应用启动",
        所以在这里统一做一次, 一次性解开所有测试文件的这层耦合。
    """
    from app.db import create_tables

    create_tables()
    yield


@pytest.fixture(scope="session")
def settings():
    """从 .env.example 强制加载一份测试用 settings."""
    from app.core.config import Settings

    return Settings(_env_file=ROOT / ".env.example")
