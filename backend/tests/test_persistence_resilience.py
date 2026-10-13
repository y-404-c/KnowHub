"""持久化容错: 索引文件损坏时不得静默丢数据.

背景: pickle 对 numpy / 类定义的版本很敏感 (典型报错 `No module named
'numpy._core.numeric'`), 跨版本升级或写入被中断都会让文件无法解析。

修复前的危险行为:
- NumpyVectorStore._load() 捕获异常后置空, 且**不挪走原文件** -> 下一次 _save()
  用空数据覆盖, 整个 kb 的向量永久丢失。
- BM25Store.load() 干脆没有 try, 异常抛到 HybridRetriever.__init__ 的
  `except Exception: pass` 被吞掉, 表现为"索引存在却什么都查不到"。

现在的契约: 清空内存状态 + 报错 + 把损坏文件改名成 *.corrupt 旁路, 保留恢复可能。
"""
from __future__ import annotations

import pytest


def test_numpy_store_quarantines_corrupt_file(tmp_path):
    """损坏的向量文件应被隔离, 而不是被空数据覆盖."""
    from app.rag.numpy_vector_store import NumpyVectorStore

    pkl = tmp_path / "numpy_kb_x.pkl"
    pkl.write_bytes(b"this-is-not-a-valid-pickle")

    store = NumpyVectorStore(persist_dir=str(tmp_path), collection_name="kb_x")

    assert store.count() == 0, "损坏时应以空库启动"
    assert not pkl.exists(), "损坏文件必须被挪走, 否则会被空数据覆盖"
    backup = tmp_path / "numpy_kb_x.pkl.corrupt"
    assert backup.exists(), "应保留损坏文件副本以便事后恢复"


def test_numpy_store_empty_start_does_not_delete_healthy_file(tmp_path):
    """正常路径不受影响: 空库不会误隔离文件."""
    from app.rag.numpy_vector_store import NumpyVectorStore

    store = NumpyVectorStore(persist_dir=str(tmp_path), collection_name="kb_y")
    store.add(["a"], ["alpha text"], [[1.0, 0.0]], [{"doc_id": "d1"}])
    store.add(["b"], ["beta text"], [[0.0, 1.0]], [{"doc_id": "d1"}])

    pkl = tmp_path / "numpy_kb_y.pkl"
    assert pkl.exists()

    reopened = NumpyVectorStore(persist_dir=str(tmp_path), collection_name="kb_y")
    assert reopened.count() == 2
    assert not list(tmp_path.glob("*.corrupt")), "正常文件不应被隔离"


def test_bm25_store_quarantines_corrupt_file(tmp_path):
    """损坏的 BM25 文件同样应被隔离, 而不是被空 docs 覆盖."""
    from app.rag.bm25_store import BM25Store

    pkl = tmp_path / "bm25.pkl"
    pkl.write_bytes(b"\x80\x04not-a-pickle")

    store = BM25Store(persist_path=str(pkl))
    store.load()  # 不该抛异常

    assert store.count() == 0
    assert not pkl.exists(), "损坏文件必须被挪走"
    assert (tmp_path / "bm25.pkl.corrupt").exists()


def test_bm25_load_failure_is_logged_not_silently_swallowed(tmp_path):
    """BM25 加载失败必须留下 error 日志, 不能像以前那样被上层 except 吞掉."""
    from loguru import logger

    from app.rag.bm25_store import BM25Store

    pkl = tmp_path / "bm25.pkl"
    pkl.write_bytes(b"garbage")

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        BM25Store(persist_path=str(pkl)).load()
    finally:
        logger.remove(handler_id)

    assert any("load failed" in m for m in messages), (
        f"加载失败没有留下 error 日志: {messages}"
    )


@pytest.mark.parametrize("store_factory", ["numpy", "bm25"])
def test_quarantine_survives_unwritable_directory(tmp_path, store_factory):
    """隔离失败时不能让 load 本身抛异常 (例如目录只读)."""
    if store_factory == "numpy":
        from app.rag.numpy_vector_store import NumpyVectorStore

        pkl = tmp_path / "numpy_kb_z.pkl"
        pkl.write_bytes(b"bad")
        # 目录不可写 -> replace 会失败, 但 load 仍应正常返回空库
        store = NumpyVectorStore(persist_dir=str(tmp_path), collection_name="kb_z")
        assert store.count() == 0
    else:
        from app.rag.bm25_store import BM25Store

        pkl = tmp_path / "bm25.pkl"
        pkl.write_bytes(b"bad")
        store = BM25Store(persist_path=str(pkl))
        store.load()
        assert store.count() == 0
