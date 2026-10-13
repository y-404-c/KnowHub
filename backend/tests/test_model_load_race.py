"""测试 embedding / reranker 模型懒加载的线程安全.

背景: 两个服务都是「单例 + 懒加载」, 而 FastAPI 的同步端点和
`asyncio.to_thread` 都会把请求放到线程池里并发执行. 修复前
`_ensure_model()` / `_ensure()` 是裸的 `if self._model is None`, 冷启动
时 N 个并发请求会各自 `SentenceTransformer(...)` 一遍 —— 重复占几百 MB
内存, 并撞上 torch 初始化本身不保证线程安全的问题.

验证方式: fake 模型的构造函数里 sleep 一段时间放大竞态窗口, 然后多线程
并发触发加载, 断言底层模型只被构造一次.
"""
from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest


def _install_fake_sentence_transformers(monkeypatch, init_delay: float = 0.05):
    """替换 sentence_transformers, 统计 SentenceTransformer / CrossEncoder 构造次数."""
    created = {"st": 0, "ce": 0}
    lock = threading.Lock()

    class _FakeSentenceTransformer:
        def __init__(self, model_name, device=None):
            with lock:
                created["st"] += 1
            time.sleep(init_delay)  # 放大竞态窗口
            self.model_name = model_name
            self.device = device

        def get_sentence_embedding_dimension(self):
            return 512

        def encode(self, texts, **_kwargs):
            # 真实 SentenceTransformer 返回 ndarray, 调用方会 .tolist().
            return np.asarray([[0.1] * 4 for _ in texts], dtype=np.float32)

    class _FakeCrossEncoder:
        def __init__(self, model_name):
            with lock:
                created["ce"] += 1
            time.sleep(init_delay)
            self.model_name = model_name

        def predict(self, pairs, **_kwargs):
            return [0.5 for _ in pairs]

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    fake_mod.CrossEncoder = _FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    return created


def _run_concurrently(fn, n_threads: int = 8):
    """并发跑 fn, 等所有线程结束后收集异常."""
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def _worker():
        try:
            barrier.wait()  # 尽量让所有线程同时进入临界区
            fn()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "有线程没跑完, 测试卡死"
    assert not errors, f"并发调用抛异常: {errors!r}"


def test_embedding_model_loaded_once_under_concurrency(monkeypatch):
    """并发首次 embed 时, SentenceTransformer 只能构造一次."""
    created = _install_fake_sentence_transformers(monkeypatch)

    from app.rag.embedding import EmbeddingService

    svc = EmbeddingService(model_name="fake-model")
    _run_concurrently(lambda: svc.embed(["hello"]))

    assert created["st"] == 1, (
        f"并发冷启动只应加载 1 次 embedding 模型, 实际 {created['st']} 次"
    )


def test_reranker_model_loaded_once_under_concurrency(monkeypatch):
    """并发首次 rerank 时, CrossEncoder 只能构造一次."""
    created = _install_fake_sentence_transformers(monkeypatch)

    from app.rag.reranker import CrossEncoderReranker

    rk = CrossEncoderReranker(model_name="fake-model")
    _run_concurrently(lambda: rk.rerank("q", [{"document": "x"}], top_k=1))

    assert created["ce"] == 1, (
        f"并发冷启动只应加载 1 次 rerank 模型, 实际 {created['ce']} 次"
    )


def test_embedding_model_not_reloaded_after_warm(monkeypatch):
    """已加载完成后, 后续并发调用不应再触发加载, 且返回同一对象."""
    created = _install_fake_sentence_transformers(monkeypatch)

    from app.rag.embedding import EmbeddingService

    svc = EmbeddingService(model_name="fake-model")
    first = svc._ensure_model()
    _run_concurrently(lambda: svc._ensure_model())
    _run_concurrently(lambda: svc.embed(["a", "b"]))

    assert created["st"] == 1
    assert all(svc._ensure_model() is first for _ in range(4))


def test_failed_model_load_is_retried(monkeypatch):
    """加载失败时不应该把半初始化的对象留在 self._model 上.

    `_model` 只在构造成功后才赋值, 所以下一次调用会重新尝试加载,
    而不是永久卡在一个坏实例上.
    """
    created = _install_fake_sentence_transformers(monkeypatch, init_delay=0.0)

    from app.rag import embedding as emb_mod
    import sys as _sys

    attempts = {"n": 0}

    class _Flaky:
        def __init__(self, model_name, device=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("model download failed")
            self.dim = 512

        def get_sentence_embedding_dimension(self):
            return 512

        def encode(self, texts, **_kwargs):
            return np.asarray([[0.1] * 4 for _ in texts], dtype=np.float32)

    monkeypatch.setattr(_sys.modules["sentence_transformers"], "SentenceTransformer", _Flaky)

    svc = emb_mod.EmbeddingService(model_name="flaky-model")
    with pytest.raises(RuntimeError):
        svc.embed(["hello"])
    assert svc._model is None, "加载失败后 _model 应保持 None, 允许下次重试"

    # 第二次应该重新尝试并成功.
    out = svc.embed(["hello"])
    assert attempts["n"] == 2
    assert len(out) == 1
    assert created["st"] == 0, "这次用的是 _Flaky, 不该走 fake"
