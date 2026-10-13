"""测试文档入库链路的容错行为.

覆盖两个修复:
1. HyDE 元数据增强里「单个 chunk 的 LLM 调用失败」不应导致整次上传 500.
   文档字符串承诺的是「降级为抽取式, 不影响入库成功」, 修复前 asyncio.gather
   会把第一个异常直接冒泡成 500, 用户白等几分钟.
2. 入库失败 (解析/切分/写索引抛异常) 时, 已落盘的原始上传文件要被清理,
   不在 data/uploads/ 里堆积孤儿文件.
"""
from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient

from tests._fakes import (
    FakeBM25Store,
    FakeEmbeddingService,
    FakeLLM,
    FakeReranker,
    FakeVectorStore,
)


@pytest.fixture
def client(monkeypatch, tmp_path):
    """patch 掉 embedding / vector / bm25 / reranker / llm 的 TestClient."""
    from app.agents import llm_router as lr_mod
    from app.api import chat as chat_mod
    from app.rag import bm25_store as bm_mod
    from app.rag import embedding as emb_mod
    from app.rag import reranker as rk_mod
    from app.rag import retriever as rt_mod
    from app.rag import vector_store as vs_mod

    monkeypatch.chdir(tmp_path)
    os.environ["UNIKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()
    monkeypatch.setattr(vs_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(bm_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(rk_mod, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(rt_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(rt_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(rt_mod, "reset_retriever_cache", lambda: None)
    # 注意签名必须是 (*a, **kw): `get_llm_text_callable()` 内部是按
    # `get_llm(provider, model)` 两个位置参数调用的, 用 `lambda: FakeLLM()`
    # 这种零参版本会抛 TypeError, 而 documents.py 拿 LLM 的那段被
    # `except Exception -> 降级抽取式` 包着, 异常会被静默吞掉 —— 结果是
    # HyDE 的 LLM 分支从来没被走到, 测试里 `calls["n"] > 0` 恒不成立,
    # 用例看着在测容错其实一直在测降级路径。
    monkeypatch.setattr(lr_mod, "get_llm", lambda *a, **kw: FakeLLM())
    monkeypatch.setattr(chat_mod, "get_llm", lambda *a, **kw: FakeLLM())

    from app.main import app

    return TestClient(app)


def _register(c: TestClient, username: str = "alice") -> str:
    r = c.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": "password123"},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _upload(c: TestClient, token: str, content: str, kb_id: str, filename: str = "doc.txt"):
    return c.post(
        "/api/v1/documents/upload",
        files={"file": (filename, io.BytesIO(content.encode()), "text/plain")},
        data={"kb_id": kb_id},
        headers={"Authorization": f"Bearer {token}"},
    )


def test_single_chunk_hyde_failure_does_not_fail_upload(client, monkeypatch):
    """第 1 个 chunk 的 HyDE LLM 调用抛异常, 上传仍应成功入库."""
    from app.core.config import get_settings
    from app.rag import metadata as md_mod

    monkeypatch.setenv("HYDE_ENABLED", "true")
    monkeypatch.setenv("HYDE_INDEX_MAX_CHUNKS", "5")
    get_settings.cache_clear()

    calls = {"n": 0}
    orig = md_mod.MetadataEnhancer.enrich_with_hyde

    def _flaky(self, text, llm=None):
        if llm is not None:
            calls["n"] += 1
            raise RuntimeError("LLM timeout")
        return orig(self, text, llm=None)

    monkeypatch.setattr(md_mod.MetadataEnhancer, "enrich_with_hyde", _flaky)

    token = _register(client, "hyde_user")
    # 内容足够长, 保证会切成多个 chunk.
    content = "这是一段用于测试的中文文档内容。" * 60
    r = _upload(client, token, content, "hyde_user_private")

    assert r.status_code == 200, f"单 chunk HyDE 失败不应 500: {r.text}"
    assert r.json()["status"] == "indexed"
    assert r.json()["chunks"] > 0
    assert calls["n"] > 0, "测试本身要确认 LLM 路径真的被调用过"


def test_all_chunk_hyde_failures_still_index(client, monkeypatch):
    """即使 LLM 路径全挂, 每个 chunk 也应拿到抽取式元数据并入库."""
    from app.core.config import get_settings
    from app.rag import metadata as md_mod

    monkeypatch.setenv("HYDE_ENABLED", "true")
    monkeypatch.setenv("HYDE_INDEX_MAX_CHUNKS", "10")
    get_settings.cache_clear()

    orig = md_mod.MetadataEnhancer.enrich_with_hyde

    def _always_fail(self, text, llm=None):
        if llm is not None:
            raise RuntimeError("upstream 502")
        return orig(self, text, llm=None)

    monkeypatch.setattr(md_mod.MetadataEnhancer, "enrich_with_hyde", _always_fail)

    token = _register(client, "hyde_user2")
    content = "知识库内容测试。" * 40
    r = _upload(client, token, content, "hyde_user2_private")

    assert r.status_code == 200, r.text
    assert r.json()["chunks"] > 0


def test_failed_ingest_removes_uploaded_file(client, monkeypatch, tmp_path):
    """入库失败时, data/uploads/ 下的原始文件必须被清掉."""
    from app.api import documents as doc_mod
    from app.rag import chunker as ch_mod

    token = _register(client, "clean_user")
    content = "会触发切分失败的内容。" * 20

    def _boom(*_args, **_kwargs):
        raise RuntimeError("chunker exploded")

    monkeypatch.setattr(ch_mod.TextChunker, "split", _boom)

    # 判据是"本次请求没有新增残留", 而不是"整个上传目录为空":
    # 同文件里排在前面的用例会**成功**上传文档, 那些文件本来就应当留在
    # data/uploads/ (它们是已入库文档的原始件)。用"目录为空"当断言会把别人的
    # 正常产物当成孤儿, 于是单跑这个文件、或者换一下用例顺序就红。
    before = set(doc_mod.UPLOAD_DIR.glob("doc_*.txt"))
    r = _upload(client, token, content, "clean_user_private")
    assert r.status_code == 500, r.text

    added = set(doc_mod.UPLOAD_DIR.glob("doc_*.txt")) - before
    assert not added, f"入库失败后不应残留上传文件, 实际残留: {added}"


def test_empty_document_rejected_and_file_removed(client, monkeypatch):
    """空文档被 400 拒绝时, 同样不留孤儿文件."""
    from app.api import documents as doc_mod

    token = _register(client, "empty_user")
    # 同上: 只比对本次请求前后的**增量**, 不要求整个目录为空.
    before = set(doc_mod.UPLOAD_DIR.glob("doc_*.txt"))
    r = _upload(client, token, "   \n\t  ", "empty_user_private")

    assert r.status_code == 400, r.text
    uploads = doc_mod.UPLOAD_DIR
    added = set(uploads.glob("doc_*.txt")) - before
    assert not added, f"空文档被拒后不应残留上传文件, 实际残留: {added}"
