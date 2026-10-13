"""Documents API: 上传、解析、切分、向量化、入库。"""
from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from loguru import logger

from app.api.deps import get_current_user
from app.core.config import DATA_DIR, get_settings
from app.core.kb_registry import ensure_kb_for_read, ensure_kb_for_write
from app.rag.chunker import TextChunker
from app.rag.cleaner import DocumentCleaner
from app.rag.metadata import MetadataEnhancer
from app.rag.parser import DocumentParser
from app.rag.retriever import get_retriever
from app.schemas.document import DocumentUploadResponse

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

def _resolve_upload_dir() -> Path:
    """上传文件根目录: 读 `settings.file_storage_dir`, 兜底 backend/data/uploads.

    为什么不再写成模块级 `UPLOAD_DIR = DATA_DIR / "uploads"`:
    那个硬编码确实挡住了 CWD 漂移, 但代价是**配置系统和测试隔离双双失效**:

      - `settings.file_storage_dir` 定义了却没有任何代码读它, 等于死配置。将来
        切 MinIO / 挂 NAS / 换 Docker volume 只能回来改代码 —— README Roadmap
        里的「MinIO 文件存储」正是被这行卡住。
      - conftest 用 `FILE_STORAGE_DIR` 环境变量把测试数据重定向到临时目录, 对
        一个不看配置的常量完全无效: 每跑一次测试就往开发者的
        `backend/data/uploads/` 里丢一批 `doc_*.txt`, 实测累积了近 200 个孤儿
        文件。更糟的是 `test_upload_resilience` 断言「上传目录无残留」时扫的
        正是这个被历史垃圾污染的目录, 于是必然失败。

    改读配置之后两个目标都保住: 显式配置优先; 相对路径由 `config._to_abs_path`
    validator 锚定到 backend/, CWD 漂移依然被挡住。
    """
    raw = (get_settings().file_storage_dir or "").strip()
    p = Path(raw) if raw else DATA_DIR / "uploads"
    # 正常情况下 settings 已把相对路径锚定成绝对路径, 这里再兜一次, 保证
    # 绕过 validator 直接构造 Settings(...) 的调用方也拿到确定位置.
    return p if p.is_absolute() else (DATA_DIR.parent / p)


# 模块级常量保留, 供测试断言残留文件时引用 (doc_mod.UPLOAD_DIR).
UPLOAD_DIR = _resolve_upload_dir()


def _ensure_upload_dir() -> Path:
    """延迟到每次请求都确保上传目录存在.

    每次重新解析而不是复用模块级常量: 测试里 settings / 环境变量可能在 fixture
    阶段被替换, 请求发生时读到的才是最新的目录。
    """
    upload_dir = _resolve_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _safe_unlink(path: Path) -> None:
    """尽力删除文件, 删不掉也不影响主流程 (比如文件已被别的清理逻辑移走)."""
    try:
        path.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to remove file {path}: {e}")


class _UploadTooLarge(Exception):
    """超过 upload_max_bytes 时抛出, 由路由统一处理 413."""

    def __init__(self, size: int, limit: int):
        self.size = size
        self.limit = limit
        super().__init__(f"upload too large: {size} > {limit}")


def _save_upload_bounded(file: UploadFile, dest: Path, limit: int) -> int:
    """把上传流 copy 到 dest, 边写边累加, 超过 limit 立刻抛 _UploadTooLarge.

    设计要点:
    1. 不一次性 read() 整个文件, 避免 OOM.
    2. 超过 limit 时立即停止, 不再消耗输入流 (尽可能让上游尽早失败).
    3. 写入失败也要清理半成品文件.
    """
    total = 0
    chunk = 1024 * 1024  # 1MB
    try:
        with dest.open("wb") as f:
            while True:
                buf = file.file.read(chunk)
                if not buf:
                    break
                total += len(buf)
                if total > limit:
                    raise _UploadTooLarge(total, limit)
                f.write(buf)
    except BaseException:
        # 清理半成品, 不留垃圾文件
        _safe_unlink(dest)
        raise
    return total


async def _enrich_chunks(chunks, enhancer: MetadataEnhancer, settings) -> None:
    """给 chunk 批量补元数据; 前 N 个 chunk 走 LLM HyDE, 其余走抽取式回退.

    为什么要限流:
      入库侧 HyDE 是"每 chunk 一次 LLM 调用". 一个 200 chunk 的文档如果全量走 LLM,
      串行下来是分钟级延迟 + 200 次计费. 所以只给**文档开头的前 N 个 chunk**
      (信息密度最高、最可能被检索命中) 生成真实假设性问题, 其余回退首句摘要.

    容错:
      拿不到 LLM (缺 key / 离线 / 测试) 时整段降级为抽取式, 不影响入库成功.
    """
    llm = None
    max_llm = getattr(settings, "hyde_index_max_chunks", 0) or 0
    if settings.hyde_enabled and max_llm > 0:
        try:
            from app.agents.llm_router import get_llm_text_callable

            llm = get_llm_text_callable()
        except Exception as e:  # 缺 key / 未装 SDK, 静默降级
            logger.info(f"HyDE indexing falls back to extractive (no LLM): {e}")
            llm = None

    if llm is None:
        for c in chunks:
            c.metadata |= enhancer.enrich_with_hyde(c.text, llm=None)
        return

    sem = asyncio.Semaphore(max(1, getattr(settings, "hyde_index_concurrency", 4)))
    failed = {"n": 0}

    async def _one(c, use_llm: bool):
        if not use_llm:
            c.metadata |= enhancer.enrich_with_hyde(c.text, llm=None)
            return
        async with sem:
            try:
                # LLM 调用是阻塞 IO, 卸到 threadpool, 避免堵住事件循环.
                meta = await asyncio.to_thread(enhancer.enrich_with_hyde, c.text, llm)
            except Exception as e:
                # 单个 chunk 的 HyDE 失败不应该拖垮整次入库: 200 个 chunk 里第 3 个
                # 抖动一次就 500, 用户白等几分钟. 这里降级为抽取式元数据,
                # chunk 照样入库 (只是少了假设性问题这一路召回信号).
                failed["n"] += 1
                logger.warning(f"HyDE indexing failed for one chunk, fallback: {e}")
                c.metadata |= enhancer.enrich_with_hyde(c.text, llm=None)
                return
        c.metadata |= meta

    await asyncio.gather(*(_one(c, i < max_llm) for i, c in enumerate(chunks)))
    if failed["n"]:
        logger.warning(f"HyDE indexing: {failed['n']}/{len(chunks)} chunks degraded to extractive")


@router.post("/upload", response_model=DocumentUploadResponse, summary="上传文档并自动入库")
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Query("default", description="目标知识库 ID, 必须属于当前用户 (私有 kb 自动创建并归属)"),
    kb_id_form: str | None = Form(default=None, alias="kb_id"),
    user: str = Depends(get_current_user),
):
    # 兼容: 上传时 kb_id 既可以在 query string 也可以在 multipart/form 中;
    # form 优先 (前端两个接口都用得上).
    resolved_kb_id = kb_id_form if kb_id_form is not None else kb_id
    kb_id = resolved_kb_id
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件名")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in DocumentParser.SUPPORTED:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix}")

    # 0) ACL: 若 kb 不存在则自动创建并归属当前用户; 若存在则必须是当前用户的 kb.
    #    公共 kb (settings.public_kb_ids, 默认 "default") 不允许写入.
    ensure_kb_for_write(kb_id, user)

    # 1) 保存到本地 (边写边校验大小, 超额立刻 413)
    settings = get_settings()
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    upload_dir = _ensure_upload_dir()
    dest = upload_dir / f"{doc_id}{suffix}"
    try:
        await asyncio.to_thread(_save_upload_bounded, file, dest, settings.upload_max_bytes)
    except _UploadTooLarge as e:
        logger.warning(f"upload too large: user={user} size={e.size} limit={e.limit} filename={file.filename}")
        raise HTTPException(
            status_code=413,
            detail=f"文件过大: {e.size} 字节, 上限 {e.limit} 字节",
        )

    # 2) 解析 + 清洗 + 自适应切分 + 元数据增强 + 入库
    #    (CPU 密集: 解析+切分+embedding+向量/BM25 写盘)
    try:
        raw = await asyncio.to_thread(DocumentParser().parse, str(dest))
        if not raw.strip():
            raise HTTPException(status_code=400, detail="文档内容为空或解析失败")
        # 2.1) 文档清洗: 42 条正则去除噪声 (控制字符/页眉页脚/标记语言残留等).
        text = await asyncio.to_thread(
            DocumentCleaner(enabled=settings.cleaner_enabled).clean, raw
        )
        if not text.strip():
            raise HTTPException(status_code=400, detail="清洗后文档内容为空")
        # 2.2) 自适应切分: 短文档整篇, 长文档按 adaptive_chunk_size 切.
        chunks = await asyncio.to_thread(
            TextChunker(
                adaptive=settings.adaptive_chunking_enabled,
                adaptive_chunk_size=settings.adaptive_chunk_size,
                adaptive_short_doc_chars=settings.adaptive_short_doc_chars,
            ).split,
            text,
            doc_id=doc_id,
        )
        if not chunks:
            raise HTTPException(status_code=400, detail="切片为空")
        # 2.3) 元数据增强 / HyDE: 给每个 chunk 追加关键词/摘要/假设性问题
        #      (无 LLM 时回退到抽取式摘要, 保证入库链路零外部依赖).
        enhancer = MetadataEnhancer(hyde_enabled=settings.hyde_enabled)
        await _enrich_chunks(chunks, enhancer, settings)
        retriever = get_retriever(kb_id)
        ids = [c.metadata["chunk_id"] for c in chunks]
        docs = [c.text for c in chunks]
        metas = [c.metadata | {"filename": file.filename, "user": user} for c in chunks]
        await retriever.add_documents_async(ids=ids, documents=docs, metadatas=metas)
        logger.info(f"Indexed {file.filename} -> {len(chunks)} chunks (doc_id={doc_id})")
        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            chunks=len(chunks),
            status="indexed",
            message=f"成功入库 {len(chunks)} 个 chunk",
        )
    except HTTPException:
        # 解析/切分失败时, 已经落盘的原始文件是孤儿, 不清理会在 data/uploads/
        # 里一直堆积 (每个失败的上传留一份). 顺手删掉脏数据.
        _safe_unlink(dest)
        raise
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        _safe_unlink(dest)
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")


@router.get("/list", summary="列出当前知识库 chunk 数")
async def list_documents(kb_id: str = "default", user: str = Depends(get_current_user)):
    # ACL: 公共 kb 直接放行; 私有 kb 必须 owner == 当前用户
    ensure_kb_for_read(kb_id, user)
    retriever = get_retriever(kb_id)
    return {
        "kb_id": kb_id,
        "bm25_count": retriever.bm25_store.count(),
        "vector_count": retriever.vector_store.count(),
    }


@router.delete("/{doc_id}", summary="删除文档及其所有 chunk")
async def delete_document(
    doc_id: str,
    kb_id: str = "default",
    user: str = Depends(get_current_user),
):
    """删除指定 doc_id 的文档: 从 vector store / BM25 中移除其所有 chunk.

    删除后 BM25 标记 dirty, 下次 query 时惰性重建, 避免立即重建拖慢响应.
    """
    ensure_kb_for_write(kb_id, user)
    retriever = get_retriever(kb_id)
    deleted = await asyncio.to_thread(retriever.delete_by_doc_id, doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="文档不存在或已删除")

    # 顺手清理上传的原始文件 (如果还在)
    try:
        for f in _ensure_upload_dir().glob(f"{doc_id}.*"):
            await asyncio.to_thread(f.unlink)
    except Exception:
        pass

    logger.info(f"Deleted doc_id={doc_id} from kb={kb_id}, chunks={deleted}, user={user}")
    return {"status": "deleted", "doc_id": doc_id, "kb_id": kb_id, "chunks_deleted": deleted}
