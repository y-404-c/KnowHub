"""Document schemas.

上传 / 列出 / 删除文档相关接口的入参与出参定义.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    """list 接口返回的单个文档元信息."""

    doc_id: str = Field(..., description="文档唯一 ID, 形如 doc_<12位hex>")
    filename: str = Field(..., description="原始上传文件名")
    kb_id: str = Field(..., description="所属知识库 ID")
    chunks: int = Field(..., description="该文档被切分出的 chunk 数量")
    status: str = Field(..., description="处理状态, 成功入库为 'indexed'")
    created_at: datetime = Field(..., description="入库时间 (UTC)")
    metadata: dict = Field(default={}, description="附加元信息 (预留扩展位)")


class DocumentUploadResponse(BaseModel):
    """upload 接口的成功响应."""

    doc_id: str = Field(..., description="后端新分配的文档 ID")
    filename: str = Field(..., description="原始上传文件名")
    chunks: int = Field(..., description="实际切分出的 chunk 数")
    status: str = Field(default="indexed", description="固定返回 'indexed'")
    message: Optional[str] = Field(default=None, description="人类可读的提示文案")
