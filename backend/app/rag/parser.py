"""文档解析器 (Document Parser) —— 入库链路的第一道关卡.

在 `api/documents.py` 的 upload 流程里, 用户上传的原始文件首先要被"读成纯文本",
才能交给后续 cleaner -> chunker -> embedding -> 存储. 本模块就负责这步:

    支持的格式 (见 `SUPPORTED`):
      - .pdf   : 用 pypdf 逐页抽取文本 (跳过空页), 页间用空行分隔.
      - .md/.markdown/.txt : 直接以 UTF-8 读全文 (errors="ignore" 容忍坏字节).
      - .docx  : 用 python-docx 逐段落抽取, 段落间用空行分隔.
      - .png/.jpg/.jpeg : 走 OCR (pytesseract, 中英文 chi_sim+eng);
                          没装 tesseract 时优雅降级返回空串, 不让上传整体失败.

设计要点:
  - 解析是**纯 IO / CPU 操作**, 调用方 (documents.py) 会用 `asyncio.to_thread`
    把它卸到线程池, 避免阻塞 FastAPI 事件循环.
  - 任何单页/单段落解析异常都被吞掉并记为空, 不让一份坏页拖垮整篇文档.
  - 不支持的后缀直接抛 ValueError, 由上游转成 400.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from loguru import logger


class DocumentParser:
    # 入库支持的扩展名白名单; documents.py 在保存前会先校验后缀落在此集合内.
    SUPPORTED = {".pdf", ".md", ".markdown", ".txt", ".docx", ".png", ".jpg", ".jpeg"}

    def parse(self, file_path: str) -> str:
        """按后缀分派到对应的私有解析方法, 返回拼接后的纯文本.

        Returns:
            纯文本 (不同页/段之间用 "\\n\\n" 分隔). 解析全失败则返回 "".
        Raises:
            ValueError: 后缀不在 SUPPORTED 内.
        """
        p = Path(file_path)
        suffix = p.suffix.lower()
        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {suffix}")
        try:
            if suffix == ".pdf":
                return self._parse_pdf(p)
            if suffix in {".md", ".markdown"}:
                return p.read_text(encoding="utf-8", errors="ignore")
            if suffix == ".txt":
                return p.read_text(encoding="utf-8", errors="ignore")
            if suffix == ".docx":
                return self._parse_docx(p)
            if suffix in {".png", ".jpg", ".jpeg"}:
                return self._parse_image_ocr(p)
        except Exception as e:
            logger.exception(f"Parse failed: {file_path}: {e}")
            raise
        return ""

    def _parse_pdf(self, p: Path) -> str:
        """PDF 解析: 用 pypdf 逐页 extract_text, 跳过抽不出文字的页 (如纯图片页).

        每页之间用 "\\n\\n" 连接, 保留基本的段落/页边界, 方便后续 cleaner 处理.
        """
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                texts.append(t)
        return "\n\n".join(texts)

    def _parse_docx(self, p: Path) -> str:
        """DOCX 解析: 用 python-docx 取每个 paragraph 的文本, 跳过空段落.

        段落间用 "\\n\\n" 连接, 保留文档结构层级 (标题/正文都是 paragraph).
        """
        from docx import Document

        doc = Document(str(p))
        return "\n\n".join(par.text for par in doc.paragraphs if par.text.strip())

    def _parse_image_ocr(self, p: Path) -> str:
        """图片 OCR: 用 pytesseract 做中英文识别 (lang="chi_sim+eng").

        依赖系统装了 Tesseract OCR 引擎; 若未安装或导入失败, 打印 warning 并返回
        空串 (降级而非抛错), 确保"图片解析不了"不会让整个上传链路 500.
        """
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(str(p))
            return pytesseract.image_to_string(img, lang="chi_sim+eng")
        except Exception as e:
            logger.warning(f"OCR unavailable: {e}; return empty")
            return ""
