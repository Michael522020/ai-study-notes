"""简历 / JD 文本抽取。支持 txt、md、docx、pdf。

为什么自己写而不用 pypdf / python-docx：
1. 零第三方依赖，`pip install` 完就能跑，读者复现成本最低；
2. 这段代码本身就是笔记的一部分 —— **解析质量直接决定下游所有环节的质量**，
   用黑盒不如看清边界。

已知边界（务必看）：
- PDF 只处理**有文本层**的文档；扫描件（图片型 PDF）抽不出文字，
  生产环境需要接 OCR，并在抽取失败时明确返回「需 OCR」而不是空字符串。
- docx 的表格会被拉平成纯文本，单元格顺序保留但不还原版式。
"""

from __future__ import annotations

import html
import re
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """抽取结果。`ok=False` 时 `error` 说明原因。"""

    text: str
    source: str
    ok: bool = True
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_ocr(self) -> bool:
        return not self.ok and self.error == "needs_ocr"


def extract_text(path: str | Path) -> Document:
    """按扩展名分派抽取器。"""
    p = Path(path)
    if not p.exists():
        return Document(text="", source=str(p), ok=False, error="file_not_found")
    suffix = p.suffix.lower()
    if suffix in (".txt", ".md", ".markdown", ".csv", ".json"):
        return _extract_plain(p)
    if suffix == ".docx":
        return _extract_docx(p)
    if suffix == ".pdf":
        return _extract_pdf(p)
    if suffix == ".doc":
        return Document(
            text="",
            source=str(p),
            ok=False,
            error="legacy_doc_unsupported",
        )
    return Document(text="", source=str(p), ok=False, error=f"unsupported_suffix:{suffix}")


def _extract_plain(p: Path) -> Document:
    try:
        raw = p.read_bytes()
    except OSError as exc:
        return Document(text="", source=str(p), ok=False, error=str(exc))
    for enc in ("utf-8", "gb18030", "utf-16"):
        try:
            return Document(text=raw.decode(enc), source=str(p), meta={"encoding": enc})
        except UnicodeDecodeError:
            continue
    return Document(text=raw.decode("utf-8", errors="replace"), source=str(p), meta={"encoding": "replace"})


_W_PARA = re.compile(r"</w:p>")
_W_TAB = re.compile(r"<w:tab\s*/>")
_W_BR = re.compile(r"<w:br\s*/>")
_TAG = re.compile(r"<[^>]+>")


def _extract_docx(p: Path) -> Document:
    try:
        with zipfile.ZipFile(p) as zf:
            names = zf.namelist()
            if "word/document.xml" not in names:
                return Document(text="", source=str(p), ok=False, error="docx_missing_document_xml")
            xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    except (OSError, zipfile.BadZipFile) as exc:
        return Document(text="", source=str(p), ok=False, error=f"docx_error:{exc}")

    xml = _W_PARA.sub("\n", xml)
    xml = _W_TAB.sub("\t", xml)
    xml = _W_BR.sub("\n", xml)
    text = html.unescape(_TAG.sub("", xml))
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return Document(text=text, source=str(p), meta={"paragraphs": len(lines)})


_PDF_STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.S)
_PDF_TEXT_OP = re.compile(rb"\((?:\\.|[^\\()])*\)")
_PDF_ESCAPES = {
    b"\\n": b"\n",
    b"\\r": b"\r",
    b"\\t": b"\t",
    b"\\b": b"\b",
    b"\\f": b"\f",
    b"\\(": b"(",
    b"\\)": b")",
    b"\\\\": b"\\",
}


def _decode_pdf_string(raw: bytes) -> str:
    """把 PDF 字面字符串里的转义还原。"""
    out = bytearray()
    i = 0
    while i < len(raw):
        if raw[i : i + 1] == b"\\" and i + 1 < len(raw):
            two = raw[i : i + 2]
            if two in _PDF_ESCAPES:
                out += _PDF_ESCAPES[two]
                i += 2
                continue
            if two[1:2].isdigit():
                octal = raw[i + 1 : i + 4]
                m = re.match(rb"[0-7]{1,3}", octal)
                if m:
                    out.append(int(m.group(), 8) & 0xFF)
                    i += 1 + len(m.group())
                    continue
            out += raw[i + 1 : i + 2]
            i += 2
            continue
        out += raw[i : i + 1]
        i += 1
    return out.decode("utf-8", errors="replace")


def _extract_pdf(p: Path) -> Document:
    """尽力抽取 PDF 文本层。没有文本层就返回 needs_ocr，不要返回空串糊弄下游。"""
    try:
        data = p.read_bytes()
    except OSError as exc:
        return Document(text="", source=str(p), ok=False, error=str(exc))

    chunks: list[str] = []
    for match in _PDF_STREAM.finditer(data):
        blob = match.group(1)
        for candidate in (blob, _try_inflate(blob)):
            if candidate is None:
                continue
            strings = _PDF_TEXT_OP.findall(candidate)
            if strings:
                chunks.append("\n".join(_decode_pdf_string(s[1:-1]) for s in strings))
                break

    text = "\n".join(c for c in chunks if c.strip())
    text = "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())
    if not text:
        return Document(
            text="",
            source=str(p),
            ok=False,
            error="needs_ocr",
            meta={"hint": "PDF 无文本层，需先做 OCR"},
        )
    return Document(text=text, source=str(p), meta={"pages_hint": data.count(b"/Type /Page")})


def _try_inflate(blob: bytes) -> bytes | None:
    try:
        return zlib.decompress(blob)
    except zlib.error:
        try:
            return zlib.decompressobj().decompress(blob)
        except zlib.error:
            return None


def extract_many(paths: list[str | Path]) -> list[Document]:
    return [extract_text(p) for p in paths]
