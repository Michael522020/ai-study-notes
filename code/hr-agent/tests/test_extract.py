"""文本抽取测试。用临时目录现场造 docx / pdf，不依赖外部样例文件。"""

from __future__ import annotations

import zipfile

from hragent.extract import extract_text


def test_extract_plain_text(tmp_path):
    p = tmp_path / "resume.txt"
    p.write_text("张三\n本科\n6年外贸经验\n", encoding="utf-8")
    doc = extract_text(p)
    assert doc.ok
    assert "张三" in doc.text
    assert "6年外贸经验" in doc.text


def test_extract_gbk_text(tmp_path):
    p = tmp_path / "resume.txt"
    p.write_bytes("王芳 女 32岁 外贸".encode("gb18030"))
    doc = extract_text(p)
    assert doc.ok
    assert "王芳" in doc.text


def test_extract_docx(tmp_path):
    p = tmp_path / "resume.docx"
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r><w:t>张伟 本科</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>主导客户开发</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("word/document.xml", xml)
    doc = extract_text(p)
    assert doc.ok
    assert "张伟 本科" in doc.text
    assert "主导客户开发" in doc.text
    # 段落应被拆成两行
    assert len([ln for ln in doc.text.splitlines() if ln.strip()]) == 2


def test_extract_docx_without_document_xml(tmp_path):
    p = tmp_path / "broken.docx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("hello.txt", "not a docx")
    doc = extract_text(p)
    assert not doc.ok
    assert doc.error == "docx_missing_document_xml"


def test_pdf_without_text_layer_requests_ocr(tmp_path):
    p = tmp_path / "scan.pdf"
    # 只有图片对象，没有文本层
    p.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Page >>\nstream\n<<binary>>\nendstream\n%%EOF")
    doc = extract_text(p)
    assert not doc.ok
    assert doc.needs_ocr
    assert doc.text == ""


def test_unsupported_suffix(tmp_path):
    p = tmp_path / "resume.doc"
    p.write_bytes(b"\xd0\xcf\x11\xe0")
    doc = extract_text(p)
    assert not doc.ok
    assert doc.error == "legacy_doc_unsupported"


def test_missing_file(tmp_path):
    doc = extract_text(tmp_path / "nope.txt")
    assert not doc.ok
    assert doc.error == "file_not_found"
