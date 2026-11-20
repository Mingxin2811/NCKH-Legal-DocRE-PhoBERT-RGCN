"""Read DOCX/TXT without changing text after offsets have been predicted."""

import json
import hashlib
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from xml.etree import ElementTree as ET

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _paragraph_text(element):
    parts = []
    def visit(node):
        if node.tag in {W_NS + "del", W_NS + "instrText"}:
            return
        if node.tag == W_NS + "t":
            parts.append(node.text or "")
        elif node.tag == W_NS + "tab":
            parts.append("\t")
        elif node.tag in {W_NS + "br", W_NS + "cr"}:
            parts.append("\n")
        else:
            for child in node:
                visit(child)
    visit(element)
    return "".join(parts)


def _docx_blocks(parent):
    blocks = []
    for child in parent:
        if child.tag == W_NS + "p":
            blocks.append(_paragraph_text(child))
        elif child.tag == W_NS + "tbl":
            for row in child.findall(W_NS + "tr"):
                cells = ["\n".join(_docx_blocks(cell)) for cell in row.findall(W_NS + "tc")]
                blocks.append("\t".join(cells))
        elif child.tag in {W_NS + "sdt", W_NS + "sdtContent", W_NS + "ins"}:
            blocks.extend(_docx_blocks(child))
    return blocks


def read_input_document(path):
    """Body paragraphs/tables in document order; scanned images require OCR separately."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp: {path}")
    suffix = path.suffix.lower()
    if suffix == ".docx":
        try:
            with ZipFile(path) as archive:
                root = ET.fromstring(archive.read("word/document.xml"))
                body = root.find(W_NS + "body")
                text = "\n".join(_docx_blocks(body)) if body is not None else ""
        except (BadZipFile, KeyError, ET.ParseError) as exc:
            raise ValueError(f"DOCX không hợp lệ hoặc có mật khẩu: {path.name}") from exc
    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8-sig")
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(raw, list) and len(raw) == 1:
            raw = raw[0]
        text = raw.get("full_text") if isinstance(raw, dict) else None
        if not isinstance(text, str):
            raise ValueError("JSON phải có full_text; nhãn cung cấp không được dùng khi suy luận.")
    else:
        raise ValueError("Chỉ hỗ trợ .docx, .txt hoặc JSON có full_text; .doc/PDF cần chuyển đổi trước.")
    if not text.strip():
        raise ValueError("Không đọc được văn bản. Nếu tệp chỉ chứa ảnh quét, cần OCR trước.")
    return text, {
        "name": path.name,
        "format": suffix,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "extraction_note": "DOCX: thân văn bản và bảng theo thứ tự; không lấy header/footer, chú thích hoặc OCR ảnh. Offset tính trên input_text.txt, không phải số trang Word.",
    }
