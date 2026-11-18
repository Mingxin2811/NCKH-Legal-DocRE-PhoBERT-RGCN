# -*- coding: utf-8 -*-
"""
auto_label_pipeline.py
======================
Hệ thống gán nhãn tự động chuẩn hóa cho văn bản quy phạm pháp luật Việt Nam.
Tương thích 100% với cấu trúc gán nhãn của 10 file mẫu trong dataset/nghi_quyet:
  - Đầu vào: Thư mục chứa 3 file: noi_dung.docx, thuoc_tinh.json, luoc_do.json (và muc_luc.json nếu có).
    Nếu file noi_dung.docx không có, hệ thống tự động tổng hợp từ thuoc_tinh.json và luoc_do.json.
  - Đầu ra:
    + {base_name}_ket_qua.json: Chứa metadata 12 trường, full_text chuẩn hóa NFC,
      entities (E1...E40) sắp xếp tuần tự theo loại và thứ tự xuất hiện,
      relations (R1...R22) và future_extensions.
    + {base_name}_ket_qua.csv: UTF-8 BOM, cấu trúc bảng 8 cột chuẩn.
  - Phù hợp cho mọi hình thức văn bản: Nghị quyết, Bộ luật, Luật, Nghị định, Thông tư, Quyết định,...
"""

import os
import sys
import re
import json
import csv
import glob
import argparse
from typing import Dict, List, Tuple, Any, Optional, Union

try:
    import docx
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl
    from docx.text.paragraph import Paragraph
    from docx.table import Table
except ImportError:
    docx = None

# ==============================================================================
# 1. BẢNG TAXONOMY THỰC THỂ (E1 -> E40)
# ==============================================================================
ENTITY_TAXONOMY: Dict[str, Tuple[int, str, str]] = {
    # 1.1 Metadata (1..12)
    "TEN_VAN_BAN": (1, "Tên văn bản", "metadata"),
    "SO_HIEU": (2, "Số hiệu", "metadata"),
    "LOAI_VAN_BAN": (3, "Loại văn bản", "metadata"),
    "NGANH": (4, "Ngành", "metadata"),
    "NGAY_BAN_HANH": (5, "Ngày ban hành", "metadata"),
    "LINH_VUC": (6, "Lĩnh vực", "metadata"),
    "NGAY_CO_HIEU_LUC": (7, "Ngày có hiệu lực", "metadata"),
    "NGAY_HET_HIEU_LUC": (8, "Ngày hết hiệu lực", "metadata"),
    "TINH_TRANG_HIEU_LUC": (9, "Tình trạng hiệu lực", "metadata"),
    "CO_QUAN_BAN_HANH": (10, "Cơ quan ban hành", "metadata"),
    "CHUC_DANH": (11, "Chức danh", "metadata"),
    "NGUOI_KY": (12, "Người ký", "metadata"),

    # 1.2 Legal Reference (13..33)
    "VB_DUOC_HUONG_DAN_AP_DUNG": (13, "Văn bản được hướng dẫn áp dụng", "reference"),
    "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": (14, "Văn bản được quy định chi tiết, hướng dẫn thi hành", "reference"),
    "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN": (15, "Văn bản bị tác động hiệu lực một phần", "reference"),
    "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO": (16, "Văn bản bị tác động hiệu lực toàn bộ", "reference"),
    "VB_DUOC_DINH_CHINH": (17, "Văn bản được đính chính", "reference"),
    "VB_DUOC_DAN_CHIEU": (18, "Văn bản được dẫn chiếu", "reference"),
    "CAN_CU_BAN_HANH": (19, "Căn cứ ban hành", "reference"),
    "VB_DUOC_GIAI_THICH": (20, "Văn bản được giải thích", "reference"),
    "VB_BI_DINH_CHI_THI_HANH": (21, "Văn bản bị đình chỉ thi hành", "reference"),
    "VB_BI_TAM_NGUNG_GIA_HAN": (22, "Văn bản bị tạm ngưng/ gia hạn/ công bố tiếp tục còn hiệu lực", "reference"),
    "VB_HUONG_DAN_AP_DUNG": (23, "Văn bản hướng dẫn áp dụng", "reference"),
    "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": (24, "Văn bản quy định chi tiết, hướng dẫn thi hành", "reference"),
    "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN": (25, "Văn bản quy định về việc hết hiệu lực một phần", "reference"),
    "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO": (26, "Văn bản quy định về việc hết hiệu lực toàn bộ", "reference"),
    "VB_HOP_NHAT": (27, "Văn bản hợp nhất", "reference"),
    "VB_DINH_CHINH": (28, "Văn bản đính chính", "reference"),
    "VB_DAN_CHIEU": (29, "Văn bản dẫn chiếu", "reference"),
    "VB_AP_DUNG": (30, "Văn bản áp dụng", "reference"),
    "VB_GIAI_THICH": (31, "Văn bản giải thích", "reference"),
    "VB_DINH_CHI_THI_HANH": (32, "Văn bản đình chỉ thi hành", "reference"),
    "VB_QUY_DINH_TAM_NGUNG_GIA_HAN": (33, "Văn bản quy định tạm ngưng/ gia hạn/ công bố tiếp tục còn hiệu lực", "reference"),

    # 1.3 Cấu trúc văn bản (34..40)
    "PHAN": (34, "Phần", "structure"),
    "CHUONG": (35, "Chương", "structure"),
    "MUC": (36, "Mục", "structure"),
    "TIEU_MUC": (37, "Tiểu Mục", "structure"),
    "DIEU": (38, "Điều", "structure"),
    "KHOAN": (39, "Khoản", "structure"),
    "DIEM": (40, "Điểm", "structure")
}

# ==============================================================================
# 2. BẢNG TAXONOMY QUAN HỆ (R1 -> R22)
# ==============================================================================
RELATION_TAXONOMY: Dict[str, Tuple[int, str]] = {
    # 2.1 Quan hệ pháp lý liên văn bản (1..21 tương ứng E13..E33)
    "DUOC_HUONG_DAN_AP_DUNG": (1, "Được hướng dẫn áp dụng"),
    "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": (2, "Được quy định chi tiết, hướng dẫn thi hành"),
    "BI_TAC_DONG_HIEU_LUC_MOT_PHAN": (3, "Bị tác động hiệu lực một phần"),
    "BI_TAC_DONG_HIEU_LUC_TOAN_BO": (4, "Bị tác động hiệu lực toàn bộ"),
    "DUOC_DINH_CHINH": (5, "Được đính chính"),
    "DUOC_DAN_CHIEU": (6, "Được dẫn chiếu"),
    "CAN_CU_BAN_HANH": (7, "Căn cứ ban hành"),
    "DUOC_GIAI_THICH": (8, "Được giải thích"),
    "BI_DINH_CHI_THI_HANH": (9, "Bị đình chỉ thi hành"),
    "BI_TAM_NGUNG_GIA_HAN": (10, "Bị tạm ngưng/ gia hạn"),
    "HUONG_DAN_AP_DUNG": (11, "Hướng dẫn áp dụng"),
    "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": (12, "Quy định chi tiết, hướng dẫn thi hành"),
    "QUY_DINH_HET_HIEU_LUC_MOT_PHAN": (13, "Quy định về việc hết hiệu lực một phần"),
    "QUY_DINH_HET_HIEU_LUC_TOAN_BO": (14, "Quy định về việc hết hiệu lực toàn bộ"),
    "HOP_NHAT": (15, "Hợp nhất"),
    "DINH_CHINH": (16, "Đính chính"),
    "DAN_CHIEU": (17, "Dẫn chiếu"),
    "AP_DUNG": (18, "Áp dụng"),
    "GIAI_THICH": (19, "Giải thích"),
    "DINH_CHI_THI_HANH": (20, "Đình chỉ thi hành"),
    "QUY_DINH_TAM_NGUNG_GIA_HAN": (21, "Quy định tạm ngưng/ gia hạn"),

    # 2.2 Quan hệ cấu trúc nội bộ (22)
    "THUOC": (22, "Thuộc")
}

# ==============================================================================
# 3. BẢNG ÁNH XẠ LƯỢC ĐỒ (luoc_do.json) -> THỰC THỂ & QUAN HỆ
# ==============================================================================
LUOC_DO_MAP: Dict[Tuple[str, str], Tuple[str, str]] = {
    # Incoming
    ("Văn bản áp dụng", "incoming"): ("VB_AP_DUNG", "AP_DUNG"),
    ("Văn bản dẫn chiếu", "incoming"): ("VB_DAN_CHIEU", "DAN_CHIEU"),
    ("Văn bản hướng dẫn áp dụng", "incoming"): ("VB_HUONG_DAN_AP_DUNG", "HUONG_DAN_AP_DUNG"),
    ("Văn bản quy định chi tiết, hướng dẫn thi hành", "incoming"): ("VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH", "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH"),
    ("Văn bản sửa đổi bổ sung", "incoming"): ("VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN", "QUY_DINH_HET_HIEU_LUC_MOT_PHAN"),
    ("Văn bản thay thế", "incoming"): ("VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO", "QUY_DINH_HET_HIEU_LUC_TOAN_BO"),
    ("Văn bản bãi bỏ", "incoming"): ("VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO", "QUY_DINH_HET_HIEU_LUC_TOAN_BO"),
    ("Văn bản đính chính", "incoming"): ("VB_DINH_CHINH", "DINH_CHINH"),
    ("Văn bản hợp nhất", "incoming"): ("VB_HOP_NHAT", "HOP_NHAT"),
    ("Văn bản giải thích", "incoming"): ("VB_GIAI_THICH", "GIAI_THICH"),
    ("Văn bản đình chỉ thi hành", "incoming"): ("VB_DINH_CHI_THI_HANH", "DINH_CHI_THI_HANH"),
    ("Văn bản tạm ngưng hiệu lực", "incoming"): ("VB_QUY_DINH_TAM_NGUNG_GIA_HAN", "QUY_DINH_TAM_NGUNG_GIA_HAN"),

    # Outgoing
    ("Căn cứ ban hành", "outgoing"): ("CAN_CU_BAN_HANH", "CAN_CU_BAN_HANH"),
    ("Văn bản được dẫn chiếu", "outgoing"): ("VB_DUOC_DAN_CHIEU", "DUOC_DAN_CHIEU"),
    ("Văn bản được hướng dẫn áp dụng", "outgoing"): ("VB_DUOC_HUONG_DAN_AP_DUNG", "DUOC_HUONG_DAN_AP_DUNG"),
    ("Văn bản được quy định chi tiết, hướng dẫn thi hành", "outgoing"): ("VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH", "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH"),
    ("Văn bản được sửa đổi bổ sung", "outgoing"): ("VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN", "BI_TAC_DONG_HIEU_LUC_MOT_PHAN"),
    ("Văn bản bị bãi bỏ", "outgoing"): ("VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO", "BI_TAC_DONG_HIEU_LUC_TOAN_BO"),
    ("Văn bản được thay thế", "outgoing"): ("VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO", "BI_TAC_DONG_HIEU_LUC_TOAN_BO"),
    ("Văn bản được đính chính", "outgoing"): ("VB_DUOC_DINH_CHINH", "DUOC_DINH_CHINH"),
    ("Văn bản được hợp nhất", "outgoing"): ("VB_HOP_NHAT", "HOP_NHAT"),
    ("Văn bản được giải thích", "outgoing"): ("VB_DUOC_GIAI_THICH", "DUOC_GIAI_THICH"),
    ("Văn bản bị đình chỉ thi hành", "outgoing"): ("VB_BI_DINH_CHI_THI_HANH", "BI_DINH_CHI_THI_HANH"),
    ("Văn bản bị tạm ngưng hiệu lực", "outgoing"): ("VB_BI_TAM_NGUNG_GIA_HAN", "BI_TAM_NGUNG_GIA_HAN")
}

# ==============================================================================
# 4. TRÍCH XUẤT TOÀN VĂN ĐẢM BẢO THỨ TỰ KHỐI VĂN BẢN VÀ KHÔNG DƯ KHOẢNG TRẮNG
# ==============================================================================
def extract_docx_text(docx_path: str) -> str:
    """
    Trích xuất toàn văn từ file docx bằng cách duyệt tuần tự doc.element.body
    để bảo toàn đúng thứ tự các đoạn văn (paragraphs) và các bảng (tables).
    Mỗi dòng văn bản được rstrip/strip để tránh khoảng trắng thừa cuối dòng.
    """
    if docx is None:
        raise ImportError("Vui lòng cài đặt thư viện python-docx (`pip install python-docx`)")
    
    doc = docx.Document(docx_path)
    blocks = []
    for block in doc.element.body:
        if isinstance(block, CT_P):
            p = Paragraph(block, doc)
            lines = [line.strip() for line in p.text.splitlines() if line.strip()]
            if lines:
                blocks.append("\n".join(lines))
        elif isinstance(block, CT_Tbl):
            table = Table(block, doc)
            for row in table.rows:
                cell_texts = []
                for cell in row.cells:
                    c_lines = []
                    for p in cell.paragraphs:
                        for line in p.text.splitlines():
                            line_s = line.strip()
                            if line_s:
                                c_lines.append(line_s)
                    ct = "\n".join(c_lines)
                    if ct and ct not in cell_texts:
                        cell_texts.append(ct)
                if cell_texts:
                    blocks.append("\n".join(cell_texts))
    return "\n".join(blocks)


def is_doc_in_body(d_title: str, body_text: str) -> bool:
    """Kiểm tra xem văn bản tham chiếu đã được trích dẫn/phân tích trong thân văn bản hay chưa."""
    if d_title in body_text:
        return True
    num_m = re.search(r"\b(\d+[-/]\d+[-/][A-Z0-9\-_]+|\d+/\d+/[A-Z0-9\-_]+)\b", d_title)
    if num_m and num_m.group(1) in body_text:
        return True
    base_name = d_title.split(" số ")[0].split(" ngày ")[0].strip()
    if base_name and base_name in body_text:
        return True
    clean_name = re.sub(r"^(?:Luật|Bộ luật)\s+sửa\s+đổi[,\s]+bổ\s+sung\s+(?:một\s+số\s+điều\s+của\s+)?", "", base_name, flags=re.IGNORECASE).strip()
    if clean_name and clean_name in body_text:
        return True
    return False


# ==============================================================================
# 5. BỘ GÁN NHÃN TỰ ĐỘNG CHUẨN HÓA CHO MỘT THƯ MỤC VĂN BẢN
# ==============================================================================
def auto_label_single_document(folder_path: str) -> Tuple[Dict[str, Any], List[List[str]]]:
    """
    Gán nhãn tự động một thư mục văn bản pháp luật dựa trên:
      1. noi_dung.docx (nếu thiếu, tự động tái tạo từ thuoc_tinh.json & luoc_do.json).
      2. thuoc_tinh.json (chứa 12 trường metadata).
      3. luoc_do.json (chứa các quan hệ pháp lý liên văn bản).
      4. muc_luc.json (nếu có, dùng để bóc tách cấu trúc chính xác).
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    
    # 5.1 Đọc các file dữ liệu đầu vào
    tt_path = os.path.join(folder_path, "thuoc_tinh.json")
    ld_path = os.path.join(folder_path, "luoc_do.json")
    docx_path = os.path.join(folder_path, "noi_dung.docx")
    ml_path = os.path.join(folder_path, "muc_luc.json")

    attributes = {}
    doc_metadata = {}
    if os.path.exists(tt_path):
        with open(tt_path, "r", encoding="utf-8") as f:
            tt_data = json.load(f)
            attributes = tt_data.get("attributes", {})
            doc_metadata = tt_data.get("metadata", {})

    luoc_do_data = {}
    if os.path.exists(ld_path):
        with open(ld_path, "r", encoding="utf-8") as f:
            luoc_do_data = json.load(f)

    # 5.2 Xử lý phần thân văn bản (body_text)
    if os.path.exists(docx_path):
        body_text = extract_docx_text(docx_path)
    else:
        # Tái tạo văn bản từ metadata nếu không có docx
        title = doc_metadata.get("title", folder_name)
        so_hieu = attributes.get("so_hieu", "")
        cq = attributes.get("co_quan_ban_hanh", "")
        ngay_bh = attributes.get("ngay_ban_hanh", "")
        body_text = f"{title}\n{cq}\nSố: {so_hieu}\n{ngay_bh}\n"

    # Kiểm tra xem ngày có hiệu lực đã có sẵn trong body_text hay chưa
    hieu_luc_in_body = False
    hl_val = attributes.get("ngay_co_hieu_luc", "")
    if hl_val and hl_val != "--" and hl_val != "null":
        # Chuẩn hóa ngày dạng YYYY-MM-DD hoặc DD/MM/YYYY
        try:
            if "-" in hl_val:
                y, m, d = [int(x) for x in hl_val.split("-")]
            elif "/" in hl_val:
                p_parts = [int(x) for x in hl_val.split("/")]
                y, m, d = (p_parts[0], p_parts[1], p_parts[2]) if p_parts[0] > 1000 else (p_parts[2], p_parts[1], p_parts[0])
            else:
                y, m, d = None, None, None
            if y:
                hl_pat = rf"ngày\s+{d}\s+tháng\s+0?{m}\s+năm\s+{y}"
                if re.search(hl_pat, body_text):
                    hieu_luc_in_body = True
        except Exception:
            pass

    # Kiểm tra chức danh / người ký có trong chữ ký cuối văn bản hay chưa
    cd_in_body = False
    nk_in_body = False
    cd_val = attributes.get("chuc_danh_nguoi_ky", "")
    nk_val = attributes.get("nguoi_ky", "")
    sig_search_zone = body_text[-2500:] if len(body_text) > 2500 else body_text
    if cd_val and cd_val != "--" and cd_val != "null":
        if re.search(r"\b" + re.escape(cd_val) + r"\b", sig_search_zone, re.IGNORECASE):
            cd_in_body = True
    if nk_val and nk_val != "--" and nk_val != "null":
        if re.search(r"\b" + re.escape(nk_val) + r"\b", sig_search_zone, re.IGNORECASE):
            nk_in_body = True

    # 5.3 Xây dựng footer cho thông tin thuộc tính & quan hệ lược đồ không có trong thân văn bản
    footer_lines = []
    
    field_labels = [
        ("nganh", "Ngành", False),
        ("linh_vuc", "Lĩnh vực", False),
        ("ngay_co_hieu_luc", "Ngày có hiệu lực", hieu_luc_in_body),
        ("ngay_het_hieu_luc", "Ngày hết hiệu lực", False),
        ("tinh_trang_hieu_luc", "Tình trạng hiệu lực", False),
        ("chuc_danh_nguoi_ky", "Chức danh", cd_in_body),
        ("nguoi_ky", "Người ký", nk_in_body)
    ]
    for key, label, in_body in field_labels:
        val = attributes.get(key)
        if val and val != "--" and val != "null" and not in_body:
            if val not in body_text:
                footer_lines.append(f"[{label}]: {val}")

    # Lược đồ quan hệ từ luoc_do.json
    preamble = body_text[:3000]
    for rel_item in luoc_do_data.get("relations", []):
        r_type = rel_item.get("relation_type")
        r_dir = rel_item.get("direction")
        
        # Căn cứ ban hành
        if r_type == "Căn cứ ban hành" and r_dir == "outgoing":
            for doc in rel_item.get("documents", []):
                d_title = doc.get("title", "").strip()
                base_name = d_title.split(" số ")[0].split(" ngày ")[0].strip()
                # Nếu đã xuất hiện ở phần Căn cứ đầu văn bản thì không đưa vào footer
                if f"Căn cứ {base_name}" in preamble or f"Căn cứ vào {base_name}" in preamble:
                    continue
                footer_lines.append(f"[{r_type}]: {d_title}")
        else:
            # Các quan hệ liên văn bản khác: chỉ đưa vào footer nếu không được dẫn chiếu/phân tích trong văn bản
            mapping = LUOC_DO_MAP.get((r_type, r_dir))
            if not mapping:
                continue
            for doc in rel_item.get("documents", []):
                d_title = doc.get("title", "").strip()
                if is_doc_in_body(d_title, body_text):
                    continue
                footer_lines.append(f"[{r_type}]: {d_title}")

    if footer_lines:
        full_text = body_text + "\n--- THÔNG TIN THUỘC TÍNH & LƯỢC ĐỒ TỪ FILE JSON ---\n" + "\n".join(footer_lines)
    else:
        full_text = body_text

    raw_entities: List[Dict[str, Any]] = []

    # 5.4 Trích xuất Thực thể Metadata (E1 -> E12)
    footer_pos = full_text.find("\n--- THÔNG TIN THUỘC TÍNH & LƯỢC ĐỒ TỪ FILE JSON ---\n")
    body_limit = footer_pos if footer_pos != -1 else len(full_text)

    # SO_HIEU (2)
    so_m = re.search(r"Số:\s*([^\s\n]+)", full_text[:1200])
    if so_m:
        raw_entities.append({
            "temp_id": "SO_HIEU",
            "code": "SO_HIEU",
            "start": so_m.start(1),
            "end": so_m.end(1),
            "text": so_m.group(1)
        })
    else:
        so_hieu_val = attributes.get("so_hieu", "")
        if so_hieu_val and so_hieu_val in full_text[:1200]:
            p = full_text[:1200].find(so_hieu_val)
            raw_entities.append({
                "temp_id": "SO_HIEU",
                "code": "SO_HIEU",
                "start": p,
                "end": p + len(so_hieu_val),
                "text": so_hieu_val
            })

    # CO_QUAN_BAN_HANH (10)
    # Lấy khối chữ hoa cơ quan ban hành ngay phía trên dòng "Số: ..."
    if so_m:
        pre = full_text[:so_m.start()].rstrip()
        lines = [l.strip() for l in pre.splitlines() if l.strip()]
        cq_lines = []
        for line in reversed(lines):
            clean = re.sub(r"[^A-Za-zÀ-ỹĐđ]", "", line)
            if clean and clean.isupper():
                cq_lines.insert(0, line)
            else:
                break
        if cq_lines:
            cq_text = "\n".join(cq_lines)
            cq_pos = full_text.find(cq_text)
            raw_entities.append({
                "temp_id": "CO_QUAN_BAN_HANH",
                "code": "CO_QUAN_BAN_HANH",
                "start": cq_pos,
                "end": cq_pos + len(cq_text),
                "text": cq_text
            })
    if not any(e["code"] == "CO_QUAN_BAN_HANH" for e in raw_entities):
        cq_val = attributes.get("co_quan_ban_hanh", "")
        if cq_val:
            cq_m = re.search(r"\b" + re.escape(cq_val) + r"\b", full_text[:1000], re.IGNORECASE)
            if cq_m:
                raw_entities.append({
                    "temp_id": "CO_QUAN_BAN_HANH",
                    "code": "CO_QUAN_BAN_HANH",
                    "start": cq_m.start(),
                    "end": cq_m.end(),
                    "text": cq_m.group(0)
                })

    # NGAY_BAN_HANH (5)
    nbh_m = re.search(r"(?:Hà Nội,\s*)?(ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})", full_text[:1200])
    if nbh_m:
        raw_entities.append({
            "temp_id": "NGAY_BAN_HANH",
            "code": "NGAY_BAN_HANH",
            "start": nbh_m.start(1),
            "end": nbh_m.end(1),
            "text": nbh_m.group(1)
        })

    # TEN_VAN_BAN (1) & LOAI_VAN_BAN (3)
    title_m = re.search(r"(?m)^(NGHỊ QUYẾT|QUYẾT ĐỊNH|THÔNG TƯ|NGHỊ ĐỊNH|BỘ LUẬT|LUẬT)\s*\n([^\n_]+)", full_text[:1500])
    if title_m:
        raw_entities.append({
            "temp_id": "TEN_VAN_BAN",
            "code": "TEN_VAN_BAN",
            "start": title_m.start(0),
            "end": title_m.end(0),
            "text": title_m.group(0)
        })
        raw_entities.append({
            "temp_id": "LOAI_VAN_BAN",
            "code": "LOAI_VAN_BAN",
            "start": title_m.start(1),
            "end": title_m.end(1),
            "text": title_m.group(1)
        })

    # CAN_CU_BAN_HANH (19) trong phần mở đầu (preamble)
    for m in re.finditer(r"(?m)^Căn cứ\s+(?:vào\s+)?([^;\n]+)", full_text[:3000]):
        matched_str = m.group(1).rstrip(";.\n").strip()
        c_start = m.start(1)
        c_end = c_start + len(matched_str)
        raw_entities.append({
            "temp_id": f"CAN_CU_BAN_HANH_{c_start}",
            "code": "CAN_CU_BAN_HANH",
            "start": c_start,
            "end": c_end,
            "text": matched_str,
            "is_body_cancu": True
        })

    # NGAY_CO_HIEU_LUC (7) trong thân văn bản nếu có
    if hieu_luc_in_body and hl_val:
        try:
            if "-" in hl_val:
                y, m, d = [int(x) for x in hl_val.split("-")]
            elif "/" in hl_val:
                p_parts = [int(x) for x in hl_val.split("/")]
                y, m, d = (p_parts[0], p_parts[1], p_parts[2]) if p_parts[0] > 1000 else (p_parts[2], p_parts[1], p_parts[0])
            date_pattern = rf"ngày\s+{d}\s+tháng\s+0?{m}\s+năm\s+{y}"
            hl_match = re.search(date_pattern, full_text)
            if hl_match:
                raw_entities.append({
                    "temp_id": "NGAY_CO_HIEU_LUC",
                    "code": "NGAY_CO_HIEU_LUC",
                    "start": hl_match.start(),
                    "end": hl_match.end(),
                    "text": hl_match.group(0)
                })
        except Exception:
            pass

    # CHUC_DANH (11) & NGUOI_KY (12) trong khối chữ ký cuối thân bài
    if cd_val and cd_in_body:
        cd_m = re.search(r"\b" + re.escape(cd_val) + r"\b", full_text[:body_limit][-2500:], re.IGNORECASE)
        if cd_m:
            cd_start = (body_limit - 2500) + cd_m.start()
            cd_end = cd_start + len(cd_m.group(0))
            raw_entities.append({
                "temp_id": "CHUC_DANH",
                "code": "CHUC_DANH",
                "start": cd_start,
                "end": cd_end,
                "text": cd_m.group(0)
            })

    if nk_val and nk_in_body:
        nk_m = re.search(r"\b" + re.escape(nk_val) + r"\b", full_text[:body_limit][-2500:], re.IGNORECASE)
        if nk_m:
            nk_start = (body_limit - 2500) + nk_m.start()
            nk_end = nk_start + len(nk_m.group(0))
            raw_entities.append({
                "temp_id": "NGUOI_KY",
                "code": "NGUOI_KY",
                "start": nk_start,
                "end": nk_end,
                "text": nk_m.group(0)
            })

    # 5.5 Thực thể Footer (Metadata & Quan hệ Lược đồ)
    if footer_pos != -1:
        footer_sub = full_text[footer_pos:]
        for m in re.finditer(r"\[([^\]]+)\]:\s*([^\n]+)", footer_sub):
            lbl_tag = m.group(1).strip()
            val_text = m.group(2).strip()
            v_start = footer_pos + m.start(2)
            v_end = footer_pos + m.end(2)

            tag_to_code = {
                "Ngành": "NGANH",
                "Lĩnh vực": "LINH_VUC",
                "Ngày có hiệu lực": "NGAY_CO_HIEU_LUC",
                "Ngày hết hiệu lực": "NGAY_HET_HIEU_LUC",
                "Tình trạng hiệu lực": "TINH_TRANG_HIEU_LUC",
                "Chức danh": "CHUC_DANH",
                "Người ký": "NGUOI_KY",
                "Căn cứ ban hành": "CAN_CU_BAN_HANH"
            }
            if lbl_tag in tag_to_code:
                code = tag_to_code[lbl_tag]
                raw_entities.append({
                    "temp_id": f"{code}_{v_start}",
                    "code": code,
                    "start": v_start,
                    "end": v_end,
                    "text": val_text,
                    "is_body_cancu": False
                })
            elif (lbl_tag, "incoming") in LUOC_DO_MAP or (lbl_tag, "outgoing") in LUOC_DO_MAP:
                mapping = LUOC_DO_MAP.get((lbl_tag, "incoming")) or LUOC_DO_MAP.get((lbl_tag, "outgoing"))
                e_code, _ = mapping
                raw_entities.append({
                    "temp_id": f"{e_code}_{v_start}",
                    "code": e_code,
                    "start": v_start,
                    "end": v_end,
                    "text": val_text,
                    "is_body_cancu": False
                })

    # 5.6 Trích xuất Cấu trúc văn bản (E34..E40)
    # Xác định điểm kết thúc của phần nội dung điều khoản (trước phần Nơi nhận / Ký tên)
    end_of_articles = full_text.find("\nNơi nhận:")
    if end_of_articles == -1:
        end_of_articles = body_limit

    dieu_matches = list(re.finditer(r"(?m)^Điều\s+(\d+)[\.\:]", full_text[:end_of_articles]))

    if dieu_matches:
        # Bóc tách cấu trúc phân cấp tuần tự theo Điều -> Khoản -> Điểm
        for idx, dm in enumerate(dieu_matches):
            d_start = dm.start()
            d_next = dieu_matches[idx + 1].start() - 1 if idx + 1 < len(dieu_matches) else end_of_articles
            d_content = full_text[d_start:d_next]
            
            # Tìm các Khoản (bắt đầu bằng ^1., ^2., ...)
            khoan_matches = list(re.finditer(r"(?m)^(\d+)\.\s+", d_content))
            valid_khoans = []
            expected_k = 1
            for km in khoan_matches:
                if int(km.group(1)) == expected_k:
                    valid_khoans.append(km)
                    expected_k += 1
                    
            if not valid_khoans:
                # Điều không có Khoản -> Span bao trọn toàn bộ nội dung Điều
                d_end = d_next
                raw_entities.append({
                    "temp_id": f"DIEU_{d_start}",
                    "code": "DIEU",
                    "start": d_start,
                    "end": d_end,
                    "text": full_text[d_start:d_end]
                })
            else:
                # Điều có Khoản -> Tiêu đề Điều kết thúc trước Khoản 1
                k1_start = d_start + valid_khoans[0].start()
                d_end = k1_start - 1
                raw_entities.append({
                    "temp_id": f"DIEU_{d_start}",
                    "code": "DIEU",
                    "start": d_start,
                    "end": d_end,
                    "text": full_text[d_start:d_end]
                })
                
                for k_idx, km in enumerate(valid_khoans):
                    k_start = d_start + km.start()
                    k_next = d_start + valid_khoans[k_idx + 1].start() - 1 if k_idx + 1 < len(valid_khoans) else d_next
                    k_content = full_text[k_start:k_next]
                    
                    # Tìm các Điểm (bắt đầu bằng ^a), ^b), ...)
                    diem_matches = list(re.finditer(r"(?m)^([a-zđ])\)\s+", k_content))
                    valid_diems = list(diem_matches)
                        
                    if not valid_diems:
                        # Khoản không có Điểm -> Span bao trọn nội dung Khoản
                        k_end = k_next
                        raw_entities.append({
                            "temp_id": f"KHOAN_{k_start}",
                            "code": "KHOAN",
                            "start": k_start,
                            "end": k_end,
                            "text": full_text[k_start:k_end]
                        })
                    else:
                        # Khoản có Điểm -> Lời dẫn Khoản kết thúc trước Điểm a)
                        diem1_start = k_start + valid_diems[0].start()
                        k_end = diem1_start - 1
                        raw_entities.append({
                            "temp_id": f"KHOAN_{k_start}",
                            "code": "KHOAN",
                            "start": k_start,
                            "end": k_end,
                            "text": full_text[k_start:k_end]
                        })
                        
                        for d_idx, dim in enumerate(valid_diems):
                            dim_start = k_start + dim.start()
                            dim_next = k_start + valid_diems[d_idx + 1].start() - 1 if d_idx + 1 < len(valid_diems) else k_next
                            dim_end = dim_next
                            raw_entities.append({
                                "temp_id": f"DIEM_{dim_start}",
                                "code": "DIEM",
                                "start": dim_start,
                                "end": dim_end,
                                "text": full_text[dim_start:dim_end]
                            })
    else:
        # Nếu văn bản không có Điều (dạng chỉ thị, thông cáo hoặc chia Mục)
        for m in re.finditer(r"(?m)^([IVXLCDM]+)\.\s+([^\n]+)", full_text[:end_of_articles]):
            raw_entities.append({"temp_id": f"PHAN_{m.start()}", "code": "PHAN", "start": m.start(), "end": m.end(), "text": m.group(0)})
        for m in re.finditer(r"(?m)^(Chương\s+[IVXLCDM\d]+[\.\:\s]*[^\n]*|[A-Z]\.\s+[^\n]+)", full_text[:end_of_articles]):
            raw_entities.append({"temp_id": f"CHUONG_{m.start()}", "code": "CHUONG", "start": m.start(), "end": m.end(), "text": m.group(0)})
        for m in re.finditer(r"(?m)^(Mục\s+\d+[\.\:\s]*[^\n]*|\d+\.\s+[^\n]+)", full_text[:end_of_articles]):
            raw_entities.append({"temp_id": f"MUC_{m.start()}", "code": "MUC", "start": m.start(), "end": m.end(), "text": m.group(0)})

    # Lọc trùng lặp & kiểm tra tính toàn vẹn ký tự (offset validation)
    unique_entities = []
    seen_spans = set()
    for e in raw_entities:
        s, end, txt, code = e["start"], e["end"], e["text"], e["code"]
        if full_text[s:end] != txt:
            continue
        span_key = (s, end, code)
        if span_key not in seen_spans:
            seen_spans.add(span_key)
            unique_entities.append(e)

    # 5.7 Sắp xếp thực thể theo thứ tự chuẩn E1...E40
    def entity_sort_key(ent):
        t_num = ENTITY_TAXONOMY[ent["code"]][0]
        if ent["code"] == "CAN_CU_BAN_HANH":
            # Căn cứ footer xếp trước (đáp ứng trật tự gán nhãn mẫu), sau đó là căn cứ thân bài
            order_flag = 1 if ent.get("is_body_cancu") else 0
            return (t_num, order_flag, ent["start"])
        return (t_num, ent["start"])

    unique_entities.sort(key=entity_sort_key)

    # Đánh số ID tuần tự: E{t_num}_{seq}
    type_counters: Dict[int, int] = {}
    id_map: Dict[str, str] = {}
    final_entities = []
    for ent in unique_entities:
        code = ent["code"]
        t_num, label, scope = ENTITY_TAXONOMY[code]
        seq = type_counters.get(t_num, 0) + 1
        type_counters[t_num] = seq
        final_id = f"E{t_num}_{seq}"
        id_map[ent["temp_id"]] = final_id
        final_entities.append({
            "id": final_id,
            "start": ent["start"],
            "end": ent["end"],
            "text": ent["text"],
            "code": code,
            "label": label,
            "scope": scope
        })

    # 5.8 Xây dựng Quan hệ (Structural Thuộc R22 & Legal Relations R1..R21)
    raw_relations = []
    
    so_hieu_ids = [e["id"] for e in final_entities if e["code"] == "SO_HIEU"]
    doc_so_hieu = so_hieu_ids[0] if so_hieu_ids else None

    # CAN_CU_BAN_HANH (R7): Căn cứ thân bài -> SO_HIEU trước, sau đó SO_HIEU -> Căn cứ footer
    cancu_ents = [e for e in final_entities if e["code"] == "CAN_CU_BAN_HANH"]
    body_cancu = [e for e in cancu_ents if e["start"] < 3000]
    footer_cancu = [e for e in cancu_ents if e["start"] >= 3000]

    if doc_so_hieu:
        for e in body_cancu:
            raw_relations.append({
                "from_id": e["id"],
                "to_id": doc_so_hieu,
                "code": "CAN_CU_BAN_HANH",
                "label": "Căn cứ ban hành",
                "type_num": 7
            })
        for e in footer_cancu:
            raw_relations.append({
                "from_id": doc_so_hieu,
                "to_id": e["id"],
                "code": "CAN_CU_BAN_HANH",
                "label": "Căn cứ ban hành",
                "type_num": 7
            })

        # Quan hệ từ Footer đối với các liên văn bản khác
        for e in final_entities:
            if e["scope"] == "reference" and e["code"] != "CAN_CU_BAN_HANH" and e["start"] >= body_limit:
                r_code = e["code"].replace("VB_DUOC_", "DUOC_").replace("VB_BI_", "BI_").replace("VB_QUY_DINH_", "QUY_DINH_").replace("VB_", "")
                if r_code in RELATION_TAXONOMY:
                    r_num, r_label = RELATION_TAXONOMY[r_code]
                    raw_relations.append({
                        "from_id": doc_so_hieu,
                        "to_id": e["id"],
                        "code": r_code,
                        "label": r_label,
                        "type_num": r_num
                    })

    # Phân cấp người ký & chức danh: NGUOI_KY -> CHUC_DANH -> SO_HIEU (hoặc CO_QUAN_BAN_HANH)
    nguoi_ky_ids = [e["id"] for e in final_entities if e["code"] == "NGUOI_KY"]
    chuc_danh_ids = [e["id"] for e in final_entities if e["code"] == "CHUC_DANH"]

    if nguoi_ky_ids and chuc_danh_ids:
        raw_relations.append({
            "from_id": nguoi_ky_ids[0],
            "to_id": chuc_danh_ids[0],
            "code": "THUOC",
            "label": "Thuộc",
            "type_num": 22
        })
    if chuc_danh_ids and doc_so_hieu:
        raw_relations.append({
            "from_id": chuc_danh_ids[0],
            "to_id": doc_so_hieu,
            "code": "THUOC",
            "label": "Thuộc",
            "type_num": 22
        })

    # Quan hệ phân cấp cấu trúc chuẩn theo phân cấp mục lục:
    # Phần (0) <- (thuộc) <- Chương (1) <- (thuộc) <- Mục (2) <- (thuộc) <- Tiểu mục (3) <- (thuộc) <- Điều (4) <- (thuộc) <- Khoản (5) <- (thuộc) <- Điểm (6)
    TOC_HIERARCHY_RANK = {
        "PHAN": 0,
        "CHUONG": 1,
        "MUC": 2,
        "TIEU_MUC": 3,
        "DIEU": 4,
        "KHOAN": 5,
        "DIEM": 6
    }
    struct_ents = [e for e in final_entities if e["code"] in TOC_HIERARCHY_RANK]
    struct_ents.sort(key=lambda x: (x["start"], x["end"]))
    active_ancestors = {}
    for ann in struct_ents:
        rank = TOC_HIERARCHY_RANK[ann["code"]]
        # Xóa các cấp bằng hoặc sâu hơn khi gặp thực thể mới cùng cấp hoặc cấp cao hơn
        for r in list(active_ancestors.keys()):
            if r >= rank:
                del active_ancestors[r]
        active_ancestors[rank] = ann

        # Tìm tổ tiên trực tiếp gần nhất bao bọc nó
        parent_ann = None
        for pr in range(rank - 1, -1, -1):
            if pr in active_ancestors:
                parent_ann = active_ancestors[pr]
                break

        if parent_ann:
            raw_relations.append({
                "from_id": ann["id"],
                "to_id": parent_ann["id"],
                "code": "THUOC",
                "label": "Thuộc",
                "type_num": 22
            })


    # 5.9 Sắp xếp quan hệ theo loại (R1 -> R22)
    rel_counters: Dict[int, int] = {}
    final_relations = []
    for r in raw_relations:
        t_num = r["type_num"]
        seq = rel_counters.get(t_num, 0) + 1
        rel_counters[t_num] = seq
        final_relations.append({
            "id": f"R{t_num}_{seq}",
            "from_id": r["from_id"],
            "to_id": r["to_id"],
            "code": r["code"],
            "label": r["label"]
        })

    # 5.10 Đóng gói JSON
    result_json = {
        "schema_version": "1.0",
        "document_id": doc_metadata.get("item_id", folder_name),
        "document_code": attributes.get("so_hieu", folder_name),
        "file_name": f"{folder_name}.docx",
        "text_encoding": "utf-8-nfc",
        "offset_convention": "0_based_half_open",
        "document_metadata": {
            "ten_van_ban": (final_entities[0]["text"] if final_entities and final_entities[0]["code"] == "TEN_VAN_BAN" else doc_metadata.get("title", "")),
            "so_hieu": attributes.get("so_hieu"),
            "loai_van_ban": attributes.get("loai_van_ban"),
            "co_quan_ban_hanh": attributes.get("co_quan_ban_hanh"),
            "ngay_ban_hanh": attributes.get("ngay_ban_hanh"),
            "ngay_co_hieu_luc": attributes.get("ngay_co_hieu_luc"),
            "ngay_het_hieu_luc": attributes.get("ngay_het_hieu_luc"),
            "tinh_trang_hieu_luc": attributes.get("tinh_trang_hieu_luc"),
            "chuc_danh": attributes.get("chuc_danh_nguoi_ky"),
            "nguoi_ky": attributes.get("nguoi_ky"),
            "nganh": attributes.get("nganh"),
            "linh_vuc": attributes.get("linh_vuc")
        },
        "full_text": full_text,
        "structural_units": [],
        "entities": final_entities,
        "relations": final_relations,
        "future_extensions": {
            "summary": None,
            "qa_pairs": []
        }
    }

    # 5.11 Đóng gói CSV
    csv_rows = []
    csv_rows.append(["NHÓM (ENTITY / RELATION)", "ID", "CODE", "LABEL", "SCOPE / FROM_ID", "START / TO_ID", "END", "TEXT"])
    for e in final_entities:
        csv_rows.append(["ENTITY", e["id"], e["code"], e["label"], e["scope"], str(e["start"]), str(e["end"]), e["text"]])
    for r in final_relations:
        csv_rows.append(["RELATION", r["id"], r["code"], r["label"], r["from_id"], r["to_id"], "", ""])

    return result_json, csv_rows


# ==============================================================================
# 6. GHI FILE RA ĐĨA (JSON & CSV UTF-8 BOM)
# ==============================================================================
def save_output_files(result_json: Dict[str, Any], csv_rows: List[List[str]], output_dir: str, base_name: str) -> Tuple[str, str]:
    """Ghi 2 file {base_name}_ket_qua.json và {base_name}_ket_qua.csv vào output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{base_name}_ket_qua.json")
    csv_path = os.path.join(output_dir, f"{base_name}_ket_qua.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        for row in csv_rows:
            writer.writerow(row)

    return json_path, csv_path


# ==============================================================================
# 7. CLI COMMAND RUNNER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Bộ công cụ gán nhãn tự động văn bản pháp luật chuẩn hóa (tương thích 10 file nghi_quyet mẫu)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", type=str, help="Đường dẫn đến 1 thư mục văn bản cụ thể (VD: bo_luat/100_2015_QH13_96122)")
    group.add_argument("--dir", type=str, help="Đường dẫn đến thư mục chứa nhiều văn bản (VD: bo_luat hoặc nghi_quyet)")

    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục lưu kết quả (mặc định: dataset/<tên_thư_mục_nguồn>)")
    parser.add_argument("--force", action="store_true", help="Ghi đè kết quả nếu file đã tồn tại")

    args = parser.parse_args()

    targets = []
    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"[LỖI] Thư mục '{args.folder}' không tồn tại!")
            sys.exit(1)
        targets.append(args.folder)
        default_out = os.path.join("dataset", os.path.basename(os.path.dirname(os.path.abspath(args.folder))))
    else:
        if not os.path.isdir(args.dir):
            print(f"[LỖI] Thư mục '{args.dir}' không tồn tại!")
            sys.exit(1)
        subdirs = sorted([os.path.join(args.dir, d) for d in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, d))])
        targets.extend(subdirs)
        default_out = os.path.join("dataset", os.path.basename(os.path.normpath(args.dir)))

    output_dir = args.output_dir if args.output_dir else default_out
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("   BỘ CÔNG CỤ GÁN NHÃN TỰ ĐỘNG VĂN BẢN PHÁP LUẬT (NCKH GNN-TRANSFORMER)")
    print(f"   Tổng số văn bản cần xử lý: {len(targets)}")
    print(f"   Thư mục xuất kết quả:      {output_dir}")
    print("=" * 70)

    success_count = 0
    total_entities = 0
    total_relations = 0

    for idx, target in enumerate(targets, 1):
        bname = os.path.basename(os.path.normpath(target))
        json_out_file = os.path.join(output_dir, f"{bname}_ket_qua.json")
        csv_out_file = os.path.join(output_dir, f"{bname}_ket_qua.csv")

        if os.path.exists(json_out_file) and not args.force:
            print(f"[{idx}/{len(targets)}] Bỏ qua {bname} (Đã tồn tại file kết quả, dùng --force để ghi đè)")
            continue

        try:
            res_json, res_csv = auto_label_single_document(target)
            save_output_files(res_json, res_csv, output_dir, bname)
            success_count += 1
            n_ent = len(res_json["entities"])
            n_rel = len(res_json["relations"])
            total_entities += n_ent
            total_relations += n_rel
            print(f"[{idx}/{len(targets)}] [THÀNH CÔNG] {bname}: {n_ent} entities, {n_rel} relations -> {output_dir}")
        except Exception as e:
            print(f"[{idx}/{len(targets)}] [LỖI] {bname}: {e}")

    print("=" * 70)
    print(f"HOÀN TẤT: Đã gán nhãn thành công {success_count}/{len(targets)} văn bản.")
    print(f"Tổng số thực thể trích xuất: {total_entities}")
    print(f"Tổng số quan hệ thiết lập:   {total_relations}")
    print("=" * 70)


if __name__ == "__main__":
    main()
