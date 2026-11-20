# -*- coding: utf-8 -*-
"""
labels.py
Định nghĩa hệ thống nhãn (Taxonomy & Label Schema) chuẩn hóa cho đề tài NCKH:
"Ứng dụng GNN kết hợp Transformer trong trích xuất quan hệ thực thể cấp độ tài liệu".

Đồng bộ 100% với:
  1. File cấu hình Label Studio: docs/label.xml
  2. Pipeline gán nhãn tự động: auto_label_pipeline.py
  3. 174 văn bản đã gán nhãn thực tế trong thư mục dataset/ (JSON & CSV)

Hệ thống bao gồm:
  - 40 Nhãn Thực thể (Entity Labels):
      + 12 trường metadata (TEN_VAN_BAN, SO_HIEU, CO_QUAN_BAN_HANH, CHUC_DANH, NGUOI_KY,...)
      + 7 cấp cấu trúc văn bản (PHAN, CHUONG, MUC, TIEU_MUC, DIEU, KHOAN, DIEM)
      + 21 loại văn bản viện dẫn / pháp lý liên quan (VB_HUONG_DAN_AP_DUNG, CAN_CU_BAN_HANH,...)
  - 22 Nhãn Quan hệ (Relation Labels):
      + 1 quan hệ cấu trúc & hành chính: THUOC (Thuộc)
      + 21 quan hệ pháp lý liên văn bản (CAN_CU_BAN_HANH, HUONG_DAN_AP_DUNG, QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH,...)
  - Hệ thống thuần túy 40 Thực thể & 22 Quan hệ chuẩn hóa (không sử dụng 10 cấp bậc đỉnh phân cấp).
"""

from typing import Dict, List, Optional, Set, Tuple

# ==============================================================================
# 1. BẢNG TAXONOMY THỰC THỂ (E1 -> E40) - ĐỒNG BỘ 100% AUTO_LABEL_PIPELINE
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

# Danh sách phân nhóm thực thể
THUOC_TINH_LABELS = [k for k, v in ENTITY_TAXONOMY.items() if v[2] == "metadata"]
DOC_REFERENCE_ENTITY_LABELS = [k for k, v in ENTITY_TAXONOMY.items() if v[2] == "reference"]
MUC_LUC_LABELS = [k for k, v in ENTITY_TAXONOMY.items() if v[2] == "structure"]
LUOC_DO_ENTITY_LABELS = DOC_REFERENCE_ENTITY_LABELS + ["van_ban_lien_quan"]
ALL_ENTITIES = list(ENTITY_TAXONOMY.keys())

# Nhãn BIO cho mô hình nhận diện thực thể (Token-level NER)
BIO_LABELS = ["o"]
for entity in ALL_ENTITIES:
    BIO_LABELS.append(f"b-{entity.lower()}")
    BIO_LABELS.append(f"i-{entity.lower()}")

LABEL2ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# ==============================================================================
# 2. BẢNG TAXONOMY QUAN HỆ (R1 -> R22) - 22 QUAN HỆ PHÁP LÝ & CẤU TRÚC
# (Đã loại bỏ hoàn toàn các quan hệ thời gian suy diễn)
# ==============================================================================
RELATION_TAXONOMY: Dict[str, Tuple[int, str]] = {
    # 2.1 Quan hệ cấu trúc nội bộ & hành chính (R22 đặt lên index 0)
    "THUOC": (0, "Thuộc"),

    # 2.2 Quan hệ pháp lý liên văn bản (R1 -> R21)
    "CAN_CU_BAN_HANH": (1, "Căn cứ ban hành"),
    "HUONG_DAN_AP_DUNG": (2, "Hướng dẫn áp dụng"),
    "DUOC_HUONG_DAN_AP_DUNG": (3, "Được hướng dẫn áp dụng"),
    "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": (4, "Quy định chi tiết, hướng dẫn thi hành"),
    "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": (5, "Được quy định chi tiết, hướng dẫn thi hành"),
    "QUY_DINH_HET_HIEU_LUC_MOT_PHAN": (6, "Quy định về việc hết hiệu lực một phần"),
    "BI_TAC_DONG_HIEU_LUC_MOT_PHAN": (7, "Bị tác động hiệu lực một phần"),
    "QUY_DINH_HET_HIEU_LUC_TOAN_BO": (8, "Quy định về việc hết hiệu lực toàn bộ"),
    "BI_TAC_DONG_HIEU_LUC_TOAN_BO": (9, "Bị tác động hiệu lực toàn bộ"),
    "DINH_CHINH": (10, "Đính chính"),
    "DUOC_DINH_CHINH": (11, "Được đính chính"),
    "DAN_CHIEU": (12, "Dẫn chiếu"),
    "DUOC_DAN_CHIEU": (13, "Được dẫn chiếu"),
    "AP_DUNG": (14, "Áp dụng"),
    "GIAI_THICH": (15, "Giải thích"),
    "DUOC_GIAI_THICH": (16, "Được giải thích"),
    "HOP_NHAT": (17, "Hợp nhất"),
    "DINH_CHI_THI_HANH": (18, "Đình chỉ thi hành"),
    "BI_DINH_CHI_THI_HANH": (19, "Bị đình chỉ thi hành"),
    "QUY_DINH_TAM_NGUNG_GIA_HAN": (20, "Quy định tạm ngưng/ gia hạn"),
    "BI_TAM_NGUNG_GIA_HAN": (21, "Bị tạm ngưng/ gia hạn")
}

ALL_RELATIONS = list(RELATION_TAXONOMY.keys())
REL2ID = {r: idx for idx, r in enumerate(ALL_RELATIONS)}
ID2REL = {idx: r for idx, r in enumerate(ALL_RELATIONS)}
REL_VN_NAMES = {r: info[1] for r, info in RELATION_TAXONOMY.items()}

# Danh sách phân nhóm quan hệ
METADATA_RELATIONS: List[str] = []
STRUCTURAL_RELATIONS: List[str] = ["THUOC"]
LEGAL_RELATIONS: List[str] = [r for r in ALL_RELATIONS if r != "THUOC"]
TEMPORAL_RELATIONS: List[str] = []
NON_TEMP_RELS = ALL_RELATIONS

# ==============================================================================
# 3. THỰC THỂ & QUAN HỆ CHUẨN HÓA (KHÔNG SỬ DỤNG 10 CẤP BẬC ĐỈNH)
# ==============================================================================
# Giữ các biến rỗng để đảm bảo tương thích ngược tuyệt đối
LEVEL_LIST: List[str] = []
LEVEL2ID: Dict[str, int] = {}
ID2LEVEL: Dict[int, str] = {}
NUM_LEVELS: int = 0
ENTITY_TO_LEVEL_MAP: Dict[str, str] = {}

def get_node_level(code_or_label: str, scope: Optional[str] = None) -> str:
    """Tương thích ngược: trả về mã loại thực thể"""
    return str(code_or_label).strip() if code_or_label else "entity"

# ==============================================================================
# 4. BẢNG ÁNH XẠ TOÀN DIỆN ALIAS CHO QUAN HỆ & THỰC THỂ
# ==============================================================================
RELATION_ALIAS_MAP: Dict[str, str] = {
    # Thuộc
    "THUOC": "THUOC",
    "THUỘC": "THUOC",
    "BELONGS_TO": "THUOC",
    "CONTAINS": "THUOC",

    # Căn cứ ban hành
    "CAN_CU_BAN_HANH": "CAN_CU_BAN_HANH",
    "CĂN CỨ BAN HÀNH": "CAN_CU_BAN_HANH",
    "CĂN CỨ VÀO": "CAN_CU_BAN_HANH",
    "LÀM CĂN CỨ BAN HÀNH CHO": "CAN_CU_BAN_HANH",
    "DUOC_CAN_CU_BAN_HANH": "CAN_CU_BAN_HANH",
    "CAN_CU_VAO": "CAN_CU_BAN_HANH",

    # Hướng dẫn áp dụng
    "HUONG_DAN_AP_DUNG": "HUONG_DAN_AP_DUNG",
    "HƯỚNG DẪN ÁP DỤNG": "HUONG_DAN_AP_DUNG",
    "HUONG_DAN": "HUONG_DAN_AP_DUNG",
    "VB_HUONG_DAN_AP_DUNG": "HUONG_DAN_AP_DUNG",
    "DUOC_HUONG_DAN_AP_DUNG": "DUOC_HUONG_DAN_AP_DUNG",
    "ĐƯỢC HƯỚNG DẪN ÁP DỤNG": "DUOC_HUONG_DAN_AP_DUNG",
    "ĐƯỢC HƯỚNG DẪN ÁP DỤNG BỞI": "DUOC_HUONG_DAN_AP_DUNG",
    "DUOC_HUONG_DAN": "DUOC_HUONG_DAN_AP_DUNG",
    "VB_DUOC_HUONG_DAN_AP_DUNG": "DUOC_HUONG_DAN_AP_DUNG",

    # Quy định chi tiết, hướng dẫn thi hành
    "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "CHI_TIET_HUONG_DAN_THI_HANH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "QUY_DINH_CHI_TIET": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "QUY ĐỊNH CHI TIẾT, HƯỚNG DẪN THI HÀNH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "QUY ĐỊNH CHI TIẾT / HƯỚNG DẪN THI HÀNH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "ĐƯỢC QUY ĐỊNH CHI TIẾT, HƯỚNG DẪN THI HÀNH": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "ĐƯỢC QUY ĐỊNH CHI TIẾT / HƯỚNG DẪN THI HÀNH BỞI": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",

    # Hết hiệu lực một phần / Bị tác động hiệu lực một phần
    "QUY_DINH_HET_HIEU_LUC_MOT_PHAN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "HET_HIEU_LUC_MOT_PHAN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "QUY ĐỊNH VỀ VIỆC HẾT HIỆU LỰC MỘT PHẦN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "LÀM HẾT HIỆU LỰC MỘT PHẦN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "BI_TAC_DONG_HIEU_LUC_MOT_PHAN": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BỊ TÁC ĐỘNG HIỆU LỰC MỘT PHẦN": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BỊ LÀM HẾT HIỆU LỰC MỘT PHẦN BỞI": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",

    # Hết hiệu lực toàn bộ / Bị tác động hiệu lực toàn bộ
    "QUY_DINH_HET_HIEU_LUC_TOAN_BO": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "HET_HIEU_LUC_TOAN_BO": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "HET_HIEU_LUC": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "QUY ĐỊNH VỀ VIỆC HẾT HIỆU LỰC TOÀN BỘ": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "LÀM HẾT HIỆU LỰC TOÀN BỘ": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "BI_TAC_DONG_HIEU_LUC_TOAN_BO": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "TAC_DONG_HIEU_LUC_TOAN_BO": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BỊ TÁC ĐỘNG HIỆU LỰC TOÀN BỘ": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BỊ LÀM HẾT HIỆU LỰC TOÀN BỘ BỞI": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",

    # Đính chính
    "DINH_CHINH": "DINH_CHINH",
    "ĐÍNH CHÍNH": "DINH_CHINH",
    "VB_DINH_CHINH": "DINH_CHINH",
    "DUOC_DINH_CHINH": "DUOC_DINH_CHINH",
    "ĐƯỢC ĐÍNH CHÍNH": "DUOC_DINH_CHINH",
    "ĐƯỢC ĐÍNH CHÍNH BỞI": "DUOC_DINH_CHINH",
    "VB_DUOC_DINH_CHINH": "DUOC_DINH_CHINH",

    # Dẫn chiếu
    "DAN_CHIEU": "DAN_CHIEU",
    "DẪN CHIẾU": "DAN_CHIEU",
    "DẪN CHIẾU ĐẾN": "DAN_CHIEU",
    "VB_DAN_CHIEU": "DAN_CHIEU",
    "DUOC_DAN_CHIEU": "DUOC_DAN_CHIEU",
    "ĐƯỢC DẪN CHIẾU": "DUOC_DAN_CHIEU",
    "ĐƯỢC DẪN CHIẾU BỞI": "DUOC_DAN_CHIEU",
    "VB_DUOC_DAN_CHIEU": "DUOC_DAN_CHIEU",

    # Áp dụng
    "AP_DUNG": "AP_DUNG",
    "ÁP DỤNG": "AP_DUNG",
    "VB_AP_DUNG": "AP_DUNG",

    # Giải thích
    "GIAI_THICH": "GIAI_THICH",
    "GIẢI THÍCH": "GIAI_THICH",
    "GIẢI THÍCH CHO": "GIAI_THICH",
    "VB_GIAI_THICH": "GIAI_THICH",
    "DUOC_GIAI_THICH": "DUOC_GIAI_THICH",
    "ĐƯỢC GIẢI THÍCH": "DUOC_GIAI_THICH",
    "ĐƯỢC GIẢI THÍCH BỞI": "DUOC_GIAI_THICH",
    "VB_DUOC_GIAI_THICH": "DUOC_GIAI_THICH",

    # Hợp nhất
    "HOP_NHAT": "HOP_NHAT",
    "HỢP NHẤT": "HOP_NHAT",
    "HỢP NHẤT VÀO / HỢP NHẤT": "HOP_NHAT",
    "VB_HOP_NHAT": "HOP_NHAT",

    # Đình chỉ thi hành
    "DINH_CHI_THI_HANH": "DINH_CHI_THI_HANH",
    "ĐÌNH CHỈ THI HÀNH": "DINH_CHI_THI_HANH",
    "VB_DINH_CHI_THI_HANH": "DINH_CHI_THI_HANH",
    "BI_DINH_CHI_THI_HANH": "BI_DINH_CHI_THI_HANH",
    "BỊ ĐÌNH CHỈ THI HÀNH": "BI_DINH_CHI_THI_HANH",
    "BỊ ĐÌNH CHỈ THI HÀNH BỞI": "BI_DINH_CHI_THI_HANH",
    "VB_BI_DINH_CHI_THI_HANH": "BI_DINH_CHI_THI_HANH",

    # Tạm ngưng / gia hạn
    "QUY_DINH_TAM_NGUNG_GIA_HAN": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "TẠM NGƯNG / GIA HẠN / CÔNG BỐ TIẾP TỤC HIỆU LỰC": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "VB_QUY_DINH_TAM_NGUNG_GIA_HAN": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "BI_TAM_NGUNG_GIA_HAN": "BI_TAM_NGUNG_GIA_HAN",
    "BỊ TẠM NGƯNG / GIA HẠN": "BI_TAM_NGUNG_GIA_HAN",
    "BỊ TẠM NGƯNG / ĐƯỢC GIA HẠN / ĐƯỢC CÔNG BỐ TIẾP TỤC HIỆU LỰC BỞI": "BI_TAM_NGUNG_GIA_HAN",
    "VB_BI_TAM_NGUNG_GIA_HAN": "BI_TAM_NGUNG_GIA_HAN"
}

def normalize_relation_label(rel: str) -> str:
    """Chuẩn hóa mọi biểu diễn nhãn quan hệ sang mã UPPERCASE chuẩn trong ALL_RELATIONS."""
    if not rel:
        return "THUOC"
    clean = str(rel).strip().upper()
    if clean in RELATION_ALIAS_MAP:
        return RELATION_ALIAS_MAP[clean]
    clean_lower = clean.lower()
    for k, v in RELATION_ALIAS_MAP.items():
        if k.lower() == clean_lower:
            return v
    if clean in ALL_RELATIONS:
        return clean
    return "THUOC"

def normalize_entity_label(label: str) -> str:
    """Chuẩn hóa tên thực thể sang mã UPPERCASE chuẩn trong ENTITY_TAXONOMY."""
    if not label:
        return "DIEU"
    clean = str(label).strip().upper()
    if clean in ENTITY_TAXONOMY:
        return clean
    for code, info in ENTITY_TAXONOMY.items():
        if info[1].strip().upper() == clean:
            return code
    return clean

# Quan hệ đối ngẫu
INVERSE_RELATION_MAP: Dict[str, str] = {
    "THUOC": "THUOC",
    "HUONG_DAN_AP_DUNG": "DUOC_HUONG_DAN_AP_DUNG",
    "DUOC_HUONG_DAN_AP_DUNG": "HUONG_DAN_AP_DUNG",
    "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "QUY_DINH_HET_HIEU_LUC_MOT_PHAN": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BI_TAC_DONG_HIEU_LUC_MOT_PHAN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "QUY_DINH_HET_HIEU_LUC_TOAN_BO": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BI_TAC_DONG_HIEU_LUC_TOAN_BO": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "DINH_CHINH": "DUOC_DINH_CHINH",
    "DUOC_DINH_CHINH": "DINH_CHINH",
    "DAN_CHIEU": "DUOC_DAN_CHIEU",
    "DUOC_DAN_CHIEU": "DAN_CHIEU",
    "CAN_CU_BAN_HANH": "AP_DUNG",
    "AP_DUNG": "CAN_CU_BAN_HANH",
    "GIAI_THICH": "DUOC_GIAI_THICH",
    "DUOC_GIAI_THICH": "GIAI_THICH",
    "HOP_NHAT": "HOP_NHAT",
    "DINH_CHI_THI_HANH": "BI_DINH_CHI_THI_HANH",
    "BI_DINH_CHI_THI_HANH": "DINH_CHI_THI_HANH",
    "QUY_DINH_TAM_NGUNG_GIA_HAN": "BI_TAM_NGUNG_GIA_HAN",
    "BI_TAM_NGUNG_GIA_HAN": "QUY_DINH_TAM_NGUNG_GIA_HAN"
}

SYMMETRIC_RELATIONS: Set[str] = {
    "HOP_NHAT"
}

def get_inverse_relation(relation: str) -> Optional[str]:
    norm = normalize_relation_label(relation)
    return INVERSE_RELATION_MAP.get(norm)

def is_symmetric_relation(relation: str) -> bool:
    norm = normalize_relation_label(relation)
    return norm in SYMMETRIC_RELATIONS

def is_strictly_directed(relation: str) -> bool:
    return not is_symmetric_relation(relation)

# Ánh xạ XML để tương thích ngược
LABEL_XML_ENTITY_MAP = {info[1]: code for code, info in ENTITY_TAXONOMY.items()}
LABEL_XML_RELATION_MAP = {info[1]: code for code, info in RELATION_TAXONOMY.items()}
CODE_TO_LABEL_XML_ENTITY = {code: info[1] for code, info in ENTITY_TAXONOMY.items()}
CODE_TO_LABEL_XML_RELATION = {code: info[1] for code, info in RELATION_TAXONOMY.items()}
