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


ENTITY_ALIAS_MAP: Dict[str, str] = {
    "AP_DUNG": "VB_AP_DUNG",
    "BAI_BO": "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "BAN_DICH": "VB_DAN_CHIEU",
    "BAN_DICH_TIENG_ANH": "VB_DAN_CHIEU",
    "BI_BAI_BO": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BI_BO_SUNG": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BI_DICH": "VB_DUOC_DAN_CHIEU",
    "BI_DINH_CHI": "VB_BI_DINH_CHI_THI_HANH",
    "BI_DINH_CHI_THI_HANH": "VB_BI_DINH_CHI_THI_HANH",
    "BI_GIA_HAN": "VB_BI_TAM_NGUNG_GIA_HAN",
    "BI_SUA_DOI": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BI_SUA_DOI_BO_SUNG": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BI_TAM_NGUNG": "VB_BI_TAM_NGUNG_GIA_HAN",
    "BI_TAM_NGUNG_GIA_HAN": "VB_BI_TAM_NGUNG_GIA_HAN",
    "BI_TAM_NGUNG_HIEU_LUC": "VB_BI_TAM_NGUNG_GIA_HAN",
    "BI_THAY_THE": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BO_SUNG": "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "BẢN DỊCH": "VB_DAN_CHIEU",
    "BẢN DỊCH TIẾNG ANH": "VB_DAN_CHIEU",
    "CAN_CU": "CAN_CU_BAN_HANH",
    "CHI_TIET": "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "CHI_TIET_HUONG_DAN_THI_HANH": "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "CONG_BAO": "SO_HIEU",
    "CONG_BO": "VB_DAN_CHIEU",
    "CÔNG BÁO": "SO_HIEU",
    "CÔNG BỐ": "VB_DAN_CHIEU",
    "CĂN CỨ": "CAN_CU_BAN_HANH",
    "DAN_CHIEU": "VB_DAN_CHIEU",
    "DINH_CHI": "VB_DINH_CHI_THI_HANH",
    "DINH_CHINH": "VB_DINH_CHINH",
    "DINH_CHI_THI_HANH": "VB_DINH_CHI_THI_HANH",
    "DINH_KEM": "VB_DAN_CHIEU",
    "DUOC_BAI_BO": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "DUOC_BO_SUNG": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "DUOC_CHI_TIET": "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_CHI_TIET_HUONG_DAN_THI_HANH": "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_CONG_BO": "VB_DUOC_DAN_CHIEU",
    "DUOC_DAN_CHIEU": "VB_DUOC_DAN_CHIEU",
    "DUOC_DICH": "VB_DUOC_DAN_CHIEU",
    "DUOC_DINH_CHINH": "VB_DUOC_DINH_CHINH",
    "DUOC_GIAI_THICH": "VB_DUOC_GIAI_THICH",
    "DUOC_HUONG_DAN": "VB_DUOC_HUONG_DAN_AP_DUNG",
    "DUOC_HUONG_DAN_AP_DUNG": "VB_DUOC_HUONG_DAN_AP_DUNG",
    "DUOC_HUONG_DAN_THI_HANH": "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_QUY_DINH_CHI_TIET": "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_SUA_DOI": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "DUOC_SUA_DOI_BO_SUNG": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "DUOC_TAM_NGUNG_HIEU_LUC": "VB_BI_TAM_NGUNG_GIA_HAN",
    "DUOC_THAY_THE": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "ENGLISH": "VB_DAN_CHIEU",
    "GIAI_THICH": "VB_GIAI_THICH",
    "GIA_HAN": "VB_QUY_DINH_TAM_NGUNG_GIA_HAN",
    "GIA_HAN_HIEU_LUC": "VB_QUY_DINH_TAM_NGUNG_GIA_HAN",
    "HET_HIEU_LUC": "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "HET_HIEU_LUC_MOT_PHAN": "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "HET_HIEU_LUC_TOAN_BO": "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "HOP_NHAT": "VB_HOP_NHAT",
    "HUONG_DAN": "VB_HUONG_DAN_AP_DUNG",
    "HUONG_DAN_AP_DUNG": "VB_HUONG_DAN_AP_DUNG",
    "HUONG_DAN_THI_HANH": "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "NGAY_DANG_CONG_BAO": "NGAY_BAN_HANH",
    "NGÀY ĐĂNG CÔNG BÁO": "NGAY_BAN_HANH",
    "PHU_LUC": "VB_DAN_CHIEU",
    "PHỤ LỤC": "VB_DAN_CHIEU",
    "QUY_DINH_CHI_TIET": "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "SO_CONG_BAO": "SO_HIEU",
    "SUA_DOI": "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "SUA_DOI_BO_SUNG": "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "SỐ CÔNG BÁO": "SO_HIEU",
    "TAI_LIEU_DINH_KEM": "VB_DAN_CHIEU",
    "TAM_NGUNG": "VB_QUY_DINH_TAM_NGUNG_GIA_HAN",
    "TAM_NGUNG_GIA_HAN": "VB_QUY_DINH_TAM_NGUNG_GIA_HAN",
    "TAM_NGUNG_HIEU_LUC": "VB_QUY_DINH_TAM_NGUNG_GIA_HAN",
    "THAY_THE": "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "TIENG_ANH": "VB_DAN_CHIEU",
    "TIẾNG ANH": "VB_DAN_CHIEU",
    "TO_TRINH": "CAN_CU_BAN_HANH",
    "TRANSLATION": "VB_DAN_CHIEU",
    "TÀI LIỆU ĐÍNH KÈM": "VB_DAN_CHIEU",
    "TỜ TRÌNH": "CAN_CU_BAN_HANH",
    "VAN_BAN_LIEN_QUAN": "VB_DAN_CHIEU",
    "VB_BAI_BO": "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "VB_BAN_DICH": "VB_DAN_CHIEU",
    "VB_BI_BAI_BO": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_BI_BO_SUNG": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_BI_DINH_CHI": "VB_BI_DINH_CHI_THI_HANH",
    "VB_BI_GIA_HAN": "VB_BI_TAM_NGUNG_GIA_HAN",
    "VB_BI_HET_HIEU_LUC": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_BI_HET_HIEU_LUC_MOT_PHAN": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_BI_HET_HIEU_LUC_TOAN_BO": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_BI_SUA_DOI": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_BI_SUA_DOI_BO_SUNG": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_BI_TAM_NGUNG": "VB_BI_TAM_NGUNG_GIA_HAN",
    "VB_BI_TAM_NGUNG_GIA_HAN": "VB_BI_TAM_NGUNG_GIA_HAN",
    "VB_BI_TAM_NGUNG_HIEU_LUC": "VB_BI_TAM_NGUNG_GIA_HAN",
    "VB_BI_THAY_THE": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_BO_SUNG": "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VB_CHI_TIET": "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_CHI_TIET_HUONG_DAN_THI_HANH": "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_CONG_BO": "VB_DAN_CHIEU",
    "VB_DINH_CHI": "VB_DINH_CHI_THI_HANH",
    "VB_DUOC_BAI_BO": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_DUOC_BO_SUNG": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_DUOC_CHI_TIET": "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_DUOC_CHI_TIET_HUONG_DAN_THI_HANH": "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_DUOC_CONG_BO": "VB_DUOC_DAN_CHIEU",
    "VB_DUOC_DICH": "VB_DUOC_DAN_CHIEU",
    "VB_DUOC_HUONG_DAN": "VB_DUOC_HUONG_DAN_AP_DUNG",
    "VB_DUOC_HUONG_DAN_THI_HANH": "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_DUOC_QUY_DINH_CHI_TIET": "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_DUOC_SUA_DOI": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_DUOC_SUA_DOI_BO_SUNG": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_DUOC_TAM_NGUNG_HIEU_LUC": "VB_BI_TAM_NGUNG_GIA_HAN",
    "VB_DUOC_THAY_THE": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_GIA_HAN": "VB_QUY_DINH_TAM_NGUNG_GIA_HAN",
    "VB_GIA_HAN_HIEU_LUC": "VB_QUY_DINH_TAM_NGUNG_GIA_HAN",
    "VB_HET_HIEU_LUC": "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "VB_HET_HIEU_LUC_MOT_PHAN": "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VB_HET_HIEU_LUC_TOAN_BO": "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "VB_HUONG_DAN": "VB_HUONG_DAN_AP_DUNG",
    "VB_HUONG_DAN_THI_HANH": "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_LAM_HET_HIEU_LUC": "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "VB_LAM_HET_HIEU_LUC_MOT_PHAN": "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VB_LAM_HET_HIEU_LUC_TOAN_BO": "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "VB_LIEN_QUAN": "VB_DAN_CHIEU",
    "VB_PHU_LUC": "VB_DAN_CHIEU",
    "VB_QUY_DINH_CHI_TIET": "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_SUA_DOI": "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VB_SUA_DOI_BO_SUNG": "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VB_TAC_DONG_HIEU_LUC_MOT_PHAN": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_TAC_DONG_HIEU_LUC_TOAN_BO": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_TAM_NGUNG": "VB_QUY_DINH_TAM_NGUNG_GIA_HAN",
    "VB_TAM_NGUNG_GIA_HAN": "VB_QUY_DINH_TAM_NGUNG_GIA_HAN",
    "VB_TAM_NGUNG_HIEU_LUC": "VB_QUY_DINH_TAM_NGUNG_GIA_HAN",
    "VB_THAY_THE": "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "VB_TIENG_ANH": "VB_DAN_CHIEU",
    "VB_TO_TRINH": "CAN_CU_BAN_HANH",
    "VĂN BẢN BÃI BỎ": "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "VĂN BẢN BỊ BÃI BỎ": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VĂN BẢN BỊ SỬA ĐỔI BỔ SUNG": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VĂN BẢN BỊ THAY THẾ": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VĂN BẢN CÔNG BỐ": "VB_DAN_CHIEU",
    "VĂN BẢN DỊCH": "VB_DAN_CHIEU",
    "VĂN BẢN HƯỚNG DẪN THI HÀNH": "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VĂN BẢN LIÊN QUAN": "VB_DAN_CHIEU",
    "VĂN BẢN QUY ĐỊNH CHI TIẾT": "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VĂN BẢN SỬA ĐỔI BỔ SUNG": "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VĂN BẢN SỬA ĐỔI, BỔ SUNG": "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VĂN BẢN THAY THẾ": "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "VĂN BẢN ĐƯỢC BÃI BỎ": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VĂN BẢN ĐƯỢC CÔNG BỐ": "VB_DAN_CHIEU",
    "VĂN BẢN ĐƯỢC HƯỚNG DẪN THI HÀNH": "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VĂN BẢN ĐƯỢC QUY ĐỊNH CHI TIẾT": "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VĂN BẢN ĐƯỢC SỬA ĐỔI BỔ SUNG": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VĂN BẢN ĐƯỢC SỬA ĐỔI, BỔ SUNG": "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VĂN BẢN ĐƯỢC THAY THẾ": "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "ĐÍNH KÈM": "VB_DAN_CHIEU",
}

def get_node_level(code_or_label: str, scope: Optional[str] = None) -> str:
    """Tương thích ngược: trả về mã loại thực thể"""
    return str(code_or_label).strip() if code_or_label else "entity"

# ==============================================================================
# 4. BẢNG ÁNH XẠ TOÀN DIỆN ALIAS CHO QUAN HỆ & THỰC THỂ
# ==============================================================================
RELATION_ALIAS_MAP: Dict[str, str] = {
    "AP_DUNG": "AP_DUNG",
    "BAI_BO": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "BAN_DICH": "DAN_CHIEU",
    "BAN_DICH_TIENG_ANH": "DAN_CHIEU",
    "BELONGS_TO": "THUOC",
    "BI_BAI_BO": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BI_BO_SUNG": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BI_DICH": "DUOC_DAN_CHIEU",
    "BI_DINH_CHI": "BI_DINH_CHI_THI_HANH",
    "BI_DINH_CHI_THI_HANH": "BI_DINH_CHI_THI_HANH",
    "BI_GIA_HAN": "BI_TAM_NGUNG_GIA_HAN",
    "BI_HET_HIEU_LUC": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BI_HET_HIEU_LUC_MOT_PHAN": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BI_HET_HIEU_LUC_TOAN_BO": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BI_SUA_DOI": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BI_SUA_DOI_BO_SUNG": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BI_TAC_DONG_HIEU_LUC_MOT_PHAN": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BI_TAC_DONG_HIEU_LUC_TOAN_BO": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BI_TAM_NGUNG": "BI_TAM_NGUNG_GIA_HAN",
    "BI_TAM_NGUNG_GIA_HAN": "BI_TAM_NGUNG_GIA_HAN",
    "BI_TAM_NGUNG_HIEU_LUC": "BI_TAM_NGUNG_GIA_HAN",
    "BI_THAY_THE": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BO_SUNG": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "BÃI BỎ": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "BẢN DỊCH": "DAN_CHIEU",
    "BẢN DỊCH TIẾNG ANH": "DAN_CHIEU",
    "BỊ BÃI BỎ": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BỊ BỔ SUNG": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BỊ GIA HẠN": "BI_TAM_NGUNG_GIA_HAN",
    "BỊ HẾT HIỆU LỰC": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BỊ HẾT HIỆU LỰC MỘT PHẦN": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BỊ HẾT HIỆU LỰC TOÀN BỘ": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BỊ LÀM HẾT HIỆU LỰC MỘT PHẦN BỞI": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BỊ LÀM HẾT HIỆU LỰC TOÀN BỘ BỞI": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BỊ SỬA ĐỔI": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BỊ SỬA ĐỔI BỔ SUNG": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BỊ THAY THẾ": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BỊ TÁC ĐỘNG HIỆU LỰC MỘT PHẦN": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "BỊ TÁC ĐỘNG HIỆU LỰC TOÀN BỘ": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "BỊ TẠM NGƯNG": "BI_TAM_NGUNG_GIA_HAN",
    "BỊ TẠM NGƯNG / GIA HẠN": "BI_TAM_NGUNG_GIA_HAN",
    "BỊ TẠM NGƯNG / ĐƯỢC GIA HẠN / ĐƯỢC CÔNG BỐ TIẾP TỤC HIỆU LỰC BỞI": "BI_TAM_NGUNG_GIA_HAN",
    "BỊ ĐÌNH CHỈ": "BI_DINH_CHI_THI_HANH",
    "BỊ ĐÌNH CHỈ THI HÀNH": "BI_DINH_CHI_THI_HANH",
    "BỊ ĐÌNH CHỈ THI HÀNH BỞI": "BI_DINH_CHI_THI_HANH",
    "BỔ SUNG": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "CAN_CU": "CAN_CU_BAN_HANH",
    "CAN_CU_BAN_HANH": "CAN_CU_BAN_HANH",
    "CAN_CU_VAO": "CAN_CU_BAN_HANH",
    "CHI_TIET": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "CHI_TIET_HUONG_DAN_THI_HANH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "CONG_BO": "DAN_CHIEU",
    "CONTAINS": "THUOC",
    "CÔNG BỐ": "DAN_CHIEU",
    "CĂN CỨ": "CAN_CU_BAN_HANH",
    "CĂN CỨ BAN HÀNH": "CAN_CU_BAN_HANH",
    "CĂN CỨ VÀO": "CAN_CU_BAN_HANH",
    "DAN_CHIEU": "DAN_CHIEU",
    "DINH_CHI": "DINH_CHI_THI_HANH",
    "DINH_CHINH": "DINH_CHINH",
    "DINH_CHI_THI_HANH": "DINH_CHI_THI_HANH",
    "DINH_KEM": "DAN_CHIEU",
    "DUOC_BAI_BO": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "DUOC_BO_SUNG": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "DUOC_CAN_CU_BAN_HANH": "CAN_CU_BAN_HANH",
    "DUOC_CHI_TIET": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_CHI_TIET_HUONG_DAN_THI_HANH": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_CONG_BO": "DUOC_DAN_CHIEU",
    "DUOC_DAN_CHIEU": "DUOC_DAN_CHIEU",
    "DUOC_DICH": "DUOC_DAN_CHIEU",
    "DUOC_DINH_CHINH": "DUOC_DINH_CHINH",
    "DUOC_GIAI_THICH": "DUOC_GIAI_THICH",
    "DUOC_HUONG_DAN": "DUOC_HUONG_DAN_AP_DUNG",
    "DUOC_HUONG_DAN_AP_DUNG": "DUOC_HUONG_DAN_AP_DUNG",
    "DUOC_HUONG_DAN_THI_HANH": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_QUY_DINH_CHI_TIET": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_QUY_DINH_CHI_TIET_THI_HANH": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_SUA_DOI": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "DUOC_SUA_DOI_BO_SUNG": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "DUOC_TAM_NGUNG_HIEU_LUC": "BI_TAM_NGUNG_GIA_HAN",
    "DUOC_THAY_THE": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "DẪN CHIẾU": "DAN_CHIEU",
    "DẪN CHIẾU ĐẾN": "DAN_CHIEU",
    "ENGLISH": "DAN_CHIEU",
    "GIA HẠN": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "GIAI_THICH": "GIAI_THICH",
    "GIA_HAN": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "GIA_HAN_HIEU_LUC": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "GIẢI THÍCH": "GIAI_THICH",
    "GIẢI THÍCH CHO": "GIAI_THICH",
    "HET_HIEU_LUC": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "HET_HIEU_LUC_MOT_PHAN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "HET_HIEU_LUC_TOAN_BO": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "HOP_NHAT": "HOP_NHAT",
    "HUONG_DAN": "HUONG_DAN_AP_DUNG",
    "HUONG_DAN_AP_DUNG": "HUONG_DAN_AP_DUNG",
    "HUONG_DAN_THI_HANH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "HƯỚNG DẪN": "HUONG_DAN_AP_DUNG",
    "HƯỚNG DẪN THI HÀNH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "HƯỚNG DẪN ÁP DỤNG": "HUONG_DAN_AP_DUNG",
    "HẾT HIỆU LỰC": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "HẾT HIỆU LỰC MỘT PHẦN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "HẾT HIỆU LỰC TOÀN BỘ": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "HỢP NHẤT": "HOP_NHAT",
    "HỢP NHẤT VÀO / HỢP NHẤT": "HOP_NHAT",
    "LAM_HET_HIEU_LUC": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "LAM_HET_HIEU_LUC_MOT_PHAN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "LAM_HET_HIEU_LUC_TOAN_BO": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "LÀM CĂN CỨ BAN HÀNH CHO": "CAN_CU_BAN_HANH",
    "LÀM HẾT HIỆU LỰC MỘT PHẦN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "LÀM HẾT HIỆU LỰC TOÀN BỘ": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "PHU_LUC": "DAN_CHIEU",
    "PHỤ LỤC": "DAN_CHIEU",
    "QUY ĐỊNH CHI TIẾT": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "QUY ĐỊNH CHI TIẾT / HƯỚNG DẪN THI HÀNH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "QUY ĐỊNH CHI TIẾT, HƯỚNG DẪN THI HÀNH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "QUY ĐỊNH HẾT HIỆU LỰC MỘT PHẦN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "QUY ĐỊNH HẾT HIỆU LỰC TOÀN BỘ": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "QUY ĐỊNH VỀ VIỆC HẾT HIỆU LỰC MỘT PHẦN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "QUY ĐỊNH VỀ VIỆC HẾT HIỆU LỰC TOÀN BỘ": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "QUY_DINH_CHI_TIET": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "QUY_DINH_CHI_TIET_THI_HANH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "QUY_DINH_HET_HIEU_LUC": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "QUY_DINH_HET_HIEU_LUC_MOT_PHAN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "QUY_DINH_HET_HIEU_LUC_TOAN_BO": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "QUY_DINH_TAM_NGUNG_GIA_HAN": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "SUA_DOI": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "SUA_DOI_BO_SUNG": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "SỬA ĐỔI": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "SỬA ĐỔI BỔ SUNG": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "SỬA ĐỔI, BỔ SUNG": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "TAC_DONG_HIEU_LUC": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "TAC_DONG_HIEU_LUC_MOT_PHAN": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "TAC_DONG_HIEU_LUC_TOAN_BO": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "TAI_LIEU_DINH_KEM": "DAN_CHIEU",
    "TAM_NGUNG": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "TAM_NGUNG_GIA_HAN": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "TAM_NGUNG_HIEU_LUC": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "THAY THẾ": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "THAY_THE": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "THUOC": "THUOC",
    "THUỘC": "THUOC",
    "TIENG_ANH": "DAN_CHIEU",
    "TIẾNG ANH": "DAN_CHIEU",
    "TO_TRINH": "CAN_CU_BAN_HANH",
    "TRANSLATION": "DAN_CHIEU",
    "TÀI LIỆU ĐÍNH KÈM": "DAN_CHIEU",
    "TẠM NGƯNG": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "TẠM NGƯNG / GIA HẠN / CÔNG BỐ TIẾP TỤC HIỆU LỰC": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "TỜ TRÌNH": "CAN_CU_BAN_HANH",
    "VAN_BAN_LIEN_QUAN": "DAN_CHIEU",
    "VB_AP_DUNG": "AP_DUNG",
    "VB_BAI_BO": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "VB_BAN_DICH": "DAN_CHIEU",
    "VB_BI_BAI_BO": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_BI_BO_SUNG": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_BI_DINH_CHI_THI_HANH": "BI_DINH_CHI_THI_HANH",
    "VB_BI_SUA_DOI": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_BI_SUA_DOI_BO_SUNG": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_BI_TAM_NGUNG_GIA_HAN": "BI_TAM_NGUNG_GIA_HAN",
    "VB_BI_TAM_NGUNG_HIEU_LUC": "BI_TAM_NGUNG_GIA_HAN",
    "VB_BI_THAY_THE": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_BO_SUNG": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VB_CAN_CU_BAN_HANH": "CAN_CU_BAN_HANH",
    "VB_CONG_BO": "DAN_CHIEU",
    "VB_DAN_CHIEU": "DAN_CHIEU",
    "VB_DINH_CHINH": "DINH_CHINH",
    "VB_DINH_CHI_THI_HANH": "DINH_CHI_THI_HANH",
    "VB_DUOC_BAI_BO": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_DUOC_BO_SUNG": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_DUOC_CONG_BO": "DUOC_DAN_CHIEU",
    "VB_DUOC_DAN_CHIEU": "DUOC_DAN_CHIEU",
    "VB_DUOC_DICH": "DUOC_DAN_CHIEU",
    "VB_DUOC_DINH_CHINH": "DUOC_DINH_CHINH",
    "VB_DUOC_GIAI_THICH": "DUOC_GIAI_THICH",
    "VB_DUOC_HUONG_DAN": "DUOC_HUONG_DAN_AP_DUNG",
    "VB_DUOC_HUONG_DAN_AP_DUNG": "DUOC_HUONG_DAN_AP_DUNG",
    "VB_DUOC_QUY_DINH_CHI_TIET": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_DUOC_SUA_DOI": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_DUOC_SUA_DOI_BO_SUNG": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "VB_DUOC_TAM_NGUNG_HIEU_LUC": "BI_TAM_NGUNG_GIA_HAN",
    "VB_DUOC_THAY_THE": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "VB_GIAI_THICH": "GIAI_THICH",
    "VB_GIA_HAN_HIEU_LUC": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "VB_HET_HIEU_LUC_MOT_PHAN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VB_HOP_NHAT": "HOP_NHAT",
    "VB_HUONG_DAN": "HUONG_DAN_AP_DUNG",
    "VB_HUONG_DAN_AP_DUNG": "HUONG_DAN_AP_DUNG",
    "VB_LIEN_QUAN": "DAN_CHIEU",
    "VB_PHU_LUC": "DAN_CHIEU",
    "VB_QUY_DINH_CHI_TIET": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH": "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "VB_QUY_DINH_TAM_NGUNG_GIA_HAN": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "VB_SUA_DOI": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VB_SUA_DOI_BO_SUNG": "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "VB_TAM_NGUNG_HIEU_LUC": "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "VB_THAY_THE": "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "VB_TIENG_ANH": "DAN_CHIEU",
    "VB_TO_TRINH": "CAN_CU_BAN_HANH",
    "VĂN BẢN CÔNG BỐ": "DAN_CHIEU",
    "VĂN BẢN DỊCH": "DAN_CHIEU",
    "VĂN BẢN LIÊN QUAN": "DAN_CHIEU",
    "VĂN BẢN ĐƯỢC CÔNG BỐ": "DAN_CHIEU",
    "ÁP DỤNG": "AP_DUNG",
    "ĐÌNH CHỈ": "DINH_CHI_THI_HANH",
    "ĐÌNH CHỈ THI HÀNH": "DINH_CHI_THI_HANH",
    "ĐÍNH CHÍNH": "DINH_CHINH",
    "ĐÍNH KÈM": "DAN_CHIEU",
    "ĐƯỢC BÃI BỎ": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "ĐƯỢC BỔ SUNG": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "ĐƯỢC DẪN CHIẾU": "DUOC_DAN_CHIEU",
    "ĐƯỢC DẪN CHIẾU BỞI": "DUOC_DAN_CHIEU",
    "ĐƯỢC GIẢI THÍCH": "DUOC_GIAI_THICH",
    "ĐƯỢC GIẢI THÍCH BỞI": "DUOC_GIAI_THICH",
    "ĐƯỢC HƯỚNG DẪN": "DUOC_HUONG_DAN_AP_DUNG",
    "ĐƯỢC HƯỚNG DẪN BỞI": "DUOC_HUONG_DAN_AP_DUNG",
    "ĐƯỢC HƯỚNG DẪN THI HÀNH": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "ĐƯỢC HƯỚNG DẪN THI HÀNH BỞI": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "ĐƯỢC HƯỚNG DẪN ÁP DỤNG": "DUOC_HUONG_DAN_AP_DUNG",
    "ĐƯỢC HƯỚNG DẪN ÁP DỤNG BỞI": "DUOC_HUONG_DAN_AP_DUNG",
    "ĐƯỢC QUY ĐỊNH CHI TIẾT": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "ĐƯỢC QUY ĐỊNH CHI TIẾT / HƯỚNG DẪN THI HÀNH BỞI": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "ĐƯỢC QUY ĐỊNH CHI TIẾT BỞI": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "ĐƯỢC QUY ĐỊNH CHI TIẾT, HƯỚNG DẪN THI HÀNH": "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "ĐƯỢC SỬA ĐỔI": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "ĐƯỢC SỬA ĐỔI BỔ SUNG": "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "ĐƯỢC THAY THẾ": "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "ĐƯỢC ĐÍNH CHÍNH": "DUOC_DINH_CHINH",
    "ĐƯỢC ĐÍNH CHÍNH BỞI": "DUOC_DINH_CHINH",
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
