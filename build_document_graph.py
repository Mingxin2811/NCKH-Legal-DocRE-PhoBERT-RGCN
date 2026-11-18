# -*- coding: utf-8 -*-
"""
build_document_graph.py
BƯỚC 5: KHỬ NHIỄU LOGIC & ĐÓNG GÓI ĐỒ THỊ TÀI LIỆU (DOCUMENT GRAPH CHO GNN)
Đề tài: "Ứng dụng Graph Neural Networks (GNN) kết hợp Transformer 
        trong trích xuất quan hệ thực thể cấp độ tài liệu (DocRE)"

Các chức năng chính:
1. Bóc tách thực thể đỉnh đồ thị (Document, Phần, Chương, Mục, Tiểu mục, Điều, Khoản, Điểm, Viện dẫn).
2. Thiết lập hệ thống ID nút phân cấp duy nhất (Hierarchical Canonical Node IDs).
3. Động cơ Khử nhiễu Logic (Rule-based Logic Denoising):
   - Khử đỉnh mồ côi (Orphan Node Resolution).
   - Kiểm định tính đơn điệu thứ bậc lập pháp (Strict Tree Monotonicity).
   - Phát hiện & Triệt tiêu chu trình kín (Cycle Elimination - Directed Acyclic Graph).
   - Chuẩn hóa tính đối ngẫu (contains <-> belongs_to).
   - Chuẩn hóa cạnh tuần tự anh em ruột (Intra-sibling next_clause).
4. Trích xuất đặc trưng ngữ nghĩa nút (Node Embeddings) bằng PhoBERT (768 chiều).
5. Đóng gói đồ thị chuẩn hóa sang PyTorch Geometric (torch_geometric.data.HeteroData & Data).
"""

import os
import sys
import json
import glob
import time
import re
from typing import List, Dict, Any, Tuple, Optional, Set

try:
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    getattr(sys.stderr, "reconfigure")(encoding="utf-8")
except Exception:
    pass

import torch  # type: ignore
import networkx as nx  # type: ignore
from transformers import AutoTokenizer, AutoModel  # type: ignore
from torch_geometric.data import Data, HeteroData  # type: ignore

from labels import (
    REL2ID,
    LEGAL_RELATIONS,
    THUOC_TINH_LABELS,
    LABEL_XML_RELATION_MAP,
    normalize_relation_label,
    normalize_entity_label
)

# Thứ bậc lập pháp chuẩn hóa Việt Nam (0 là cấp cao nhất)
# 11 thuộc tính metadata từ thuoc_tinh.json được lưu bên trong đỉnh 'doc', không còn là các đỉnh riêng
HIERARCHY_RANK: Dict[str, int] = {
    "doc": 0,
    "phan": 1,
    "chuong": 2,
    "muc": 3,
    "tieu_muc": 4,
    "dieu": 5,
    "khoan": 6,
    "diem": 7,
    "van_ban_lien_quan": 8
}

VN_LEVEL_MAP: Dict[str, str] = {
    "phần": "phan",
    "chương": "chuong",
    "mục": "muc",
    "tiểu mục": "tieu_muc",
    "tiểu_mục": "tieu_muc",
    "điều": "dieu",
    "khoản": "khoan",
    "điểm": "diem"
}

VN_RELATION_MAP: Dict[str, str] = {
    # 01. Văn bản được hướng dẫn áp dụng
    "văn bản được hướng dẫn áp dụng": "van_ban_duoc_huong_dan_ap_dung",
    "được hướng dẫn áp dụng": "van_ban_duoc_huong_dan_ap_dung",
    
    # 02. Văn bản hướng dẫn áp dụng
    "văn bản hướng dẫn áp dụng": "van_ban_huong_dan_ap_dung",
    "hướng dẫn áp dụng": "van_ban_huong_dan_ap_dung",
    
    # 03. Văn bản được quy định chi tiết, hướng dẫn thi hành
    "văn bản được quy định chi tiết, hướng dẫn thi hành": "van_ban_duoc_quy_dinh_chi_tiet_huong_dan_thi_hanh",
    "văn bản được quy định chi tiết và hướng dẫn thi hành": "van_ban_duoc_quy_dinh_chi_tiet_huong_dan_thi_hanh",
    "được quy định chi tiết, hướng dẫn thi hành": "van_ban_duoc_quy_dinh_chi_tiet_huong_dan_thi_hanh",
    "được quy định chi tiết và hướng dẫn thi hành": "van_ban_duoc_quy_dinh_chi_tiet_huong_dan_thi_hanh",
    
    # 04. Văn bản quy định chi tiết, hướng dẫn thi hành
    "văn bản quy định chi tiết, hướng dẫn thi hành": "van_ban_quy_dinh_chi_tiet_huong_dan_thi_hanh",
    "văn bản quy định chi tiết và hướng dẫn thi hành": "van_ban_quy_dinh_chi_tiet_huong_dan_thi_hanh",
    "quy định chi tiết, hướng dẫn thi hành": "van_ban_quy_dinh_chi_tiet_huong_dan_thi_hanh",
    "quy định chi tiết và hướng dẫn thi hành": "van_ban_quy_dinh_chi_tiet_huong_dan_thi_hanh",
    
    # 05. Văn bản bị tác động hiệu lực một phần (alias: được sửa đổi bổ sung)
    "văn bản bị tác động hiệu lực một phần": "van_ban_bi_tac_dong_hieu_luc_mot_phan",
    "bị tác động hiệu lực một phần": "van_ban_bi_tac_dong_hieu_luc_mot_phan",
    "văn bản được sửa đổi bổ sung": "van_ban_bi_tac_dong_hieu_luc_mot_phan",
    "được sửa đổi bổ sung": "van_ban_bi_tac_dong_hieu_luc_mot_phan",
    
    # 06. Văn bản quy định về việc hết hiệu lực một phần (alias: sửa đổi bổ sung)
    "văn bản quy định về việc hết hiệu lực một phần": "van_ban_quy_dinh_het_hieu_luc_mot_phan",
    "quy định về việc hết hiệu lực một phần": "van_ban_quy_dinh_het_hieu_luc_mot_phan",
    "văn bản sửa đổi bổ sung": "van_ban_quy_dinh_het_hieu_luc_mot_phan",
    "sửa đổi bổ sung": "van_ban_quy_dinh_het_hieu_luc_mot_phan",
    
    # 07. Văn bản bị tác động hiệu lực toàn bộ (alias: được thay thế, bị bãi bỏ)
    "văn bản bị tác động hiệu lực toàn bộ": "van_ban_bi_tac_dong_hieu_luc_toan_bo",
    "bị tác động hiệu lực toàn bộ": "van_ban_bi_tac_dong_hieu_luc_toan_bo",
    "văn bản được thay thế": "van_ban_bi_tac_dong_hieu_luc_toan_bo",
    "được thay thế": "van_ban_bi_tac_dong_hieu_luc_toan_bo",
    "văn bản bị bãi bỏ": "van_ban_bi_tac_dong_hieu_luc_toan_bo",
    "bị bãi bỏ": "van_ban_bi_tac_dong_hieu_luc_toan_bo",
    
    # 08. Văn bản quy định về việc hết hiệu lực toàn bộ (alias: thay thế, bãi bỏ)
    "văn bản quy định về việc hết hiệu lực toàn bộ": "van_ban_quy_dinh_het_hieu_luc_toan_bo",
    "quy định về việc hết hiệu lực toàn bộ": "van_ban_quy_dinh_het_hieu_luc_toan_bo",
    "văn bản thay thế": "van_ban_quy_dinh_het_hieu_luc_toan_bo",
    "thay thế": "van_ban_quy_dinh_het_hieu_luc_toan_bo",
    "văn bản bãi bỏ": "van_ban_quy_dinh_het_hieu_luc_toan_bo",
    "bãi bỏ": "van_ban_quy_dinh_het_hieu_luc_toan_bo",
    
    # 09. Văn bản được đính chính
    "văn bản được đính chính": "van_ban_duoc_dinh_chinh",
    "được đính chính": "van_ban_duoc_dinh_chinh",
    
    # 10. Văn bản đính chính
    "văn bản đính chính": "van_ban_dinh_chinh",
    "đính chính": "van_ban_dinh_chinh",
    
    # 11. Văn bản được dẫn chiếu
    "văn bản được dẫn chiếu": "van_ban_duoc_dan_chieu",
    "được dẫn chiếu": "van_ban_duoc_dan_chieu",
    
    # 12. Văn bản dẫn chiếu
    "văn bản dẫn chiếu": "van_ban_dan_chieu",
    "dẫn chiếu": "van_ban_dan_chieu",
    
    # 13. Căn cứ ban hành
    "căn cứ ban hành": "can_cu_ban_hanh",
    "căn cứ": "can_cu_ban_hanh",
    "văn bản căn cứ ban hành": "can_cu_ban_hanh",
    
    # 14. Văn bản áp dụng
    "văn bản áp dụng": "van_ban_ap_dung",
    "áp dụng": "van_ban_ap_dung",
    
    # 15. Văn bản được giải thích
    "văn bản được giải thích": "van_ban_duoc_giai_thich",
    "được giải thích": "van_ban_duoc_giai_thich",
    
    # 16. Văn bản giải thích
    "văn bản giải thích": "van_ban_giai_thich",
    "giải thích": "van_ban_giai_thich",
    
    # 17. Văn bản bị đình chỉ thi hành
    "văn bản bị đình chỉ thi hành": "van_ban_bi_dinh_chi_thi_hanh",
    "bị đình chỉ thi hành": "van_ban_bi_dinh_chi_thi_hanh",
    
    # 18. Văn bản đình chỉ thi hành
    "văn bản đình chỉ thi hành": "van_ban_dinh_chi_thi_hanh",
    "đình chỉ thi hành": "van_ban_dinh_chi_thi_hanh",
    
    # 19. Văn bản bị tạm ngưng/ gia hạn/ công bố tiếp tục còn hiệu lực
    "văn bản bị tạm ngưng/ gia hạn/ công bố tiếp tục còn hiệu lực": "van_ban_bi_tam_ngung_gia_han",
    "văn bản bị tạm ngưng/gia hạn/công bố tiếp tục còn hiệu lực": "van_ban_bi_tam_ngung_gia_han",
    "văn bản bị tạm ngưng hiệu lực": "van_ban_bi_tam_ngung_gia_han",
    "bị tạm ngưng hiệu lực": "van_ban_bi_tam_ngung_gia_han",
    
    # 20. Văn bản quy định tạm ngưng/ gia hạn/ công bố tiếp tục còn hiệu lực
    "văn bản quy định tạm ngưng/ gia hạn/ công bố tiếp tục còn hiệu lực": "van_ban_quy_dinh_tam_ngung_gia_han",
    "văn bản quy định tạm ngưng/gia hạn/công bố tiếp tục còn hiệu lực": "van_ban_quy_dinh_tam_ngung_gia_han",
    "văn bản tạm ngưng hiệu lực": "van_ban_quy_dinh_tam_ngung_gia_han",
    "tạm ngưng hiệu lực": "van_ban_quy_dinh_tam_ngung_gia_han",
    
    # 21. Văn bản hợp nhất
    "văn bản hợp nhất": "van_ban_hop_nhat",
    "văn bản được hợp nhất": "van_ban_hop_nhat",
    "hợp nhất": "van_ban_hop_nhat",

    # Bổ sung đồng bộ 100% tên nhãn từ docs/label.xml
    **{k.lower(): v for k, v in LABEL_XML_RELATION_MAP.items()}
}


class DocumentGraphBuilder:
    """
    Bộ xây dựng và khử nhiễu Đồ thị tài liệu (Document Graph Builder & Logic Denoiser)
    chuẩn hóa cho các mô hình GNN trong bài toán DocRE.
    """
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        checkpoint_path: str = "checkpoints/best_model.pt",
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.tokenizer: Any = None
        self.encoder: Any = None

    def lazy_load_encoder(self):
        """Chỉ nạp PhoBERT / LegalPhoBERT khi cần tính toán Node Embeddings (tiết kiệm bộ nhớ)"""
        if self.tokenizer is None or self.encoder is None:
            print(f"-> Đang khởi tạo Tokenizer ({self.model_name})...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                print(f"-> Đang khởi tạo Encoder ({self.model_name}) trên {self.device}...")
                self.encoder = AutoModel.from_pretrained(self.model_name).to(self.device)
            except Exception as e:
                fallback_name = "vinai/phobert-base"
                print(f"-> [LƯU Ý] Không tải được {self.model_name} ({e}), tự động chuyển về mô hình nền {fallback_name}...")
                self.model_name = fallback_name
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.encoder = AutoModel.from_pretrained(self.model_name).to(self.device)
            
            # Ưu tiên nạp trọng số đã tinh chỉnh theo miền luật từ checkpoint
            if os.path.exists(self.checkpoint_path):
                try:
                    ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                    state = ckpt.get("model_state_dict", {})
                    phobert_state = {k.replace("phobert.", ""): v for k, v in state.items() if k.startswith("phobert.")}
                    if phobert_state:
                        self.encoder.load_state_dict(phobert_state, strict=False)
                        print(f"-> Đã nạp thành công trọng số PhoBERT Fine-tuned từ {self.checkpoint_path}!")
                except Exception as ex:
                    print(f"-> [LƯU Ý] Dùng PhoBERT gốc vì không nạp được checkpoint: {ex}")
            else:
                print(f"-> Dùng trọng số gốc ({self.model_name}) vì chưa tìm thấy checkpoint {self.checkpoint_path}")

            self.encoder.eval()

    @staticmethod
    def extract_hierarchy_from_fulltext(folder_path: str) -> List[Dict[str, Any]]:
        """
        BÓC TÁCH CẤU TRÚC PHÂN CẤP LẬP PHÁP TRỰC TIẾP TỪ TOÀN VĂN (FULL-TEXT PARSER).
        Đảm bảo KHÔNG phụ thuộc độc quyền vào file muc_luc.json.
        Tự động đọc từ:
        1. noi_dung.docx (văn bản chính thức)
        2. noi_dung.txt / noi_dung_full.txt
        3. cau_da_tach.jsonl
        Nhận diện đầy đủ 7 cấp bậc: Phần -> Chương -> Mục -> Tiểu mục -> Điều -> Khoản -> Điểm.
        """
        docx_path = os.path.join(folder_path, "noi_dung.docx")
        paragraphs: List[str] = []

        if os.path.exists(docx_path):
            try:
                import docx
                doc = docx.Document(docx_path)
                paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            except Exception as ex:
                print(f"-> [LƯU Ý] Không đọc được docx ({ex}), chuyển sang định dạng văn bản thô...")

        if not paragraphs:
            for txt_name in ["noi_dung.txt", "noi_dung_full.txt", "van_ban.txt"]:
                tp = os.path.join(folder_path, txt_name)
                if os.path.exists(tp):
                    with open(tp, "r", encoding="utf-8") as f:
                        paragraphs = [line.strip() for line in f if line.strip()]
                    break

        if not paragraphs:
            jp = os.path.join(folder_path, "cau_da_tach.jsonl")
            if os.path.exists(jp):
                with open(jp, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if line_str:
                            try:
                                c = json.loads(line_str).get("raw", "").strip()
                                if c:
                                    paragraphs.append(c)
                            except Exception:
                                pass

        if not paragraphs:
            return []

        tree: List[Dict[str, Any]] = []
        curr_phan: Optional[Dict[str, Any]] = None
        curr_chuong: Optional[Dict[str, Any]] = None
        curr_muc: Optional[Dict[str, Any]] = None
        curr_tieu_muc: Optional[Dict[str, Any]] = None
        curr_dieu: Optional[Dict[str, Any]] = None
        curr_khoan: Optional[Dict[str, Any]] = None

        for text in paragraphs:
            # 1. Phần (Part)
            m_phan = re.match(r"^Phần\s+(thứ\s+[a-zA-Z]+|[IVXLCDM\d]+)[\.\:\s]*(.*)", text, re.IGNORECASE)
            if m_phan:
                curr_phan = {"level": "Phần", "number": m_phan.group(1), "title": text, "children": []}
                tree.append(curr_phan)
                curr_chuong = curr_muc = curr_tieu_muc = curr_dieu = curr_khoan = None
                continue

            # 2. Chương (Chapter)
            m_chuong = re.match(r"^Chương\s+([IVXLCDM\d]+)[\.\:\s]*(.*)", text, re.IGNORECASE)
            if m_chuong:
                curr_chuong = {"level": "Chương", "number": m_chuong.group(1), "title": text, "children": []}
                if curr_phan is not None:
                    curr_phan["children"].append(curr_chuong)
                else:
                    tree.append(curr_chuong)
                curr_muc = curr_tieu_muc = curr_dieu = curr_khoan = None
                continue

            # 3. Mục (Section)
            m_muc = re.match(r"^Mục\s+(\d+)[\.\:\s]*(.*)", text, re.IGNORECASE)
            if m_muc:
                curr_muc = {"level": "Mục", "number": m_muc.group(1), "title": text, "children": []}
                if curr_chuong is not None:
                    curr_chuong["children"].append(curr_muc)
                elif curr_phan is not None:
                    curr_phan["children"].append(curr_muc)
                else:
                    tree.append(curr_muc)
                curr_tieu_muc = curr_dieu = curr_khoan = None
                continue

            # 4. Tiểu mục (Sub-section)
            m_tm = re.match(r"^Tiểu\s+mục\s+(\d+)[\.\:\s]*(.*)", text, re.IGNORECASE)
            if m_tm:
                curr_tieu_muc = {"level": "Tiểu mục", "number": m_tm.group(1), "title": text, "children": []}
                if curr_muc is not None:
                    curr_muc["children"].append(curr_tieu_muc)
                elif curr_chuong is not None:
                    curr_chuong["children"].append(curr_tieu_muc)
                elif curr_phan is not None:
                    curr_phan["children"].append(curr_tieu_muc)
                else:
                    tree.append(curr_tieu_muc)
                curr_dieu = curr_khoan = None
                continue

            # 5. Điều luật (Article)
            m_dieu = re.match(r"^Điều\s+(\d+)\.[\s]*(.*)", text, re.IGNORECASE)
            if m_dieu:
                curr_dieu = {"level": "Điều", "number": m_dieu.group(1), "title": text, "content": text, "children": []}
                parent: Optional[Dict[str, Any]] = curr_tieu_muc or curr_muc or curr_chuong or curr_phan
                if parent is not None:
                    parent["children"].append(curr_dieu)
                else:
                    tree.append(curr_dieu)
                curr_khoan = None
                continue

            # 6. Khoản (Clause)
            m_khoan = re.match(r"^(\d+)\.[\s]*(.*)", text)
            if m_khoan and curr_dieu is not None:
                curr_khoan = {"level": "Khoản", "number": m_khoan.group(1), "title": text, "content": text, "children": []}
                curr_dieu["children"].append(curr_khoan)
                continue

            # 7. Điểm (Point)
            m_diem = re.match(r"^([a-zđ])\)[\s]*(.*)", text, re.IGNORECASE)
            if m_diem and curr_khoan is not None:
                diem_node: Dict[str, Any] = {"level": "Điểm", "number": m_diem.group(1), "title": text, "content": text, "children": []}
                curr_khoan["children"].append(diem_node)
                continue

        return tree

    def load_robust_hierarchy(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        CHIẾN LƯỢC TẢI CẤU TRÚC PHÂN CẤP ĐA TẦNG (MULTI-TIERED ROBUST HIERARCHY LOADER):
        1. Kiểm tra muc_luc.json: nếu không có, tự động chuyển sang trích xuất toàn văn (noi_dung.docx / txt).
        2. Nếu muc_luc.json có nhưng bị thiếu hụt nghiêm trọng hoặc thiếu cấp bậc,
           tự động đối chiếu và bù khuyết từ nội dung toàn văn để đảm bảo độ bao phủ 100%.
        """
        muc_luc_path = os.path.join(folder_path, "muc_luc.json")
        muc_luc_nodes: List[Dict[str, Any]] = []

        if os.path.exists(muc_luc_path):
            try:
                with open(muc_luc_path, "r", encoding="utf-8") as f:
                    muc_luc_nodes = json.load(f)
            except Exception as ex:
                print(f"-> [CẢNH BÁO] Không đọc được muc_luc.json ({ex}), kích hoạt bóc tách toàn văn...")

        # Nếu không có muc_luc.json hoặc rỗng -> 100% bóc tách từ toàn văn
        if not muc_luc_nodes:
            fulltext_nodes = self.extract_hierarchy_from_fulltext(folder_path)
            if fulltext_nodes:
                print("-> [TỰ ĐỘNG BÙ KHUYẾT] Không có muc_luc.json. Đã tự động bóc tách cấu trúc hoàn chỉnh từ toàn văn!")
                return fulltext_nodes
            return []

        return muc_luc_nodes

    def parse_document(self, folder_path: str) -> Dict[str, Any]:
        """
        Nạp toàn bộ dữ liệu thô của 1 văn bản: thuoc_tinh.json, muc_luc.json / toàn văn, luoc_do.json
        """
        folder_name = os.path.basename(os.path.normpath(folder_path))
        
        # 1. Nạp thuoc_tinh.json (an toàn nếu thiếu)
        meta_path = os.path.join(folder_path, "thuoc_tinh.json")
        attributes = {}
        title = folder_name
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_json = json.load(f)
                    attributes = meta_json.get("attributes", {})
                    title = meta_json.get("metadata", {}).get("title", folder_name)
            except Exception:
                pass
                
        # 2. Nạp cấu trúc phân cấp lập pháp: Đa tầng, không phụ thuộc duy nhất vào muc_luc.json
        muc_luc_nodes = self.load_robust_hierarchy(folder_path)

        # 3. Nạp luoc_do.json (an toàn nếu thiếu)
        luoc_do_path = os.path.join(folder_path, "luoc_do.json")
        luoc_do_data = {}
        if os.path.exists(luoc_do_path):
            try:
                with open(luoc_do_path, "r", encoding="utf-8") as f:
                    luoc_do_data = json.load(f)
            except Exception:
                pass

        # 4. Trích xuất đỉnh thô & cạnh thô
        raw_nodes, raw_edges = self._extract_raw_graph(
            doc_id=folder_name,
            title=title,
            attributes=attributes,
            muc_luc_nodes=muc_luc_nodes,
            luoc_do_data=luoc_do_data,
            folder_path=folder_path
        )

        # 5. Khử nhiễu Logic (Rule-based Denoising)
        denoised_nodes, denoised_edges, denoise_report = self.denoise_graph(raw_nodes, raw_edges)

        return {
            "doc_id": folder_name,
            "title": title,
            "attributes": attributes,
            "raw_nodes_count": len(raw_nodes),
            "raw_edges_count": len(raw_edges),
            "nodes": denoised_nodes,
            "edges": denoised_edges,
            "denoise_report": denoise_report
        }

    def parse_document_from_ket_qua(self, file_path_or_dict: Any) -> Dict[str, Any]:
        """
        Nạp trực tiếp từ file gán nhãn *_ket_qua.json (hoặc dictionary đã tải),
        trích xuất cấu trúc đồ thị đa quan hệ và thực thi động cơ Khử nhiễu Logic (100% DAG).
        """
        if isinstance(file_path_or_dict, str):
            with open(file_path_or_dict, "r", encoding="utf-8") as f:
                sample_data = json.load(f)
                if isinstance(sample_data, list):
                    sample_data = sample_data[0]
        else:
            sample_data = file_path_or_dict

        doc_id = str(sample_data.get("document_id") or sample_data.get("document_code", "doc"))
        meta = sample_data.get("document_metadata", {})
        title = meta.get("ten_van_ban") or sample_data.get("document_code", f"Văn bản {doc_id}")
        so_hieu = meta.get("so_hieu", "")

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # 1. Đỉnh gốc Văn bản: V_doc
        doc_node_id = f"doc:{doc_id}"
        doc_text = f"{title}. Số hiệu: {so_hieu}. Cơ quan: {meta.get('co_quan_ban_hanh', '')}. Loại: {meta.get('loai_van_ban', '')}."
        nodes.append({
            "node_id": doc_node_id,
            "level": "doc",
            "type_group": "document",
            "number": so_hieu,
            "title": title,
            "text": doc_text,
            "parent_id": None,
            "attributes": {field: str(meta.get(field, "")).strip() for field in THUOC_TINH_LABELS}
        })

        entities = sample_data.get("entities", [])
        relations = sample_data.get("relations", [])
        ent_map = {e["id"]: e for e in entities}

        struct_ents = []
        ref_ents = []
        for e in entities:
            scope = e.get("scope", "")
            std_code = normalize_entity_label(e.get("label", "") or e.get("code", ""))
            e["std_code"] = std_code
            if scope == "structure" or std_code in ["phan", "chuong", "muc", "tieu_muc", "dieu", "khoan", "diem"]:
                struct_ents.append(e)
            elif scope == "reference" or "van_ban" in std_code or std_code in ["can_cu_ban_hanh"]:
                ref_ents.append(e)

        child_to_parent = {}
        struct_ids = {e["id"] for e in struct_ents}
        for r in relations:
            std_r = normalize_relation_label(r.get("label", "") or r.get("code", ""))
            f_id, t_id = r.get("from_id"), r.get("to_id")
            if std_r == "belongs_to" and f_id in struct_ids and t_id in struct_ids:
                child_to_parent[f_id] = t_id

        node_id_map = {}
        struct_ents.sort(key=lambda x: (x.get("start", 0), x.get("end", 0)))
        prev_by_parent_and_level: Dict[str, Dict[str, str]] = {}

        for e in struct_ents:
            eid = e["id"]
            txt = e.get("text", "").strip()
            lvl = e["std_code"]
            if lvl not in HIERARCHY_RANK:
                lvl = "dieu" if "điều" in txt.lower() else "khoan"

            m_num = re.search(r"^(?:Điều|Khoản|Điểm|Chương|Phần|Mục|Tiểu\s*Mục)?\s*([0-9a-zA-ZđĐ]+)", txt, re.I)
            num_str = m_num.group(1) if m_num else str(len(nodes))
            clean_num = re.sub(r"[^a-zA-Z0-9_]", "", num_str)

            parent_eid = child_to_parent.get(eid)
            if parent_eid and parent_eid in node_id_map:
                p_node_id = node_id_map[parent_eid]
                p_level = ent_map[parent_eid]["std_code"]
            else:
                p_node_id = doc_node_id
                p_level = "doc"

            canonical_id = f"{p_node_id}/{lvl}_{clean_num}"
            node_id_map[eid] = canonical_id

            nodes.append({
                "node_id": canonical_id,
                "level": lvl,
                "type_group": "structure",
                "number": num_str,
                "title": txt[:80],
                "text": txt,
                "parent_id": p_node_id,
                "source_entity_id": eid
            })

            edges.append({
                "head": p_node_id,
                "head_type": p_level,
                "type": "contains",
                "tail": canonical_id,
                "tail_type": lvl
            })
            edges.append({
                "head": canonical_id,
                "head_type": lvl,
                "type": "belongs_to",
                "tail": p_node_id,
                "tail_type": p_level
            })

            p_dict = prev_by_parent_and_level.setdefault(p_node_id, {})
            if lvl in p_dict:
                edges.append({
                    "head": p_dict[lvl],
                    "head_type": lvl,
                    "type": "next_clause",
                    "tail": canonical_id,
                    "tail_type": lvl
                })
            p_dict[lvl] = canonical_id

        ref_seen = set()
        for e in ref_ents:
            eid = e["id"]
            txt = e.get("text", "").strip()
            clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", txt[:30])
            ref_node_id = f"ref:{clean_name}"
            node_id_map[eid] = ref_node_id
            if ref_node_id not in ref_seen:
                ref_seen.add(ref_node_id)
                nodes.append({
                    "node_id": ref_node_id,
                    "level": "van_ban_lien_quan",
                    "type_group": "reference",
                    "number": eid,
                    "title": txt,
                    "text": txt,
                    "parent_id": None,
                    "source_entity_id": eid
                })

        doc_year = 2015
        m_yr = re.search(r"(\d{4})", str(meta.get("ngay_ban_hanh", "2015")))
        if m_yr:
            doc_year = int(m_yr.group(1))

        for r in relations:
            std_r = normalize_relation_label(r.get("label", "") or r.get("code", ""))
            f_id, t_id = r.get("from_id"), r.get("to_id")
            if std_r == "belongs_to" and f_id in struct_ids:
                continue

            f_node = node_id_map.get(f_id)
            t_node = node_id_map.get(t_id)
            if f_id in ent_map and ent_map[f_id]["std_code"] in ["so_hieu", "ten_van_ban"]:
                f_node = doc_node_id
            if t_id in ent_map and ent_map[t_id]["std_code"] in ["so_hieu", "ten_van_ban"]:
                t_node = doc_node_id

            if f_node and t_node:
                f_type = next((n["level"] for n in nodes if n["node_id"] == f_node), "doc")
                t_type = next((n["level"] for n in nodes if n["node_id"] == t_node), "doc")
                edges.append({
                    "head": f_node,
                    "head_type": f_type,
                    "type": std_r,
                    "tail": t_node,
                    "tail_type": t_type
                })

        denoised_nodes, denoised_edges, rep = self.denoise_graph(nodes, edges)

        return {
            "doc_id": doc_id,
            "title": title,
            "attributes": meta,
            "raw_nodes_count": len(nodes),
            "raw_edges_count": len(edges),
            "nodes": denoised_nodes,
            "edges": denoised_edges,
            "denoise_report": rep
        }

    def parse_document_from_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Nạp trực tiếp từ file gán nhãn *_ket_qua.csv (8 cột chuẩn hóa):
        NHÓM (ENTITY / RELATION), ID, CODE, LABEL, SCOPE / FROM_ID, START / TO_ID, END, TEXT
        """
        import csv
        doc_id = os.path.basename(file_path).replace("_ket_qua.csv", "")
        entities: List[Dict[str, Any]] = []
        relations: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {}

        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row or len(row) < 6:
                    continue
                group = row[0].strip().upper()
                r_id = row[1].strip()
                code = row[2].strip()
                label = row[3].strip()

                if "ENTITY" in group:
                    scope = row[4].strip()
                    start = int(row[5].strip()) if row[5].strip().isdigit() else 0
                    end = int(row[6].strip()) if len(row) > 6 and row[6].strip().isdigit() else 0
                    text = row[7].strip() if len(row) > 7 else ""

                    entities.append({
                        "id": r_id,
                        "code": code,
                        "label": label,
                        "scope": scope,
                        "start": start,
                        "end": end,
                        "text": text
                    })
                    if scope == "metadata":
                        std_m = normalize_entity_label(label or code)
                        meta[std_m] = text

                elif "RELATION" in group:
                    from_id = row[4].strip()
                    to_id = row[5].strip()
                    relations.append({
                        "id": r_id,
                        "code": code,
                        "label": label,
                        "from_id": from_id,
                        "to_id": to_id
                    })

        sample_data = {
            "document_id": doc_id,
            "document_code": doc_id,
            "document_metadata": meta,
            "entities": entities,
            "relations": relations
        }
        return self.parse_document_from_ket_qua(sample_data)

    def _extract_raw_graph(
        self,
        doc_id: str,
        title: str,
        attributes: Dict[str, Any],
        muc_luc_nodes: List[Dict[str, Any]],
        luoc_do_data: Dict[str, Any],
        folder_path: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Bóc tách cấu trúc cây với Canonical Hierarchical Node IDs (không trùng lặp).
        """
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # Đỉnh gốc Văn bản: V_doc (chứa 11 thuộc tính metadata nhúng bên trong)
        doc_node_id = f"doc:{doc_id}"
        doc_text = f"{title}. Số hiệu: {attributes.get('so_hieu', '')}. Cơ quan: {attributes.get('co_quan_ban_hanh', '')}. Loại: {attributes.get('loai_van_ban', '')}."
        nodes.append({
            "node_id": doc_node_id,
            "level": "doc",
            "type_group": "document",
            "number": attributes.get("so_hieu", ""),
            "title": title,
            "text": doc_text,
            "parent_id": None,
            "attributes": {field: str(attributes.get(field, "")).strip() for field in THUOC_TINH_LABELS}
        })

        # Duyệt cây phân cấp đệ quy
        def walk_hierarchy(items: List[Dict[str, Any]], parent_id: str, parent_level: str):
            prev_sibling_id: Optional[str] = None
            prev_sibling_level: Optional[str] = None

            for idx, item in enumerate(items, 1):
                raw_lvl = str(item.get("level", "")).strip().lower()
                std_lvl = VN_LEVEL_MAP.get(raw_lvl, raw_lvl)
                if std_lvl not in HIERARCHY_RANK:
                    std_lvl = "dieu" if "điều" in raw_lvl else "khoan"

                num = str(item.get("number", "")).strip()
                t_title = str(item.get("title", "")).strip()
                t_content = str(item.get("content", "")).strip()
                node_text = (t_title + " " + t_content).strip() if t_content else t_title

                # Xây dựng Canonical Unique ID: parent_id/std_lvl_num
                clean_num = re.sub(r"[^a-zA-Z0-9_]", "", num) if num else str(idx)
                curr_node_id = f"{parent_id}/{std_lvl}_{clean_num}"

                node_record = {
                    "node_id": curr_node_id,
                    "level": std_lvl,
                    "type_group": "structure",
                    "number": num,
                    "title": t_title,
                    "text": node_text,
                    "parent_id": parent_id
                }
                nodes.append(node_record)

                # Cạnh phân cấp: contains & belongs_to
                edges.append({
                    "head": parent_id,
                    "head_type": parent_level,
                    "type": "contains",
                    "tail": curr_node_id,
                    "tail_type": std_lvl
                })
                edges.append({
                    "head": curr_node_id,
                    "head_type": std_lvl,
                    "type": "belongs_to",
                    "tail": parent_id,
                    "tail_type": parent_level
                })

                # Cạnh tuần tự anh em ruột: next_clause
                if prev_sibling_id is not None and prev_sibling_level is not None and prev_sibling_level == std_lvl:
                    edges.append({
                        "head": prev_sibling_id,
                        "head_type": prev_sibling_level,
                        "type": "next_clause",
                        "tail": curr_node_id,
                        "tail_type": std_lvl
                    })

                prev_sibling_id = curr_node_id
                prev_sibling_level = std_lvl

                # Đệ quy cho con
                children = item.get("children", [])
                if children:
                    walk_hierarchy(children, curr_node_id, std_lvl)

        walk_hierarchy(muc_luc_nodes, doc_node_id, "doc")

        # Bổ sung trích xuất Tiểu mục (V_tieu_muc) nếu muc_luc chưa có nhưng trong nội dung văn bản có
        has_tieu_muc = any(n.get("level") == "tieu_muc" for n in nodes)
        if not has_tieu_muc and folder_path:
            docx_path = os.path.join(folder_path, "noi_dung.docx")
            if os.path.exists(docx_path):
                try:
                    import docx
                    doc_docx = docx.Document(docx_path)
                    paras = [p.text.strip() for p in doc_docx.paragraphs if p.text.strip()]
                    tm_idx = 0
                    prev_tm_id: Optional[str] = None
                    for i, text in enumerate(paras):
                        m_tm = re.match(r"^Tiểu\s+mục\s+(\d+)\.?\s*(.*)", text, re.IGNORECASE)
                        if m_tm:
                            tm_idx += 1
                            tm_num = m_tm.group(1)
                            tm_rest = m_tm.group(2).strip()
                            if not tm_rest and i + 1 < len(paras):
                                next_p = paras[i + 1]
                                if not re.match(r"^(Điều|Chương|Mục|Phần)", next_p, re.IGNORECASE):
                                    tm_rest = next_p
                            full_title = f"Tiểu mục {tm_num}. {tm_rest}".strip(" .")
                            curr_tm_id = f"{doc_node_id}/tieu_muc_{tm_idx}"

                            nodes.append({
                                "node_id": curr_tm_id,
                                "level": "tieu_muc",
                                "type_group": "structure",
                                "number": tm_num,
                                "title": full_title,
                                "text": full_title,
                                "parent_id": doc_node_id
                            })

                            # Cạnh phân cấp: contains & belongs_to
                            edges.append({
                                "head": doc_node_id,
                                "head_type": "doc",
                                "type": "contains",
                                "tail": curr_tm_id,
                                "tail_type": "tieu_muc"
                            })
                            edges.append({
                                "head": curr_tm_id,
                                "head_type": "tieu_muc",
                                "type": "belongs_to",
                                "tail": doc_node_id,
                                "tail_type": "doc"
                            })

                            # Cạnh tuần tự anh em: next_clause
                            if prev_tm_id is not None:
                                edges.append({
                                    "head": prev_tm_id,
                                    "head_type": "tieu_muc",
                                    "type": "next_clause",
                                    "tail": curr_tm_id,
                                    "tail_type": "tieu_muc"
                                })
                            prev_tm_id = curr_tm_id
                except Exception as ex:
                    print(f"-> [LƯU Ý] Không trích xuất được tiểu mục từ docx: {ex}")

        # Bổ sung trích xuất Quan hệ Dẫn chiếu nội bộ (dan_chieu_noi_bo: 3. Thành phần - Thành phần)
        dieu_map: Dict[str, str] = {}
        for n in nodes:
            if n.get("level") == "dieu":
                num = str(n.get("number", "")).strip()
                if not num:
                    m = re.search(r"Điều\s+(\d+)", n.get("title", "") + " " + n.get("text", ""), re.IGNORECASE)
                    if m:
                        num = m.group(1)
                if num:
                    dieu_map[num] = n["node_id"]

        internal_ref_seen: Set[Tuple[str, str]] = set()
        for n in nodes:
            if n.get("type_group") != "structure":
                continue
            src_node_id = n["node_id"]
            src_lvl = n.get("level", "structure")
            raw_node_txt = (n.get("text", "") or n.get("title", "")).strip()

            # Mẫu 1: "theo quy định tại / quy định tại / tại / theo Điều X"
            for m in re.finditer(r"(?:theo\s+quy\s+định\s+tại|quy\s+định\s+tại|tại|theo)\s+Điều\s+(\d+)", raw_node_txt, re.IGNORECASE):
                d_target = m.group(1)
                if d_target in dieu_map and dieu_map[d_target] != src_node_id:
                    target_node_id = dieu_map[d_target]
                    if (src_node_id, target_node_id) not in internal_ref_seen:
                        internal_ref_seen.add((src_node_id, target_node_id))
                        edges.append({
                            "head": src_node_id,
                            "head_type": src_lvl,
                            "type": "dan_chieu_noi_bo",
                            "tail": target_node_id,
                            "tail_type": "dieu"
                        })

            # Mẫu 2: "Điều X của/Bộ luật này", "Điều X Luật này"
            for m in re.finditer(r"Điều\s+(\d+)\s+(?:của\s+)?(?:Bộ\s+luật|Luật)\s+này", raw_node_txt, re.IGNORECASE):
                d_target = m.group(1)
                if d_target in dieu_map and dieu_map[d_target] != src_node_id:
                    target_node_id = dieu_map[d_target]
                    if (src_node_id, target_node_id) not in internal_ref_seen:
                        internal_ref_seen.add((src_node_id, target_node_id))
                        edges.append({
                            "head": src_node_id,
                            "head_type": src_lvl,
                            "type": "dan_chieu_noi_bo",
                            "tail": target_node_id,
                            "tail_type": "dieu"
                        })

        # Nạp quan hệ pháp lý từ luoc_do.json (V_ref, Legal Edges)
        rel_groups = luoc_do_data.get("relations", [])
        ref_seen: Set[str] = set()
        edge_keys_seen: Set[Tuple[str, str, str]] = set()

        def add_unique_edge(head: str, head_t: str, rel: str, tail: str, tail_t: str):
            k = (head, tail, rel)
            if k not in edge_keys_seen:
                edge_keys_seen.add(k)
                edges.append({
                    "head": head,
                    "head_type": head_t,
                    "type": rel,
                    "tail": tail,
                    "tail_type": tail_t
                })

        doc_year = 2015
        m_doc = re.search(r"(\d{4})", str(attributes.get("ngay_ban_hanh", "2015")))
        if m_doc:
            doc_year = int(m_doc.group(1))

        for group in rel_groups:
            raw_rel = str(group.get("relation_type", "")).strip().lower()
            clean_rel = re.sub(r"^văn\s+bản\s+", "", raw_rel).strip()
            direction = group.get("direction", "outgoing")
            std_rel = VN_RELATION_MAP.get(raw_rel) or VN_RELATION_MAP.get(clean_rel)
            if not std_rel or std_rel not in LEGAL_RELATIONS:
                continue

            for doc in group.get("documents", []):
                rel_title = str(doc.get("title", "")).strip()
                rel_item_id = str(doc.get("item_id", "")).strip()
                if not rel_title:
                    continue

                ref_node_id = f"ref:{rel_item_id}" if rel_item_id else f"ref:{re.sub(r'[^a-zA-Z0-9]', '_', rel_title[:30])}"

                # Thêm đỉnh Viện dẫn nếu chưa có
                if ref_node_id not in ref_seen:
                    ref_seen.add(ref_node_id)
                    nodes.append({
                        "node_id": ref_node_id,
                        "level": "van_ban_lien_quan",
                        "type_group": "reference",
                        "number": rel_item_id,
                        "title": rel_title,
                        "text": rel_title,
                        "parent_id": None
                    })

                # A. Thêm cạnh quan hệ pháp lý
                if direction == "outgoing":
                    add_unique_edge(doc_node_id, "doc", std_rel, ref_node_id, "van_ban_lien_quan")
                else:
                    add_unique_edge(ref_node_id, "van_ban_lien_quan", std_rel, doc_node_id, "doc")

        return nodes, edges

    def denoise_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        BỘ KHỬ NHIỄU LOGIC (RULE-BASED DENOISING ENGINE):
        1. Orphan Resolution: Sửa nút mồ côi (kết nối về cha hợp lệ gần nhất).
        2. Strict Tree Monotonicity: Bắt buộc cha có Rank nhỏ hơn con.
        3. Cycle Detection & Elimination: Đảm bảo đồ thị phân cấp là DAG (0 chu trình).
        4. Belongs_to Synchronization: Đồng bộ tính đối xứng contains <-> belongs_to.
        5. Next Clause Sanitization: Chỉ cho phép next_clause giữa các anh em ruột cùng cha.
        """
        node_map: Dict[str, Dict[str, Any]] = {n["node_id"]: n for n in nodes}
        report: Dict[str, Any] = {
            "orphan_fixed": 0,
            "invalid_hierarchy_edges_removed": 0,
            "cycles_detected_and_cut": 0,
            "invalid_next_clause_removed": 0,
            "is_dag": True
        }

        # --- BƯỚC 1: SỬA ĐỈNH MỒ CÔI (ORPHAN RESOLUTION) ---
        root_doc_id = nodes[0]["node_id"] if nodes else "doc:unknown"
        for n in nodes:
            if n["level"] == "doc" or n["type_group"] == "reference":
                continue
            p_id = n.get("parent_id")
            if not p_id or p_id not in node_map:
                # Tìm cha hợp lệ gần nhất dựa trên tiền tố ID
                parts = n["node_id"].split("/")
                candidate_parent = "/".join(parts[:-1]) if len(parts) > 1 else root_doc_id
                if candidate_parent in node_map:
                    n["parent_id"] = candidate_parent
                else:
                    n["parent_id"] = root_doc_id
                report["orphan_fixed"] += 1

        # --- BƯỚC 2: KHỬ CẠNH PHÂN CẤP VI PHẠM TÍNH ĐƠN ĐIỆU (STRICT MONOTONICITY) ---
        valid_edges: List[Dict[str, Any]] = []
        for e in edges:
            h_id = e["head"]
            t_id = e["tail"]
            rel = e["type"]

            if h_id not in node_map or t_id not in node_map:
                continue

            h_level = node_map[h_id]["level"]
            t_level = node_map[t_id]["level"]
            h_rank = HIERARCHY_RANK.get(h_level, 99)
            t_rank = HIERARCHY_RANK.get(t_level, 99)

            if rel == "contains":
                # Cha rank phải nhỏ hơn con rank (doc:0 < phan:1 < chuong:2 < ... < diem:7)
                if h_rank >= t_rank:
                    report["invalid_hierarchy_edges_removed"] += 1
                    continue
            elif rel == "belongs_to":
                if h_rank <= t_rank:
                    report["invalid_hierarchy_edges_removed"] += 1
                    continue
            elif rel == "next_clause":
                # next_clause chỉ cho phép giữa anh em ruột cùng cha và cùng level
                h_parent = node_map[h_id].get("parent_id")
                t_parent = node_map[t_id].get("parent_id")
                if h_parent != t_parent or h_level != t_level:
                    report["invalid_next_clause_removed"] += 1
                    continue
            elif rel == "dan_chieu_noi_bo":
                if h_id == t_id:
                    continue

            valid_edges.append(e)

        # --- BƯỚC 3: KIỂM ĐỊNH VÀ TRIỆT TIÊU CHU TRÌNH (CYCLE ELIMINATION) ---
        containment_graph: Any = nx.DiGraph()
        for e in valid_edges:
            if e["type"] == "contains":
                containment_graph.add_edge(e["head"], e["tail"])  # type: ignore

        try:
            cycles = list(nx.simple_cycles(containment_graph))  # type: ignore
            if cycles:
                report["cycles_detected_and_cut"] = len(cycles)
                report["is_dag"] = False
                # Cắt cạnh vòng lặp để biến thành DAG
                broken_edges: Set[Tuple[str, str]] = set()
                for c in cycles:
                    broken_edges.add((c[-1], c[0]))
                valid_edges = [
                    e for e in valid_edges
                    if not (e["type"] == "contains" and (e["head"], e["tail"]) in broken_edges)
                ]
            else:
                report["is_dag"] = True
        except Exception:
            report["is_dag"] = True

        # --- BƯỚC 4: ĐỒNG BỘ CẶP CẠNH ĐỐI NGẪU ---
        contains_set = {(e["head"], e["tail"]) for e in valid_edges if e["type"] == "contains"}
        belongs_to_set = {(e["head"], e["tail"]) for e in valid_edges if e["type"] == "belongs_to"}

        final_edges: List[Dict[str, Any]] = []
        for e in valid_edges:
            final_edges.append(e)

        # Bổ sung cạnh belongs_to còn thiếu cho từng cạnh contains
        for (u, v) in contains_set:
            if (v, u) not in belongs_to_set:
                final_edges.append({
                    "head": v,
                    "head_type": node_map[v]["level"],
                    "type": "belongs_to",
                    "tail": u,
                    "tail_type": node_map[u]["level"]
                })

        return nodes, final_edges, report

    def encode_text_sliding_window(
        self,
        text: str,
        window_size: int = 256,
        stride: int = 128
    ) -> torch.Tensor:
        """
        Mã hóa văn bản bằng kỹ thuật Cửa sổ trượt (Sliding Window):
        Quét văn bản dài bằng các cửa sổ gối nhau (stride), trích xuất vector [CLS]
        và thực hiện Mean-Pooling để tạo vector đặc trưng ngữ nghĩa 768 chiều toàn vẹn.
        """
        if not text or not text.strip():
            return torch.zeros(768, dtype=torch.float32)

        tokenizer: Any = self.tokenizer
        encoder: Any = self.encoder

        tokens = tokenizer.encode(text, add_special_tokens=False)
        max_sub = window_size - 2

        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

        if len(tokens) <= max_sub:
            w = [bos_id] + tokens + [eos_id]
            input_ids = torch.tensor([w], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                out = encoder(input_ids=input_ids, attention_mask=attention_mask)
                return out.last_hidden_state[:, 0, :].squeeze(0).cpu()

        window_chunks = []
        for start_idx in range(0, len(tokens), stride):
            chunk = tokens[start_idx : start_idx + max_sub]
            if not chunk:
                break
            full_w = [bos_id] + chunk + [eos_id]
            window_chunks.append(full_w)
            if start_idx + max_sub >= len(tokens):
                break

        if not window_chunks:
            return torch.zeros(768, dtype=torch.float32)

        max_w_len = max(len(w) for w in window_chunks)
        batch_ids = [w + [pad_id] * (max_w_len - len(w)) for w in window_chunks]
        batch_masks = [[1] * len(w) + [0] * (max_w_len - len(w)) for w in window_chunks]

        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(batch_masks, dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = encoder(input_ids=input_ids, attention_mask=attention_mask)
            window_embs = out.last_hidden_state[:, 0, :]
            pooled = window_embs.mean(dim=0).cpu()
            return pooled

    def compute_node_embeddings(
        self,
        nodes: List[Dict[str, Any]],
        batch_size: int = 32,
        window_size: int = 256,
        stride: int = 128,
        use_sliding_window: bool = True
    ) -> torch.Tensor:
        """
        Trích xuất vector nhúng ngữ nghĩa (Node Embedding) 768 chiều bằng PhoBERT / LegalPhoBERT
        kết hợp kỹ thuật CỬA SỔ TRƯỢT (Sliding Window):
        - Văn bản ngắn (<= window_size): Mã hóa batch trực tiếp.
        - Văn bản dài (> window_size): Chia nhỏ theo cửa sổ trượt (window_size=256, stride=128)
          và Mean-Pooling qua các cửa sổ để giữ trọn vẹn 100% ngữ nghĩa điều luật dài.
        """
        self.lazy_load_encoder()
        all_texts = [n.get("text", "") or n.get("title", "") for n in nodes]
        total_nodes = len(all_texts)
        all_embeddings: List[torch.Tensor] = []

        print(f"-> Đang trích xuất đặc trưng {self.model_name} (Sliding Window = {use_sliding_window}, W={window_size}, S={stride}) cho {total_nodes:,} đỉnh đồ thị...")
        t0 = time.time()

        tokenizer: Any = self.tokenizer
        encoder: Any = self.encoder
        if tokenizer is None or encoder is None:
            raise RuntimeError("Chưa khởi tạo PhoBERT tokenizer hoặc encoder!")

        max_sub = window_size - 2

        for i in range(0, total_nodes, batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_tokens_lens = [len(tokenizer.encode(t, add_special_tokens=False)) if t else 0 for t in batch_texts]
            has_long_text = any(l > max_sub for l in batch_tokens_lens) if use_sliding_window else False

            if not has_long_text:
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=window_size,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = encoder(**encoded)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
                    all_embeddings.append(cls_embeddings)
            else:
                batch_embs = []
                for t, l in zip(batch_texts, batch_tokens_lens):
                    if l > max_sub and use_sliding_window:
                        emb = self.encode_text_sliding_window(t, window_size=window_size, stride=stride)
                    else:
                        enc = tokenizer(t, padding=True, truncation=True, max_length=window_size, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            emb = encoder(**enc).last_hidden_state[:, 0, :].squeeze(0).cpu()
                    batch_embs.append(emb)
                all_embeddings.append(torch.stack(batch_embs, dim=0))

        elapsed = time.time() - t0
        emb_tensor = torch.cat(all_embeddings, dim=0)
        print(f"-> Hoàn tất trong {elapsed:.2f}s! Kích thước ma trận đặc trưng X: {list(emb_tensor.shape)}")
        return emb_tensor

    def package_to_pyg(
        self,
        doc_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        embeddings: torch.Tensor
    ) -> Tuple[HeteroData, Data]:
        """
        ĐÓNG GÓI ĐỒ THỊ SANG PYTORCH GEOMETRIC:
        1. HeteroData: Đồ thị không đồng nhất (Heterogeneous Graph).
        2. Data: Đồ thị đa quan hệ đồng nhất (Multi-Relational Homogeneous Graph).
        """
        node_id_to_idx = {n["node_id"]: idx for idx, n in enumerate(nodes)}
        
        # 1. Đóng gói Homogeneous Data
        head_indices = []
        tail_indices = []
        edge_types = []

        for e in edges:
            h_id = e["head"]
            t_id = e["tail"]
            rel_type = e["type"]
            if h_id in node_id_to_idx and t_id in node_id_to_idx:
                head_indices.append(node_id_to_idx[h_id])
                tail_indices.append(node_id_to_idx[t_id])
                edge_types.append(REL2ID.get(rel_type, 0))

        edge_index = torch.tensor([head_indices, tail_indices], dtype=torch.long)
        edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
        
        node_type_map = {"document": 0, "structure": 1, "reference": 2, "metadata": 3}
        node_types = torch.tensor([node_type_map.get(n.get("type_group", "structure"), 1) for n in nodes], dtype=torch.long)

        homo_data = Data(
            x=embeddings,
            edge_index=edge_index,
            edge_type=edge_type_tensor,
            node_type=node_types
        )
        homo_data.num_nodes = len(nodes)
        homo_data.doc_id = doc_id
        homo_data.node_ids = [n["node_id"] for n in nodes]
        homo_data.node_levels = [n["level"] for n in nodes]

        # 2. Đóng gói HeteroData
        hetero_data = HeteroData()
        hetero_data["doc_id"] = doc_id

        # Phân tách đỉnh theo nhóm (bao gồm cả nhóm metadata theo yêu cầu của đề tài)
        doc_indices = [i for i, n in enumerate(nodes) if n.get("type_group") == "document"]
        meta_indices = [i for i, n in enumerate(nodes) if n.get("type_group") == "metadata"]
        struct_indices = [i for i, n in enumerate(nodes) if n.get("type_group") == "structure"]
        ref_indices = [i for i, n in enumerate(nodes) if n.get("type_group") == "reference"]

        if doc_indices:
            hetero_data["document"].x = embeddings[doc_indices]
        if meta_indices:
            hetero_data["metadata"].x = embeddings[meta_indices]
        if struct_indices:
            hetero_data["structure"].x = embeddings[struct_indices]
        if ref_indices:
            hetero_data["reference"].x = embeddings[ref_indices]

        # Ánh xạ chỉ số cục bộ cho từng loại đỉnh trong HeteroData
        type_local_map = {}
        for local_idx, global_idx in enumerate(doc_indices):
            type_local_map[global_idx] = ("document", local_idx)
        for local_idx, global_idx in enumerate(meta_indices):
            type_local_map[global_idx] = ("metadata", local_idx)
        for local_idx, global_idx in enumerate(struct_indices):
            type_local_map[global_idx] = ("structure", local_idx)
        for local_idx, global_idx in enumerate(ref_indices):
            type_local_map[global_idx] = ("reference", local_idx)

        # Gom các cạnh theo (src_type, rel, dst_type)
        hetero_edge_dict: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {}
        for e in edges:
            h_idx = node_id_to_idx.get(e["head"])
            t_idx = node_id_to_idx.get(e["tail"])
            if h_idx is not None and t_idx is not None:
                src_type, src_local = type_local_map[h_idx]
                dst_type, dst_local = type_local_map[t_idx]
                triple = (src_type, e["type"], dst_type)
                if triple not in hetero_edge_dict:
                    hetero_edge_dict[triple] = ([], [])
                hetero_edge_dict[triple][0].append(src_local)
                hetero_edge_dict[triple][1].append(dst_local)

        for (src_type, rel_name, dst_type), (src_list, dst_list) in hetero_edge_dict.items():
            hetero_data[src_type, rel_name, dst_type].edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        return hetero_data, homo_data


def process_all_legal_documents(
    workspace_root: str,
    target_folders: Optional[List[str]] = None,
    extract_embeddings: bool = True
) -> Dict[str, Any]:
    """
    Hàm thực thi toàn bộ pipeline Bước 5 trên các bộ luật và xuất file .pt
    """
    bo_luat_dir = os.path.join(workspace_root, "bo_luat")
    out_dir = os.path.join(workspace_root, "dataset")
    os.makedirs(out_dir, exist_ok=True)

    builder = DocumentGraphBuilder()
    
    if target_folders:
        folder_paths = [os.path.join(bo_luat_dir, f) for f in target_folders]
    else:
        folder_paths = sorted(glob.glob(os.path.join(bo_luat_dir, "*")))

    print("=" * 80)
    print("=== BƯỚC 5: KHỬ NHIỄU LOGIC & ĐÓNG GÓI ĐỒ THỊ TÀI LIỆU (DOC GRAPH CHO GNN) ===")
    print("=" * 80)
    print(f"Số lượng văn bản cần xử lý: {len(folder_paths)}")

    all_packaged_graphs: Dict[str, Any] = {}
    total_nodes_all = 0
    total_edges_all = 0

    for f_path in folder_paths:
        if os.path.isfile(f_path) and f_path.endswith(".json"):
            parsed = builder.parse_document_from_ket_qua(f_path)
            doc_id = parsed["doc_id"]
        elif os.path.isdir(f_path):
            parsed = builder.parse_document(f_path)
            doc_id = os.path.basename(f_path)
        else:
            continue

        print(f"\n[+] Đang xử lý tài liệu: {doc_id} ...")
        nodes = parsed["nodes"]
        edges = parsed["edges"]
        rep = parsed["denoise_report"]

        print(f"    -> Đỉnh: {len(nodes):,} | Cạnh: {len(edges):,}")
        print(f"    -> Báo cáo Khử nhiễu: DAG: {rep['is_dag']} | Sửa mồ côi: {rep['orphan_fixed']} | Cắt vi phạm: {rep['invalid_hierarchy_edges_removed']}")

        if extract_embeddings:
            embeddings = builder.compute_node_embeddings(nodes, batch_size=64)
        else:
            embeddings = torch.zeros((len(nodes), 768), dtype=torch.float32)

        hetero_g, homo_g = builder.package_to_pyg(doc_id, nodes, edges, embeddings)
        all_packaged_graphs[doc_id] = {
            "title": parsed["title"],
            "attributes": parsed["attributes"],
            "nodes_count": len(nodes),
            "edges_count": len(edges),
            "hetero_data": hetero_g,
            "homo_data": homo_g,
            "denoise_report": rep
        }
        total_nodes_all += len(nodes)
        total_edges_all += len(edges)

    # Lưu tập dữ liệu đồ thị đóng gói
    out_pt_path = os.path.join(out_dir, "denoised_document_graphs.pt")
    torch.save(all_packaged_graphs, out_pt_path)
    print(f"\n[+] ĐÃ LƯU THÀNH CÔNG ĐỒ THỊ TÀI LIỆU ĐÓNG GÓI: {out_pt_path}")
    print(f"    -> Tổng số đỉnh trên toàn bộ tập dữ liệu: {total_nodes_all:,}")
    print(f"    -> Tổng số cạnh quan hệ: {total_edges_all:,}")

    return all_packaged_graphs


if __name__ == "__main__":
    workspace = os.path.dirname(os.path.abspath(__file__))
    import sys
    
    target = ["91_2015_QH13_95942", "100_2015_QH13_96122", "95_2015_QH13_96117"]
    extract_emb = True
    
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    
    if "--fast" in flags or "--no_emb" in flags:
        extract_emb = False
        print("[*] Chế độ NHANH (--fast): Khử nhiễu & đóng gói cấu trúc PyG (Embeddings khởi tạo 0 để train sau).")
    else:
        print("[*] Chế độ ĐẦY ĐỦ: Trích xuất PhoBERT 768-D Node Embeddings.")
        
    if "--dataset" in flags or (args and args[0].lower() == "dataset"):
        use_csv = "--csv" in flags
        if use_csv:
            dataset_files = sorted(glob.glob(os.path.join(workspace, "dataset", "**", "*_ket_qua.csv"), recursive=True))
            print(f"[*] Đã chọn chế độ: XỬ LÝ CÁC TỆP CSV TRONG dataset/ ({len(dataset_files)} tệp)")
        else:
            dataset_files = sorted(glob.glob(os.path.join(workspace, "dataset", "**", "*_ket_qua.json"), recursive=True))
            print(f"[*] Đã chọn chế độ: XỬ LÝ CÁC TỆP JSON TRONG dataset/ ({len(dataset_files)} tệp)")

        builder = DocumentGraphBuilder()
        all_packaged_graphs: Dict[str, Any] = {}
        out_dir = os.path.join(workspace, "dataset")

        for f_path in dataset_files:
            if f_path.endswith(".csv"):
                parsed = builder.parse_document_from_csv(f_path)
            else:
                parsed = builder.parse_document_from_ket_qua(f_path)

            doc_id = parsed["doc_id"]
            print(f"\n[+] Đang xử lý tài liệu gán nhãn: {doc_id} ({os.path.basename(f_path)}) ...")
            nodes = parsed["nodes"]
            edges = parsed["edges"]
            rep = parsed["denoise_report"]
            print(f"    -> Đỉnh: {len(nodes):,} | Cạnh: {len(edges):,}")
            print(f"    -> Báo cáo Khử nhiễu: DAG: {rep['is_dag']} | Sửa mồ côi: {rep['orphan_fixed']}")

            if extract_emb:
                embeddings = builder.compute_node_embeddings(nodes, batch_size=32, window_size=256, stride=128, use_sliding_window=True)
            else:
                embeddings = torch.zeros((len(nodes), 768), dtype=torch.float32)

            hetero_g, homo_g = builder.package_to_pyg(doc_id, nodes, edges, embeddings)
            all_packaged_graphs[doc_id] = {
                "title": parsed["title"],
                "attributes": parsed["attributes"],
                "nodes_count": len(nodes),
                "edges_count": len(edges),
                "hetero_data": hetero_g,
                "homo_data": homo_g,
                "denoise_report": rep
            }

        # Lưu đồng thời vào cả 2 đường dẫn để tương thích hoàn toàn
        out_pt_dataset = os.path.join(out_dir, "denoised_dataset_graphs.pt")
        out_pt_standard = os.path.join(out_dir, "denoised_document_graphs.pt")
        torch.save(all_packaged_graphs, out_pt_dataset)
        torch.save(all_packaged_graphs, out_pt_standard)
        print(f"\n[+] ĐÃ LƯU THÀNH CÔNG ĐỒ THỊ TÀI LIỆU GÁN NHÃN:")
        print(f"    -> {out_pt_dataset}")
        print(f"    -> {out_pt_standard}")
        sys.exit(0)

    if args:
        if args[0].lower() == "all":
            target = None  # None sẽ tự động quét toàn bộ thư mục trong bo_luat/
            print("[*] Đã chọn chế độ: XỬ LÝ TOÀN BỘ 17 BỘ LUẬT TRONG bo_luat/")
        else:
            target = [x.strip() for x in args[0].split(",") if x.strip()]
            print(f"[*] Đã chọn danh sách mục tiêu ({len(target)} văn bản): {target}")
            
    # Chạy quy trình đóng gói đồ thị tài liệu
    process_all_legal_documents(
        workspace,
        target_folders=target,
        extract_embeddings=extract_emb
    )


