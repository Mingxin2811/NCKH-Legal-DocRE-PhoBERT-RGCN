# -*- coding: utf-8 -*-
"""
predict.py
Module suy luận (Inference) cho mô hình LegalPhoBERT_CRF:
1. Nhận chuỗi văn bản pháp lý tiếng Việt bất kỳ.
2. Tách từ tiếng Việt chuẩn bằng pyvi (ViTokenizer).
3. Đưa qua LegalPhoBERT_CRF để giải mã Viterbi chuỗi nhãn tối ưu.
4. Trích xuất danh sách thực thể có cấu trúc: Tên thực thể, Loại thực thể, Vị trí từ.
"""

import os
import sys
try:
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    getattr(sys.stderr, "reconfigure")(encoding="utf-8")
except Exception:
    pass

import re
import torch
from pyvi import ViTokenizer  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from typing import List, Dict, Any, Optional

from model import LegalPhoBERT_CRF
from labels import ID2LABEL, LABEL2ID, BIO_LABELS, ALL_ENTITIES, ALL_RELATIONS


class LegalNERPredictor:
    """
    Công cụ suy luận thực thể pháp lý (Legal NER Predictor)
    sử dụng mô hình đã huấn luyện LegalPhoBERT_CRF.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        model_name: str = "vinai/phobert-base",
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Đang nạp PhoBERT Tokenizer ({model_name})...")
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.bos_token: str = str(getattr(self.tokenizer, "bos_token", "<s>") or "<s>")
        self.eos_token: str = str(getattr(self.tokenizer, "eos_token", "</s>") or "</s>")

        print(f"Đang nạp mô hình LegalPhoBERT_CRF từ {checkpoint_path}...")
        self.model: Any = LegalPhoBERT_CRF(
            model_name=model_name,
            num_labels=len(BIO_LABELS)
        ).to(self.device)

        if os.path.exists(checkpoint_path):
            checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_compatible_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint.get("epoch", "?")
            f1 = float(checkpoint.get("val_f1", 0.0)) * 100
            print(f"-> Đã nạp thành công Checkpoint Epoch {epoch} (Val F1: {f1:.2f}%)")
        else:
            print(f"-> [CẢNH BÁO] Không tìm thấy file {checkpoint_path}. Mô hình đang dùng trọng số ngẫu nhiên!")

        self.model.eval()

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Dự đoán thực thể cho một chuỗi văn bản tiếng Việt.
        Returns:
            Dict chứa:
              - raw_text: Văn bản gốc
              - words: Danh sách các từ đã tách
              - tags: Danh sách nhãn BIO của từng từ
              - entities: Danh sách thực thể được gom cụm logic
        """
        text = text.strip()
        if not text:
            return {"raw_text": "", "words": [], "tags": [], "entities": []}

        # 1. Tách từ tiếng Việt bằng pyvi
        segmented = ViTokenizer.tokenize(text)
        words = segmented.split()

        # 2. Xây dựng subwords và word_indices
        subwords: List[str] = [self.bos_token]
        word_indices: List[int] = []
        valid_words: List[str] = []

        for w in words:
            w_subwords: List[str] = self.tokenizer.tokenize(w)  # type: ignore
            if not w_subwords:
                continue
            if len(subwords) + len(w_subwords) >= 255:
                break
            word_indices.append(len(subwords))
            valid_words.append(w)
            subwords.extend(w_subwords)

        subwords.append(self.eos_token)
        subword_ids: List[int] = self.tokenizer.convert_tokens_to_ids(subwords)  # type: ignore
        input_ids = torch.tensor([subword_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([[1] * len(subwords)], dtype=torch.long, device=self.device)
        word_indices_tensor = torch.tensor([word_indices], dtype=torch.long, device=self.device)
        word_mask = torch.tensor([[True] * len(word_indices)], dtype=torch.bool, device=self.device)

        # 3. Suy luận qua mô hình
        with torch.no_grad():
            _, predictions = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_indices=word_indices_tensor,
                word_mask=word_mask,
                labels=None
            )

        pred_ids = predictions[0]
        tags: List[str] = [ID2LABEL.get(pid, "o") for pid in pred_ids]

        # 3.1. Hậu xử lý ngữ pháp văn bản pháp luật (Legal Grammar Constraints):
        # a. Xử lý cấp mục lục: Phần, Chương, Mục, Tiểu mục, Điều, Khoản, Điểm
        for i in range(len(valid_words) - 1):
            curr_tag = tags[i]
            next_w = valid_words[i + 1]
            if curr_tag in ["b-phan", "i-phan"] and next_w.lower() in ["thứ", "nhất", "hai", "ba", "tư", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười", "i", "ii", "iii", "iv", "v"]:
                tags[i + 1] = "i-phan"
            elif curr_tag == "b-dieu" and re.match(r"^(\d+[a-zA-Z]?|[IVXLCDM]+)$", next_w):
                tags[i + 1] = "i-dieu"
            elif curr_tag == "b-khoan" and re.match(r"^\d+$", next_w):
                tags[i + 1] = "i-khoan"
            elif curr_tag == "b-chuong" and re.match(r"^(\d+|[IVXLCDM]+)$", next_w, re.IGNORECASE):
                tags[i + 1] = "i-chuong"
            elif curr_tag == "b-muc" and re.match(r"^(\d+|[IVXLCDM]+)$", next_w, re.IGNORECASE):
                tags[i + 1] = "i-muc"
            elif curr_tag == "b-tieu_muc" and re.match(r"^(\d+|[IVXLCDM]+)$", next_w, re.IGNORECASE):
                tags[i + 1] = "i-tieu_muc"
            elif curr_tag == "b-diem" and re.match(r"^[a-zđ]\)?$", next_w, re.IGNORECASE):
                tags[i + 1] = "i-diem"

        # b. Nhận diện chuẩn xác thực thể Tiểu mục (kể cả khi PyVi tách 'Tiểu_mục' hoặc 'Tiểu' + 'mục')
        for i in range(len(valid_words)):
            w_lower = valid_words[i].lower()
            if w_lower == "tiểu_mục":
                tags[i] = "b-tieu_muc"
                if i + 1 < len(valid_words) and re.match(r"^(\d+|[IVXLCDM]+)$", valid_words[i + 1], re.IGNORECASE):
                    tags[i + 1] = "i-tieu_muc"
            elif w_lower == "tiểu" and i + 1 < len(valid_words) and valid_words[i + 1].lower() == "mục":
                tags[i] = "b-tieu_muc"
                tags[i + 1] = "i-tieu_muc"
                if i + 2 < len(valid_words) and re.match(r"^(\d+|[IVXLCDM]+)$", valid_words[i + 2], re.IGNORECASE):
                    tags[i + 2] = "i-tieu_muc"

        # c. Khử nhầm lẫn: Số La Mã sau chữ 'khóa'/'khoá' thuộc Quốc hội, không phải cấp Chương
        for i in range(len(valid_words) - 1):
            if valid_words[i].lower() in ["khóa", "khoá"] and tags[i + 1].endswith("-chuong"):
                tags[i + 1] = "o"

        # 3.2. Bổ trợ nhận diện thực thể viện dẫn pháp lý chuẩn (Statutory Citation Patterns):
        # a. Loại văn bản quy phạm pháp luật
        LOAI_VB_SET = {"Bộ_luật", "Luật", "Nghị_định", "Thông_tư", "Nghị_quyết", "Quyết_định", "Pháp_lệnh", "Hiến_pháp", "Chỉ_thị"}
        for i, w in enumerate(valid_words):
            if w in LOAI_VB_SET and tags[i] == "o":
                tags[i] = "b-loai_van_ban"

        # b. Cơ quan ban hành nhà nước
        CO_QUAN_SET = {"Quốc_hội", "Chính_phủ", "Chủ_tịch_nước", "Thủ_tướng_Chính_phủ", "Bộ_Tư_pháp", "Bộ_Công_an", "Bộ_Tài_chính", "Tòa_án_nhân_dân_tối_cao", "Viện_kiểm_sát_nhân_dân_tối_cao"}
        for i, w in enumerate(valid_words):
            if w in CO_QUAN_SET and tags[i] == "o":
                tags[i] = "b-co_quan_ban_hanh"

        # c. Số hiệu văn bản pháp lý (Ví dụ: 100/2015/QH13, 59/2020/QH14, 01/2021/NĐ-CP)
        for i in range(len(valid_words) - 4):
            if (re.match(r"^\d+$", valid_words[i]) and valid_words[i+1] == "/" and
                re.match(r"^\d{4}$", valid_words[i+2]) and valid_words[i+3] == "/" and
                re.match(r"^[A-Z0-9\-]+$", valid_words[i+4])):
                tags[i] = "b-so_hieu"
                for k in range(i + 1, i + 5):
                    tags[k] = "i-so_hieu"

        # d. Ngày ban hành / Ngày có hiệu lực / Ngày hết hiệu lực
        for i in range(len(valid_words) - 4):
            context_before = " ".join(valid_words[max(0, i - 6):i]).lower()
            if "hết" in context_before and "hiệu" in context_before:
                date_tag = "ngay_het_hieu_luc"
            elif "hiệu" in context_before:
                date_tag = "ngay_co_hieu_luc"
            else:
                date_tag = "ngay_ban_hanh"

            # Định dạng: ngày DD/MM/YYYY
            if (valid_words[i].lower() == "ngày" and
                re.match(r"^\d{1,2}$", valid_words[i+1]) and valid_words[i+2] == "/" and
                re.match(r"^\d{1,2}$", valid_words[i+3]) and valid_words[i+4] == "/" and
                i + 5 < len(valid_words) and re.match(r"^\d{4}$", valid_words[i+5])):
                tags[i] = f"b-{date_tag}"
                for k in range(i + 1, i + 6):
                    tags[k] = f"i-{date_tag}"
            # Định dạng: ngày DD tháng MM năm YYYY
            elif (valid_words[i].lower() == "ngày" and
                  re.match(r"^\d{1,2}$", valid_words[i+1]) and valid_words[i+2].lower() == "tháng" and
                  re.match(r"^\d{1,2}$", valid_words[i+3]) and valid_words[i+4].lower() == "năm" and
                  i + 5 < len(valid_words) and re.match(r"^\d{4}$", valid_words[i+5])):
                tags[i] = f"b-{date_tag}"
                for k in range(i + 1, i + 6):
                    tags[k] = f"i-{date_tag}"

        # 4. Gom cụm các nhãn BIO thành thực thể hoàn chỉnh
        entities: List[Dict[str, Any]] = []
        current_entity: Optional[Dict[str, Any]] = None

        for w, tag in zip(valid_words, tags):
            clean_word = w.replace("_", " ")

            if tag.startswith("b-"):
                if current_entity is not None:
                    current_entity["text"] = re.sub(r"\s*/\s*", "/", str(current_entity["text"]))
                    entities.append(current_entity)
                ent_type = tag[2:]
                current_entity = {
                    "text": clean_word,
                    "type": ent_type,
                    "words": [w]
                }
            elif tag.startswith("i-"):
                ent_type = tag[2:]
                if current_entity is not None and current_entity.get("type") == ent_type:
                    current_entity["text"] = str(current_entity["text"]) + " " + clean_word
                    current_entity["words"].append(w)
                else:
                    if current_entity is not None:
                        current_entity["text"] = re.sub(r"\s*/\s*", "/", str(current_entity["text"]))
                        entities.append(current_entity)
                    current_entity = {
                        "text": clean_word,
                        "type": ent_type,
                        "words": [w]
                    }
            else:
                if current_entity is not None:
                    current_entity["text"] = re.sub(r"\s*/\s*", "/", str(current_entity["text"]))
                    entities.append(current_entity)
                    current_entity = None

        if current_entity is not None:
            current_entity["text"] = re.sub(r"\s*/\s*", "/", str(current_entity["text"]))
            entities.append(current_entity)

        # 5. Xây dựng entities_manifest chuẩn hóa (Closed-World Schema Completeness)
        # Đảm bảo 100% các thực thể trong ALL_ENTITIES đều có mặt trong từ điển;
        # Thực thể nào xuất hiện thì lưu danh sách text, thực thể nào vắng mặt thì để None (để trống).
        entities_manifest: Dict[str, Any] = {ent: None for ent in ALL_ENTITIES}
        for ent in entities:
            e_type = ent["type"]
            e_text = ent["text"]
            if e_type in entities_manifest:
                if entities_manifest[e_type] is None:
                    entities_manifest[e_type] = []
                if e_text not in entities_manifest[e_type]:
                    entities_manifest[e_type].append(e_text)

        # 6. Xây dựng relations_manifest chuẩn hóa (Closed-World Schema Completeness)
        # Đảm bảo 100% các loại quan hệ trong ALL_RELATIONS đều có mặt;
        # Quan hệ nào phát hiện được giữa các thực thể thì lưu danh sách, không có thì để None (null / no_relation).
        relations_manifest: Dict[str, Any] = {rel: None for rel in ALL_RELATIONS}

        def add_rel(r_type: str, head_txt: str, tail_txt: str):
            if r_type in relations_manifest:
                if relations_manifest[r_type] is None:
                    relations_manifest[r_type] = []
                relations_manifest[r_type].append({"head": head_txt, "tail": tail_txt})

        # Suy luận một số quan hệ nội tại giữa các thực thể vừa phát hiện trong câu
        ent_by_type: Dict[str, List[str]] = {}
        for ent in entities:
            ent_by_type.setdefault(ent["type"], []).append(ent["text"])

        # a. Quan hệ cấu trúc contains / belongs_to giữa các cấp
        hierarchy_chain = ["phan", "chuong", "muc", "tieu_muc", "dieu", "khoan", "diem"]
        for idx in range(len(hierarchy_chain) - 1):
            parent_lvl = hierarchy_chain[idx]
            child_lvl = hierarchy_chain[idx + 1]
            if parent_lvl in ent_by_type and child_lvl in ent_by_type:
                for p_text in ent_by_type[parent_lvl]:
                    for c_text in ent_by_type[child_lvl]:
                        add_rel("contains", p_text, c_text)
                        add_rel("belongs_to", c_text, p_text)

        # b. Thuộc tính văn bản (metadata) được lưu trữ trực tiếp trong thuộc tính của văn bản (không tạo node/cạnh has_attribute và attribute_of)

        # c. Dẫn chiếu căn cứ ban hành / văn bản liên quan
        if "van_ban_lien_quan" in ent_by_type:
            for ref_doc in ent_by_type["van_ban_lien_quan"]:
                if "loai_van_ban" in ent_by_type:
                    add_rel("can_cu_ban_hanh", ent_by_type["loai_van_ban"][0], ref_doc)

        return {
            "raw_text": text,
            "words": valid_words,
            "tags": tags,
            "entities": entities,
            "entities_manifest": entities_manifest,
            "relations_manifest": relations_manifest
        }


def format_prediction_result(result: Dict[str, Any], show_manifest: bool = True) -> str:
    """Định dạng kết quả dự đoán thành bảng biểu trực quan kèm Manifest chuẩn hóa"""
    lines: List[str] = []
    lines.append(f"Văn bản: \"{result['raw_text']}\"")
    lines.append("-" * 70)
    if not result["entities"]:
        lines.append("  (Không tìm thấy thực thể nào)")
    else:
        lines.append(f"Tìm thấy {len(result['entities'])} thực thể được trích xuất:")
        for idx, ent in enumerate(result["entities"], 1):
            lines.append(f"  [{idx:02d}] {ent['text']} -> <{ent['type']}>")
    lines.append("-" * 70)

    if show_manifest and "entities_manifest" in result:
        lines.append("BẢNG SCHEMA THỰC THỂ MANIFEST CHUẨN HÓA (Đầy đủ 19 thực thể):")
        for ent_name, val in result["entities_manifest"].items():
            if val:
                val_str = ", ".join(val)
                lines.append(f"  - {ent_name:<24}: {val_str}")
            else:
                lines.append(f"  - {ent_name:<24}: [TRỐNG / NULL / NO_ENTITY]")
        lines.append("-" * 70)

    if show_manifest and "relations_manifest" in result:
        lines.append(f"BẢNG SCHEMA QUAN HỆ MANIFEST CHUẨN HÓA (Đầy đủ {len(result['relations_manifest'])} quan hệ):")
        for r_name, r_list in result["relations_manifest"].items():
            if r_list:
                r_str = ", ".join([f"({item['head']}) -> ({item['tail']})" for item in r_list])
                lines.append(f"  - {r_name:<50}: {r_str}")
            else:
                lines.append(f"  - {r_name:<50}: [TRỐNG / NULL / NO_RELATION]")
        lines.append("-" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    try:
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Chạy suy luận thực thể pháp lý với LegalPhoBERT_CRF")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="File checkpoint mô hình")
    parser.add_argument("--text", type=str, default=None, help="Câu văn bản pháp lý cần trích xuất")
    parser.add_argument("--interactive", action="store_true", help="Chế độ nhập tương tác trực tiếp từ bàn phím")

    args = parser.parse_args()

    predictor = LegalNERPredictor(checkpoint_path=args.checkpoint)

    sample_texts = [
        "Căn cứ Luật Doanh nghiệp số 59/2020/QH14 ban hành ngày 17 tháng 06 năm 2020 của Quốc hội.",
        "Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng của Bộ luật này.",
        "Chủ tịch Quốc hội Nguyễn Thị Kim Ngân đã ký chứng thực văn bản này.",
        "Nghị định số 100/2015/NĐ-CP hết hiệu lực thi hành kể từ ngày 01 tháng 01 năm 2022."
    ]

    if args.text:
        res = predictor.predict(args.text)
        print(format_prediction_result(res))
    elif args.interactive:
        print("\n=== CHẾ ĐỘ SUY LUẬN TƯƠNG TÁC (Gõ 'exit' hoặc 'quit' để thoát) ===")
        while True:
            try:
                user_input = input("\nNhập câu pháp lý: ").strip()
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                if not user_input:
                    continue
                res = predictor.predict(user_input)
                print(format_prediction_result(res))
            except (KeyboardInterrupt, EOFError):
                break
    else:
        print("\n=== CHẠY KIỂM THỬ TRÊN CÁC CÂU PHÁP LÝ MẪU ===")
        for t in sample_texts:
            res = predictor.predict(t)
            print(format_prediction_result(res))
