# -*- coding: utf-8 -*-


import os
import re
import unicodedata

from pyvi import ViTokenizer  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from labels import LABEL2ID, ID2LABEL


def read_docx(docx_path: str) -> str:
    """
    Đọc toàn bộ văn bản từ file .docx bao gồm cả các đoạn văn và bảng biểu (bảng chữ ký ở cuối).
    Duyệt tuần tự theo đúng cấu trúc tài liệu để không bỏ sót thông tin người ký và chức danh.
    """
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"Không tìm thấy file: {docx_path}")
    
    try:
        from docx import Document
        from docx.text.paragraph import Paragraph
        from docx.table import Table
        
        doc = Document(docx_path)
        lines = []
        for element in doc.element.body:
            if element.tag.endswith("p"):
                p = Paragraph(element, doc)
                text = p.text.strip()
                if text:
                    lines.append(text)
            elif element.tag.endswith("tbl"):
                table = Table(element, doc)
                for row in table.rows:
                    for cell in row.cells:
                        for p in cell.paragraphs:
                            text = p.text.strip()
                            # Tránh trùng lặp do merged cells
                            if text and (not lines or text != lines[-1]):
                                lines.append(text)
        return "\n".join(lines)
    except Exception:
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(docx_path) as z:
            tree = ET.fromstring(z.read("word/document.xml"))
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for p in tree.iterfind(".//w:p", ns):
                texts = [node.text for node in p.iterfind(".//w:t", ns) if node.text]
                if texts:
                    clean_str = "".join(texts).strip()
                    if clean_str and (not paragraphs or clean_str != paragraphs[-1]):
                        paragraphs.append(clean_str)
            return "\n".join(paragraphs)


def normalize_text(text: str) -> str:

    if not text:
        return ""
    
    # 1. Ép về chuẩn Unicode NFC
    text = unicodedata.normalize("NFC", text)
    
    # 2. Xử lý các khoảng trắng đặc biệt và xuống dòng
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    
    # 3. Chuẩn hóa dấu gạch ngang và dấu nháy
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)
    
    return text.strip()


def segment_vietnamese_words(text: str) -> str:

    if not text:
        return ""
    return ViTokenizer.tokenize(text)


def split_into_sentences(text: str, max_words: int = 100) -> list[str]:
    """
    Tách văn bản thành các đơn vị câu/khoản pháp lý hoàn chỉnh.
    Ưu tiên giữ nguyên vẹn từng đoạn/khoản (paragraph) trong văn bản luật.
    Chỉ chia nhỏ nếu đoạn văn vượt quá max_words (để tránh tràn cửa sổ 256 tokens của PhoBERT).
    Không cắt ngang sau số hiệu điều/khoản (ví dụ: 'Điều 1.', '1. ', 'a) ').
    """
    raw_lines = text.split("\n")
    sentences = []
    
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
            
        words = line.split()
        if len(words) <= max_words:
            sentences.append(line)
        else:
            # Đoạn văn quá dài (> max_words), cắt theo dấu chấm/chấm phẩy
            parts = re.split(r"(?<=[.;])\s+", line)
            merged_parts = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # Nếu phần liền trước là số khoản ('1.', '2.') hoặc điều ('Điều 1.') hoặc từ viết tắt thì gộp lại
                if merged_parts and (
                    re.match(r"^(\d+|[a-zđ]\))$", merged_parts[-1].rstrip(".")) or
                    re.match(r"^Điều\s+\d+[a-zA-Z]?\.?$", merged_parts[-1], re.IGNORECASE) or
                    (len(merged_parts[-1].split()) <= 2 and merged_parts[-1].endswith("."))
                ):
                    merged_parts[-1] = merged_parts[-1] + " " + p
                else:
                    merged_parts.append(p)
                    
            # Đảm bảo các khối sau gộp không vượt quá max_words
            buffer = []
            for mp in merged_parts:
                mp_words = mp.split()
                if len(" ".join(buffer).split()) + len(mp_words) <= max_words:
                    buffer.append(mp)
                else:
                    if buffer:
                        sentences.append(" ".join(buffer).strip())
                    buffer = [mp]
            if buffer:
                sentences.append(" ".join(buffer).strip())
                
    return sentences


def align_subwords_with_labels(words: list[str], word_labels: list[str], tokenizer, max_length: int = 256):

    bos = tokenizer.bos_token or "<s>"
    eos = tokenizer.eos_token or "</s>"
    
    subwords = [bos]
    aligned_labels = [-100]
    
    for word, label in zip(words, word_labels):
        w_subwords = tokenizer.tokenize(word)
        if not w_subwords:
            continue
        for i, sub in enumerate(w_subwords):
            subwords.append(sub)
            if i == 0:
                aligned_labels.append(LABEL2ID.get(label, 0) if isinstance(label, str) else label)
            else:
                aligned_labels.append(-100)
                
    # Cắt gọn theo max_length để nhường chỗ cho token kết thúc
    if len(subwords) >= max_length:
        subwords = subwords[:max_length - 1]
        aligned_labels = aligned_labels[:max_length - 1]
        
    subwords.append(eos)
    aligned_labels.append(-100)
    
    input_ids = tokenizer.convert_tokens_to_ids(subwords)
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "subwords": subwords,
        "labels": aligned_labels
    }


def preprocess_document_sample(docx_path: str, tokenizer):

    raw_content = read_docx(docx_path)
    clean_content = normalize_text(raw_content)
    sentences = split_into_sentences(clean_content)
    
    processed_samples = []
    for sent in sentences[:10]: # Lấy 10 câu đầu để demo
        segmented_sent = segment_vietnamese_words(sent)
        words = segmented_sent.split()
        # Giả lập nhãn 'o' ban đầu nếu chưa có nhãn
        dummy_labels = ["o"] * len(words)
        
        aligned_data = align_subwords_with_labels(words, dummy_labels, tokenizer)
        processed_samples.append({
            "raw_sentence": sent,
            "segmented_sentence": segmented_sent,
            "words": words,
            "subwords": aligned_data["subwords"],
            "input_ids": aligned_data["input_ids"],
            "aligned_labels": aligned_data["labels"]
        })
        
    return processed_samples


def export_preprocessed_sentences(sentences: list[str], output_jsonl: str):

    import json
    
    with open(output_jsonl, "w", encoding="utf-8") as f_jsonl:
        for idx, sent in enumerate(sentences, 1):
            seg_sent = segment_vietnamese_words(sent)
            words = seg_sent.split()
            
            record = {
                "id": idx,
                "raw": sent,
                "segmented": seg_sent,
                "word_count": len(words)
            }
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import sys
    import glob
    try:
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    except Exception:
        pass
    
    print("=" * 60)
    print("=== KIỂM THỬ BƯỚC 1: TIỀN XỬ LÝ VĂN BẢN TRÊN FILE .DOCX THỰC TẾ ===")
    print("=" * 60)
    
    # 1. Xác định file .docx cần test (nhận từ tham số dòng lệnh hoặc mặc định 100_2015_QH13_96122)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        target_docx = sys.argv[1]
    else:
        target_docx = "bo_luat/100_2015_QH13_96122/noi_dung.docx"
        
    if not os.path.exists(target_docx):
        sample_files = glob.glob("bo_luat/*/noi_dung.docx")
        if sample_files:
            target_docx = sample_files[0]
        else:
            print(f"Không tìm thấy file văn bản tại: {target_docx}")
            sys.exit(1)
            
    print(f"\n[1] Đang đọc file văn bản: {target_docx}")
    raw_content = read_docx(target_docx)
    print(f"-> Tổng độ dài văn bản: {len(raw_content):,} ký tự")
    
    # 2. Chuẩn hóa văn bản
    print("\n[2] Đang chuẩn hóa Unicode NFC và làm sạch khoảng trắng...")
    clean_content = normalize_text(raw_content)
    
    # 3. Tách câu / đoạn
    sentences = split_into_sentences(clean_content, max_words=100)
    print(f"-> Tổng số câu/khoản được tách: {len(sentences):,} câu")
    
    # 4. Tách từ tiếng Việt trên các câu mẫu
    print("\n[3] Minh họa tách từ tiếng Việt (pyvi) trên 3 câu mở đầu:")
    for idx, sent in enumerate(sentences[:3], 1):
        seg_sent = segment_vietnamese_words(sent)
        print(f"\n--- Câu {idx} ---")
        print(f"Gốc       : {sent}")
        print(f"Tách từ   : {seg_sent}")
        
    # Minh họa thêm 1 câu điều luật cụ thể (ví dụ Điều 1)
    dieu1_sample = None
    dieu1_idx = None
    for idx, s in enumerate(sentences):
        if s.startswith("Điều 1.") or s.startswith("Điều 1 "):
            dieu1_sample = s
            dieu1_idx = idx + 1
            break
            
    if dieu1_sample:
        seg_dieu1 = segment_vietnamese_words(dieu1_sample)
        print(f"\n--- Câu {dieu1_idx} (Mẫu Điều luật) ---")
        print(f"Gốc       : {dieu1_sample}")
        print(f"Tách từ   : {seg_dieu1}")
        
    # 5. Tokenize với PhoBERT
    print("\n[4] Kiểm tra Tokenize Subword với PhoBERT:")
    try:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        sample_words = segment_vietnamese_words(sentences[0]).split()
        dummy_labels = ["o"] * len(sample_words)
        
        aligned = align_subwords_with_labels(sample_words, dummy_labels, tokenizer)
        print(f"-> Tokenizer nạp thành công: vinai/phobert-base")
        print(f"-> Số từ gốc (words)    : {len(sample_words)}")
        print(f"-> Số subwords (PhoBERT): {len(aligned['subwords'])}")
        print(f"-> Danh sách subwords   : {aligned['subwords'][:15]} ...")
        print(f"-> Nhãn căn chỉnh (-100): {aligned['labels'][:15]} ...")
    except Exception as e:
        print(f"-> Lưu ý về PhoBERT Tokenizer: {e}")
        
    # 6. Tự động xuất toàn bộ 100% các câu ra file JSONL chuẩn cho mô hình
    doc_dir = os.path.dirname(target_docx) or "."
    out_jsonl = os.path.join(doc_dir, "cau_da_tach.jsonl")
    
    print(f"\n[5] Đang xuất toàn bộ {len(sentences):,} câu ra file JSONL chuẩn...")
    export_preprocessed_sentences(sentences, out_jsonl)
    print(f"-> Đã xuất file JSONL (dùng cho huấn luyện & Bước 2): {out_jsonl}")
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH XỬ LÝ VÀ XUẤT ĐẦY ĐỦ 100% CÂU TRONG VĂN BẢN!")
    print("=" * 60)
