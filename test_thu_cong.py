# -*- coding: utf-8 -*-
"""
test_thu_cong.py
Công cụ Kiểm thử Thủ công (Manual Inference & Interactive Testing)
Mô hình Lai PhoBERT + RGCN (LegalRGCN) trên Trọng số best_model/best_model_rgcn.pt
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Thiết lập encoding utf-8 cho console Windows
try:
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    getattr(sys.stderr, "reconfigure")(encoding="utf-8")
except Exception:
    pass

# 22 Quan hệ pháp lý & cấu trúc thực tế (đã loại bỏ 4 quan hệ thời gian suy diễn)
ACTIVE_RELATIONS = [
    "THUOC",
    "CAN_CU_BAN_HANH",
    "HUONG_DAN_AP_DUNG",
    "DUOC_HUONG_DAN_AP_DUNG",
    "QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH",
    "QUY_DINH_HET_HIEU_LUC_MOT_PHAN",
    "BI_TAC_DONG_HIEU_LUC_MOT_PHAN",
    "QUY_DINH_HET_HIEU_LUC_TOAN_BO",
    "BI_TAC_DONG_HIEU_LUC_TOAN_BO",
    "DINH_CHINH",
    "DUOC_DINH_CHINH",
    "DAN_CHIEU",
    "DUOC_DAN_CHIEU",
    "AP_DUNG",
    "GIAI_THICH",
    "DUOC_GIAI_THICH",
    "HOP_NHAT",
    "DINH_CHI_THI_HANH",
    "BI_DINH_CHI_THI_HANH",
    "QUY_DINH_TAM_NGUNG_GIA_HAN",
    "BI_TAM_NGUNG_GIA_HAN"
]

try:
    from labels import REL_VN_NAMES
except ImportError:
    REL_VN_NAMES = {}

NUM_RELATIONS = 22
NUM_RELATION_CLASSES = 23
NO_RELATION_LABEL = 22

# ==============================================================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH KHỚP VỚI CHECKPOINT
# ==============================================================================
try:
    from torch_geometric.nn import RGCNConv
except ImportError:
    RGCNConv = None

class GatedFeatureFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GatedFeatureFusion, self).__init__()
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_sem: torch.Tensor, h_topo: torch.Tensor):
        concat = torch.cat([h_sem, h_topo], dim=-1)
        gate = self.sigmoid(self.gate_linear(concat))
        h_fused = gate * h_sem + (1.0 - gate) * h_topo
        return h_fused, gate

class LegalRGCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 128,
        num_relations: int = 22,
        num_bases: int = 8,
        num_rgcn_layers: int = 2,
        dropout: float = 0.2
    ):
        super(LegalRGCN, self).__init__()
        self.node_proj = nn.Linear(in_channels, hidden_channels)
        if RGCNConv is not None:
            self.rgcn_layers = nn.ModuleList([
                RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations, num_bases=num_bases)
                for _ in range(num_rgcn_layers)
            ])
        else:
            self.rgcn_layers = nn.ModuleList([])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_rgcn_layers)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fusion = GatedFeatureFusion(hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, NUM_RELATION_CLASSES)
        )
        self.head_legal = self.classifier

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, pair_edges: torch.Tensor):
        h0 = self.relu(self.node_proj(x))
        h = h0
        if edge_index.numel() > 0 and len(self.rgcn_layers) > 0:
            for conv, norm in zip(self.rgcn_layers, self.norms):
                h_next = conv(h, edge_index, edge_type)
                h = norm(self.relu(h_next))
                h = self.dropout(h)
        h_fused, _ = self.fusion(h0, h)

        if pair_edges.numel() == 0:
            return torch.empty((0, NUM_RELATION_CLASSES), device=x.device), h_fused

        u_emb = h_fused[pair_edges[0]]
        v_emb = h_fused[pair_edges[1]]
        pair_repr = torch.cat([u_emb, v_emb, torch.abs(u_emb - v_emb), u_emb * v_emb], dim=-1)
        logits = self.classifier(pair_repr)
        return logits, h_fused


# ==============================================================================
# 2. HÀM TRÍCH XUẤT ĐẶC TRƯNG TỪ PHOBERT BẰNG CỬA SỔ TRƯỢT
# ==============================================================================
def extract_phobert_embedding(
    text: str,
    tokenizer,
    phobert_model,
    device: torch.device,
    max_len: int = 256,
    stride: int = 128
) -> torch.Tensor:
    """
    Trích xuất vector ngữ nghĩa 768 chiều từ text bằng cơ chế cửa sổ trượt PhoBERT.
    """
    clean_text = " ".join(text.strip().split())
    if not clean_text:
        return torch.zeros(768, device=device)

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True
    )
    input_ids = inputs["input_ids"][0]
    total_len = input_ids.size(0)

    if total_len <= max_len:
        batch_ids = input_ids.unsqueeze(0).to(device)
        attn_mask = torch.ones_like(batch_ids)
        with torch.no_grad():
            outputs = phobert_model(input_ids=batch_ids, attention_mask=attn_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            return cls_emb.squeeze(0)

    # Cửa sổ trượt
    chunks = []
    start = 0
    while start < total_len:
        end = min(start + max_len, total_len)
        chunk_ids = input_ids[start:end]
        if chunk_ids[0].item() != tokenizer.cls_token_id:
            chunk_ids = torch.cat([torch.tensor([tokenizer.cls_token_id]), chunk_ids])
        if chunk_ids[-1].item() != tokenizer.sep_token_id:
            chunk_ids = torch.cat([chunk_ids, torch.tensor([tokenizer.sep_token_id])])
        if chunk_ids.size(0) > max_len:
            chunk_ids = chunk_ids[:max_len]
        chunks.append(chunk_ids)
        if end >= total_len:
            break
        start += stride

    max_c_len = max(c.size(0) for c in chunks)
    padded = torch.full((len(chunks), max_c_len), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(chunks), max_c_len), dtype=torch.long)
    for i, c in enumerate(chunks):
        padded[i, :c.size(0)] = c
        mask[i, :c.size(0)] = 1

    padded = padded.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        out = phobert_model(input_ids=padded, attention_mask=mask)
        chunk_embs = out.last_hidden_state[:, 0, :]
        return chunk_embs.mean(dim=0)


# ==============================================================================
# 3. HÀM DỰ ĐOÁN QUAN HỆ GIỮA 2 ĐOẠN VĂN / THỰC THỂ
# ==============================================================================
def predict_relation(
    text_u: str,
    text_v: str,
    model: LegalRGCN,
    tokenizer,
    phobert_model,
    device: torch.device,
    top_k: int = 3
):
    """
    Dự đoán quan hệ có hướng từ u sang v (u -> v)
    """
    model.eval()
    emb_u = extract_phobert_embedding(text_u, tokenizer, phobert_model, device)
    emb_v = extract_phobert_embedding(text_v, tokenizer, phobert_model, device)

    x = torch.stack([emb_u, emb_v], dim=0)  # [2, 768]
    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    edge_type = torch.empty((0,), dtype=torch.long, device=device)
    pair_edges = torch.tensor([[0], [1]], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(x, edge_index, edge_type, pair_edges)
        probs = F.softmax(logits[0], dim=-1)

    top_probs, top_indices = torch.topk(probs, min(top_k, NUM_RELATION_CLASSES))
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        if idx == NO_RELATION_LABEL:
            rel_code = "NO_RELATION"
            rel_name = "Không có quan hệ (no_relation)"
        else:
            rel_code = ACTIVE_RELATIONS[idx] if idx < len(ACTIVE_RELATIONS) else f"REL_{idx}"
            rel_name = REL_VN_NAMES.get(rel_code, rel_code)
        results.append({
            "class_id": idx,
            "relation_code": rel_code,
            "relation_name": rel_name,
            "probability": prob * 100
        })
    return results


# ==============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Kiểm thử thủ công mô hình PhoBERT + RGCN")
    parser.add_argument("--test-cases-only", action="store_true", help="Chỉ chạy 3 test cases mẫu và dừng")
    parser.add_argument("--u", type=str, default="", help="Câu/Thực thể nguồn A")
    parser.add_argument("--v", type=str, default="", help="Câu/Thực thể đích B")
    args = parser.parse_args()

    print("=" * 75)
    print("🔬 CHƯƠNG TRÌNH KIỂM THỬ THỦ CÔNG MÔ HÌNH LAI PHOBERT + RGCN")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Thiết bị thực thi: {device}")

    # 1. Tải PhoBERT (Ưu tiên cache cục bộ)
    print("⏳ [1/3] Đang tải mô hình ngôn ngữ PhoBERT...")
    candidate_models = ["vinai/phobert-base", "vinai/phobert-base-v2"]
    tokenizer = None
    phobert_model = None

    for model_name in candidate_models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            phobert_model = AutoModel.from_pretrained(model_name).to(device)
            phobert_model.eval()
            print(f"   ✅ Đã tải xong PhoBERT ({model_name}).")
            break
        except Exception:
            continue

    if phobert_model is None:
        print("   ❌ Không thể tải PhoBERT. Vui lòng kiểm tra kết nối mạng!")
        return

    # 2. Khởi tạo và nạp trọng số mô hình RGCN
    print("⏳ [2/3] Đang nạp trọng số mô hình từ best_model/best_model_rgcn.pt...")
    ckpt_path = os.path.join(os.path.dirname(__file__), "best_model", "best_model_rgcn.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = "best_model/best_model_rgcn.pt"

    if not os.path.exists(ckpt_path):
        for alt in [
            "output_results/checkpoints/best_model_rgcn.pt",
            "checkpoints/best_model_rgcn.pt",
            "../best_model/best_model_rgcn.pt"
        ]:
            if os.path.exists(alt):
                ckpt_path = alt
                break

    if not os.path.exists(ckpt_path):
        print(f"   ❌ Không tìm thấy file checkpoint tại: {ckpt_path}")
        return

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        
        # Tự động nhận diện hidden_channels từ checkpoint
        hidden_channels = 64
        if "node_proj.weight" in sd:
            hidden_channels = sd["node_proj.weight"].shape[0]
        elif "node_proj.0.weight" in sd:
            hidden_channels = sd["node_proj.0.weight"].shape[0]
        elif "hidden_dim" in ckpt:
            hidden_channels = ckpt["hidden_dim"]
        elif "hidden_channels" in ckpt:
            hidden_channels = ckpt["hidden_channels"]

        model = LegalRGCN(
            in_channels=768,
            hidden_channels=hidden_channels,
            num_relations=NUM_RELATIONS,
            num_bases=8,
            num_rgcn_layers=2
        ).to(device)

        model.load_state_dict(sd, strict=False)
        model.eval()
        epoch = ckpt.get("epoch", "N/A")
        best_f1 = ckpt.get("best_val_f1", 0.0)
        print(f"   ✅ Nạp trọng số thành công! (Epoch: {epoch}, Val Macro-F1: {best_f1:.4f}, Hidden: {hidden_channels})")
    except Exception as e:
        print(f"   ❌ Lỗi khi nạp trọng số: {e}")
        return

    # Nếu người dùng truyền trực tiếp qua tham số dòng lệnh
    if args.u and args.v:
        print("\n" + "=" * 75)
        print("🎯 KẾT QUẢ DỰ ĐOÁN TỪ THAM SỐ DÒNG LỆNH:")
        print(f"   - Thực thể A: \"{args.u}\"")
        print(f"   - Thực thể B: \"{args.v}\"")
        preds = predict_relation(args.u, args.v, model, tokenizer, phobert_model, device, top_k=3)
        for rank, p in enumerate(preds, 1):
            mark = "⭐️ [TOP 1]" if rank == 1 else "        "
            print(f"     {mark} #{rank}: {p['relation_code']} ({p['relation_name']}) - Độ tin cậy: {p['probability']:.2f}%")
        return

    # 3. Chạy các Test Cases mẫu định sẵn
    print("\n" + "=" * 75)
    print("🧪 [3/3] CHẠY KIỂM THỬ TRÊN 3 TEST CASES MẪU")
    print("=" * 75)

    test_cases = [
        {
            "id": 1,
            "title": "Quan hệ Căn cứ Ban hành (CAN_CU_BAN_HANH)",
            "u": "Nghị quyết số 01/2021/NQ-HĐTP của Hội đồng Thẩm phán Tòa án nhân dân tối cao",
            "v": "Căn cứ Bộ luật Tố tụng hình sự số 101/2015/QH13 đã được sửa đổi, bổ sung",
            "expected": "CAN_CU_BAN_HANH"
        },
        {
            "id": 2,
            "title": "Quan hệ Phân cấp Cấu trúc (THUOC)",
            "u": "Điều 15. Thẩm quyền quyết định áp dụng biện pháp khẩn cấp tạm thời",
            "v": "Khoản 1. Tòa án có thẩm quyền áp dụng các biện pháp khẩn cấp tạm thời quy định tại Điều 111 của Bộ luật này",
            "expected": "THUOC"
        },
        {
            "id": 3,
            "title": "Cặp Âm - Không có Quan hệ (no_relation)",
            "u": "Người điều khiển xe mô tô hai bánh không đội mũ bảo hiểm bị phạt tiền từ 400.000 đồng đến 600.000 đồng",
            "v": "Doanh nghiệp có vốn đầu tư nước ngoài được phép thành lập chi nhánh tại Việt Nam",
            "expected": "no_relation"
        }
    ]

    for tc in test_cases:
        print(f"\n🔹 TEST CASE {tc['id']}: {tc['title']}")
        print(f"   - Thực thể A (Head): \"{tc['u']}\"")
        print(f"   - Thực thể B (Tail): \"{tc['v']}\"")
        print(f"   - Kỳ vọng (Ground Truth): {tc['expected']}")

        preds = predict_relation(tc['u'], tc['v'], model, tokenizer, phobert_model, device, top_k=3)
        print("   👉 Kết quả dự đoán Top 3:")
        for rank, p in enumerate(preds, 1):
            mark = "⭐️ [TOP 1]" if rank == 1 else "        "
            print(f"     {mark} #{rank}: {p['relation_code']} ({p['relation_name']}) - Độ tin cậy: {p['probability']:.2f}%")

    if args.test_cases_only:
        print("\n✅ Hoàn thành chạy kiểm thử các Test Cases mẫu!")
        return

    # 4. Chế độ Tương tác Người dùng Nhập trực tiếp
    print("\n" + "=" * 75)
    print("💬 CHẾ ĐỘ NHẬP TỰ DO ĐỂ KIỂM THỬ THỦ CÔNG (Nhấn Enter trống để thoát)")
    print("=" * 75)

    while True:
        try:
            print("\n-----------------------------------------------------------")
            input_u = input("👉 Nhập Thực thể / Câu văn A (Nguồn): ").strip()
            if not input_u:
                print("Đã thoát chế độ kiểm thử thủ công.")
                break
            input_v = input("👉 Nhập Thực thể / Câu văn B (Đích)  : ").strip()
            if not input_v:
                print("Đã thoát chế độ kiểm thử thủ công.")
                break

            preds = predict_relation(input_u, input_v, model, tokenizer, phobert_model, device, top_k=3)
            print("\n📊 KẾT QUẢ DỰ ĐOÁN QUAN HỆ (A -> B):")
            for rank, p in enumerate(preds, 1):
                star = "🏆 [DỰ ĐOÁN CHÍNH]" if rank == 1 else f"   [Top {rank}]       "
                print(f"   {star}: {p['relation_code']} - {p['relation_name']} (Xác suất: {p['probability']:.2f}%)")

        except KeyboardInterrupt:
            print("\nĐã hủy bởi người dùng.")
            break
        except Exception as err:
            print(f"❌ Có lỗi xảy ra trong lúc dự đoán: {err}")

if __name__ == "__main__":
    main()
