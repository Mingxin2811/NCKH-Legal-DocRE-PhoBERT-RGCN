# -*- coding: utf-8 -*-
"""
evaluate_relation.py
Đánh giá Độc lập Mô hình Lai GNN + Transformer Tối ưu (Legal RGCN):
Kiểm định toàn diện 22 quan hệ pháp lý chuẩn hóa trên tập Test.
"""

import os
import sys
import json
import random
import torch
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score  # type: ignore

try:
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")
except Exception:
    pass

from labels import (
    ALL_RELATIONS,
    METADATA_RELATIONS,
    STRUCTURAL_RELATIONS,
    LEGAL_RELATIONS,
    REL_VN_NAMES,
    REL2ID,
    ID2REL
)
from model_gnn import (
    LegalRGCN,
    DualHeadLegalGNN,
    NUM_RELATION_CLASSES,
    NO_RELATION_LABEL,
    NUM_LEGAL_CLASSES,
    NO_LEGAL_LABEL
)
from train_relation import build_relation_splits, SEED


def evaluate_test_set(workspace_root: str):
    dataset_path = os.path.join(workspace_root, "dataset", "denoised_document_graphs.pt")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(workspace_root, "dataset", "denoised_dataset_graphs.pt")
    checkpoint_path = os.path.join(workspace_root, "checkpoints", "best_relation_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Chưa tìm thấy checkpoint tại: {checkpoint_path}")

    print("=" * 80)
    print("=== BÁO CÁO ĐÁNH GIÁ THỰC NGHIỆM TẬP TEST ĐỘC LẬP (LEGAL RGCN) ===")
    print("=" * 80)
    print(f"[*] Nạp checkpoint: {checkpoint_path}")
    cp = torch.load(checkpoint_path, weights_only=False)

    hidden_dim = cp.get("hidden_dim", 128)
    level_dim = cp.get("level_dim", 64)
    num_bases = cp.get("num_bases", 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LegalRGCN(
        in_channels=768,
        level_dim=level_dim,
        hidden_channels=hidden_dim,
        num_relations=len(ALL_RELATIONS),
        num_bases=num_bases,
        num_rgcn_layers=2,
        dropout=0.0
    ).to(device)

    model.load_state_dict(cp["model_state_dict"])
    model.eval()

    raw_graphs = torch.load(dataset_path, weights_only=False)
    rng = random.Random(SEED)

    test_preds, test_trues = [], []

    with torch.no_grad():
        for doc_id, g_info in raw_graphs.items():
            homo = g_info["homo_data"]
            sp = build_relation_splits(homo, rng)

            x = homo.x.to(device)
            lvls = sp["levels"].to(device)
            ei = sp["train_mp_edge_index"].to(device)
            et = sp["train_mp_edge_type"].to(device)

            te = sp["test"]
            h = te["heads"].to(device)
            t = te["tails"].to(device)
            y = te["labels"].to(device)

            logits, _ = model(x, lvls, ei, et, h, t)

            test_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            test_trues.extend(y.cpu().numpy())

    test_preds = np.array(test_preds)
    test_trues = np.array(test_trues)
    total_pairs = len(test_trues)

    acc = accuracy_score(test_trues, test_preds)
    f1_weighted = f1_score(test_trues, test_preds, average="weighted", zero_division=0)
    f1_macro = f1_score(test_trues, test_preds, average="macro", zero_division=0)

    print(f"\n[+] BẢNG 1: HIỆU NĂNG TỔNG THỂ TRÊN TẬP TEST (Tổng: {total_pairs:,} cặp thực thể)")
    print("-" * 75)
    print(f"| {'Mô hình (Model)':<35} | {'Accuracy':<12} | {'Weighted-F1':<12} | {'Macro-F1':<10} |")
    print("-" * 75)
    print(f"| {f'Legal RGCN ({NUM_RELATION_CLASSES} lớp)':<35} | {acc*100:>10.2f}% | {f1_weighted*100:>10.2f}% | {f1_macro*100:>8.2f}% |")
    print("-" * 75)

    # Phân rã theo cấp độ
    struct_mask = (test_trues == 0)
    f1_struct = f1_score(test_trues[struct_mask], test_preds[struct_mask], average="micro", zero_division=0) if struct_mask.sum() > 0 else 0.0
    rec_struct = recall_score(test_trues[struct_mask], test_preds[struct_mask], average="micro", zero_division=0) if struct_mask.sum() > 0 else 0.0
    prec_struct = precision_score(test_trues[struct_mask], test_preds[struct_mask], average="micro", zero_division=0) if struct_mask.sum() > 0 else 0.0

    legal_mask = np.isin(test_trues, list(range(1, len(ALL_RELATIONS))))
    f1_inter = f1_score(test_trues[legal_mask], test_preds[legal_mask], average="micro", zero_division=0) if legal_mask.sum() > 0 else 0.0
    rec_inter = recall_score(test_trues[legal_mask], test_preds[legal_mask], average="micro", zero_division=0) if legal_mask.sum() > 0 else 0.0
    prec_inter = precision_score(test_trues[legal_mask], test_preds[legal_mask], average="micro", zero_division=0) if legal_mask.sum() > 0 else 0.0

    no_rel_true = (test_trues == NO_RELATION_LABEL)
    no_rel_pred = (test_preds == NO_RELATION_LABEL)
    f1_no_rel = f1_score(no_rel_true, no_rel_pred, zero_division=0)
    rec_no_rel = recall_score(no_rel_true, no_rel_pred, zero_division=0)
    prec_no_rel = precision_score(no_rel_true, no_rel_pred, zero_division=0)

    print(f"\n[+] BẢNG 2: PHÂN RÃ HIỆU NĂNG THEO CÁC CẤP ĐỘ QUAN HỆ")
    print("-" * 88)
    print(f"| {'Cấp độ Quan Hệ':<45} | {'Số mẫu Test':<12} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8} |")
    print("-" * 88)
    print(f"| {'Cấp 2: Quan hệ Cấu trúc (THUOC)':<45} | {struct_mask.sum():>10,} | {prec_struct*100:>8.2f}% | {rec_struct*100:>6.2f}% | {f1_struct*100:>6.2f}% |")
    print(f"| {'Cấp 1: Pháp lý Liên văn bản (21 quan hệ)':<45} | {legal_mask.sum():>10,} | {prec_inter*100:>8.2f}% | {rec_inter*100:>6.2f}% | {f1_inter*100:>6.2f}% |")
    print(f"| {'Cạnh âm: Phát hiện Cặp không liên kết':<45} | {no_rel_true.sum():>10,} | {prec_no_rel*100:>8.2f}% | {rec_no_rel*100:>6.2f}% | {f1_no_rel*100:>6.2f}% |")
    print("-" * 88)

    # Detailed report for present labels
    present_labels = sorted(list(set(test_trues).union(set(test_preds))))
    target_names = [REL_VN_NAMES.get(ID2REL.get(lbl, ""), "Không liên kết" if lbl == NO_RELATION_LABEL else ID2REL.get(lbl, f"Label_{lbl}")) for lbl in present_labels]

    print("\n[+] BẢNG 3: BÁO CÁO CHI TIẾT TỪNG NHÃN XUẤT HIỆN TRÊN TẬP TEST:")
    print(classification_report(test_trues, test_preds, labels=present_labels, target_names=target_names, zero_division=0))

    results = {
        "overall": {
            "accuracy": acc,
            "weighted_f1": f1_weighted,
            "macro_f1": f1_macro
        },
        "level_breakdown": {
            "structure_f1": f1_struct,
            "legal_inter_f1": f1_inter,
            "no_relation_f1": f1_no_rel
        }
    }
    res_path = os.path.join(workspace_root, "checkpoints", "test_evaluation_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Đã lưu báo cáo đánh giá hoàn chỉnh tại: {res_path}")


if __name__ == "__main__":
    ws = r"c:\Mingxin's documents\Self-Study\AI-NCKH\NCKH"
    evaluate_test_set(ws)
