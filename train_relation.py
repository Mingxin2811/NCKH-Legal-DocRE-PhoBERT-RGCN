# -*- coding: utf-8 -*-
"""
train_relation.py
Huấn luyện Mô hình Lai GNN + Transformer Tối ưu (Legal RGCN + Gated Feature Fusion):
Trích xuất quan hệ thực thể (Relation Extraction) và Dự đoán liên kết (Link Prediction)
với Hierarchy Level Embedding và PhoBERT Semantic Features.
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import f1_score, accuracy_score  # type: ignore

try:
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")
except Exception:
    pass

from labels import (
    ALL_RELATIONS,
    METADATA_RELATIONS,
    STRUCTURAL_RELATIONS,
    LEGAL_RELATIONS,
    REL2ID,
    ID2REL
)
from model_gnn import (
    LegalRGCN,
    DualHeadLegalGNN,
    LEVEL2ID,
    NUM_LEVELS,
    NUM_RELATION_CLASSES,
    NO_RELATION_LABEL,
    NUM_LEGAL_CLASSES,
    NO_LEGAL_LABEL
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_relation_splits(
    homo_data,
    rng: random.Random,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    neg_to_pos_ratio: float = 2.0  # tăng tỷ lệ mẫu âm để giảm overfit và giảm precision
) -> Dict[str, Any]:
    """
    Sinh tập Train/Val/Test cho bài toán phân loại quan hệ:
    - 22 quan hệ dương (0..21)
    - 1 nhãn âm no_relation (22)
    """
    num_nodes = homo_data.num_nodes
    node_levels = homo_data.node_levels
    level_ids = torch.tensor([LEVEL2ID.get(lvl, 0) for lvl in node_levels], dtype=torch.long)

    edge_index = homo_data.edge_index
    edge_type = homo_data.edge_type

    pair_rels: Dict[Tuple[int, int], List[int]] = {}
    for i in range(edge_index.shape[1]):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        r = int(edge_type[i])
        pair_rels.setdefault((u, v), []).append(r)

    unique_pos_pairs = list(pair_rels.keys())
    rng.shuffle(unique_pos_pairs)

    n_pos = len(unique_pos_pairs)
    n_train = max(1, int(n_pos * train_ratio))
    n_val = max(1, int(n_pos * val_ratio))

    train_pos = unique_pos_pairs[:n_train]
    val_pos = unique_pos_pairs[n_train:n_train + n_val]
    test_pos = unique_pos_pairs[n_train + n_val:]

    train_pos_set = set(train_pos)
    train_mask = [
        (int(edge_index[0, i]), int(edge_index[1, i])) in train_pos_set
        for i in range(edge_index.shape[1])
    ]
    train_mp_edge_index = edge_index[:, train_mask]
    train_mp_edge_type = edge_type[train_mask]

    all_pos_set = set(unique_pos_pairs)

    def sample_negatives(count: int) -> List[Tuple[int, int]]:
        neg_p = set()
        attempts = 0
        while len(neg_p) < count and attempts < count * 20:
            attempts += 1
            u = rng.randint(0, num_nodes - 1)
            v = rng.randint(0, num_nodes - 1)
            if u != v and (u, v) not in all_pos_set:
                neg_p.add((u, v))
        return list(neg_p)

    def package_split(pos_list: List[Tuple[int, int]], neg_ratio: float):
        n_neg = int(len(pos_list) * neg_ratio)
        neg_list = sample_negatives(n_neg)
        all_p = pos_list + neg_list
        heads = [p[0] for p in all_p]
        tails = [p[1] for p in all_p]

        labels = []
        for p in pos_list:
            rels = pair_rels[p]
            labels.append(rels[0] if rels else NO_RELATION_LABEL)

        for _ in neg_list:
            labels.append(NO_RELATION_LABEL)

        return {
            "heads": torch.tensor(heads, dtype=torch.long),
            "tails": torch.tensor(tails, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "legal_labels": torch.tensor(labels, dtype=torch.long),  # tương thích ngược
            "pos_count": len(pos_list),
            "neg_count": len(neg_list)
        }

    return {
        "levels": level_ids,
        "train_mp_edge_index": train_mp_edge_index,
        "train_mp_edge_type": train_mp_edge_type,
        "train": package_split(train_pos, neg_to_pos_ratio),
        "val": package_split(val_pos, neg_to_pos_ratio),
        "test": package_split(test_pos, neg_to_pos_ratio)
    }

# Alias tương thích ngược
build_dual_head_splits = build_relation_splits


def compute_smoothed_weights(labels_list: List[int], num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for l in labels_list:
        if 0 <= l < num_classes:
            counts[l] += 1
    total = len(labels_list)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = np.sqrt(total / (counts[c] + 1.0))
        else:
            weights[c] = 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    workspace_root: str,
    num_epochs: int = 8,  # giảm epoch để tránh overfit
    lr: float = 2e-4,    # giảm learning rate nhẹ
    hidden_dim: int = 64,  # giảm kích thước ẩn cho mô hình nhẹ hơn
    level_dim: int = 64,
    num_bases: int = 8
):
    dataset_path = os.path.join(workspace_root, "dataset", "denoised_document_graphs.pt")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(workspace_root, "dataset", "denoised_dataset_graphs.pt")
    checkpoints_dir = os.path.join(workspace_root, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Chưa tìm thấy tập đồ thị đã khử nhiễu tại: {dataset_path}")

    print("=" * 80)
    print("=== HUẤN LUYỆN MÔ HÌNH LAI LEGAL RGCN (GNN + TRANSFORMER + GATED FUSION) ===")
    print("=" * 80)
    print(f"[*] Nạp dataset đồ thị từ: {dataset_path}")
    raw_graphs = torch.load(dataset_path, weights_only=False)
    print(f"[*] Tổng số văn bản sẵn sàng huấn luyện: {len(raw_graphs)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị huấn luyện: {device}")

    doc_splits: Dict[str, Any] = {}
    rng = random.Random(SEED)
    all_tr_labels: List[int] = []

    for doc_id, g_info in raw_graphs.items():
        homo_data = g_info["homo_data"]
        sp = build_relation_splits(homo_data, rng)
        doc_splits[doc_id] = {
            "homo_data": homo_data,
            "split": sp
        }
        all_tr_labels.extend(sp["train"]["labels"].tolist())
        print(f"\n[+] Tài liệu {doc_id} ({g_info['title'][:40]}...):")
        print(f"    - Đỉnh: {homo_data.num_nodes} | Train pairs: {sp['train']['heads'].shape[0]} | Val: {sp['val']['heads'].shape[0]} | Test: {sp['test']['heads'].shape[0]}")

    w_rel = compute_smoothed_weights(all_tr_labels, NUM_RELATION_CLASSES).to(device)
    crit_rel = nn.CrossEntropyLoss(weight=w_rel)

    model = LegalRGCN(
        in_channels=768,
        level_dim=level_dim,
        hidden_channels=hidden_dim,
        num_relations=len(ALL_RELATIONS),
        num_bases=num_bases,
        num_rgcn_layers=2,
        dropout=0.2
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[*] Kiến trúc LegalRGCN đã sẵn sàng: {total_params:,} tham số học được.")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_epoch = -1
    checkpoint_path = os.path.join(checkpoints_dir, "best_relation_model.pt")
    history: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print(f"{'Epoch':<6}| {'Train Loss':<14}| {'Val Macro F1':<16}| {'Val Accuracy':<14}")
    print("-" * 80)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_total = 0.0

        for doc_id, item in doc_splits.items():
            homo = item["homo_data"]
            sp = item["split"]

            x = homo.x.to(device)
            lvls = sp["levels"].to(device)
            ei = sp["train_mp_edge_index"].to(device)
            et = sp["train_mp_edge_type"].to(device)

            tr = sp["train"]
            h = tr["heads"].to(device)
            t = tr["tails"].to(device)
            y = tr["labels"].to(device)

            optimizer.zero_grad()
            logits, _ = model(x, lvls, ei, et, h, t)
            loss = crit_rel(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()

        scheduler.step()

        # Đánh giá Val
        model.eval()
        val_preds, val_trues = [], []

        with torch.no_grad():
            for doc_id, item in doc_splits.items():
                homo = item["homo_data"]
                sp = item["split"]

                x = homo.x.to(device)
                lvls = sp["levels"].to(device)
                ei = sp["train_mp_edge_index"].to(device)
                et = sp["train_mp_edge_type"].to(device)

                va = sp["val"]
                h = va["heads"].to(device)
                t = va["tails"].to(device)
                y = va["labels"].to(device)

                logits, _ = model(x, lvls, ei, et, h, t)

                val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                val_trues.extend(y.cpu().numpy())

        f1_val = f1_score(val_trues, val_preds, average="macro", zero_division=0)
        acc_val = accuracy_score(val_trues, val_preds)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss_total,
            "val_macro_f1": f1_val,
            "val_acc": acc_val
        })

        is_best = f1_val > best_val_f1
        if is_best:
            best_val_f1 = f1_val
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "hidden_dim": hidden_dim,
                "level_dim": level_dim,
                "num_bases": num_bases
            }, checkpoint_path)

        star = " *" if is_best else ""
        if epoch % 5 == 0 or is_best or epoch == num_epochs:
            print(f"{epoch:<6}| {train_loss_total:<14.4f}| {f1_val*100:<14.2f}% | {acc_val*100:<12.2f}%{star}")

    print("-" * 80)
    print(f"[+] Huấn luyện tối ưu hoàn tất! Best Epoch {best_epoch} | Val Macro-F1: {best_val_f1*100:.2f}%")
    print(f"[+] Đã lưu trọng số: {checkpoint_path}")

    metrics_path = os.path.join(checkpoints_dir, "training_history.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[+] Đã lưu lịch sử huấn luyện: {metrics_path}")


if __name__ == "__main__":
    ws = r"c:\Mingxin's documents\Self-Study\AI-NCKH\NCKH"
    train_model(ws, num_epochs=15)
