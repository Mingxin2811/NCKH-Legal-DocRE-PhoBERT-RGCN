# -*- coding: utf-8 -*-
"""
train_ner.py
Quy trình Huấn luyện & Đánh giá (Training & Evaluation Pipeline) kiến trúc LegalPhoBERT_CRF:
1. Nạp dữ liệu silver_dataset.jsonl (phân chia 80% Train, 10% Val, 10% Test).
2. Tối ưu hóa với AdamW áp dụng Phân tầng Learning Rate (Differential LR):
   - PhoBERT backbone: lr=2e-5, weight_decay=0.01
   - Classifier & CRF head: lr=1e-3, weight_decay=0.01
3. Linear Warmup & Decay Learning Rate Scheduler.
4. Đánh giá chuẩn quốc tế bằng thư viện seqeval (Precision, Recall, F1 cấp thực thể).
5. Lưu trữ Best Checkpoint (best_model.pt) và lịch sử huấn luyện (training_history.json).
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np
from typing import Any, Tuple, List, Dict, Optional

try:
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    getattr(sys.stderr, "reconfigure")(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # type: ignore
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report  # type: ignore

from model import LegalPhoBERT_CRF
from dataset import LegalNERDataset, collate_fn, load_and_split_dataset
from labels import LABEL2ID, ID2LABEL, BIO_LABELS


def set_seed(seed: int = 42):
    """Cố định seed ngẫu nhiên cho tính tái lập (Reproducibility)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def id_to_seqeval_tag(tag_id: int) -> str:
    """
    Chuyển ID sang nhãn định dạng chuẩn của seqeval (B-*, I-*, O):
    - 0 -> 'O'
    - 'b-so_hieu' -> 'B-so_hieu'
    - 'i-chuc_danh_nguoi_ky' -> 'I-chuc_danh_nguoi_ky'
    """
    if tag_id == 0:
        return "O"
    raw_tag = ID2LABEL.get(tag_id, "o")
    if raw_tag == "o":
        return "O"
    if raw_tag.startswith("b-") or raw_tag.startswith("i-"):
        return raw_tag[0].upper() + raw_tag[1:]
    return raw_tag


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    max_grad_norm: float = 1.0,
    log_interval: int = 20
) -> float:
    """Huấn luyện 1 epoch và trả về Loss trung bình (Negative Log-Likelihood)"""
    model.train()
    total_loss = 0.0
    total_batches = len(dataloader)
    start_time = time.time()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        word_indices = batch["word_indices"].to(device)
        word_mask = batch["word_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        loss, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_indices=word_indices,
            word_mask=word_mask,
            labels=labels
        )

        loss.backward()
        # Gradient Clipping để ngăn chặn bùng nổ gradient trong CRF/Transformer
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (step + 1) % log_interval == 0 or (step + 1) == total_batches:
            elapsed = time.time() - start_time
            avg_loss_so_far = total_loss / (step + 1)
            lr_backbone = optimizer.param_groups[0]["lr"]
            lr_head = optimizer.param_groups[2]["lr"]
            print(f"  [Batch {step + 1:>3}/{total_batches}] | Loss: {avg_loss_so_far:.4f} | "
                  f"lr_bb: {lr_backbone:.2e} | lr_head: {lr_head:.2e} | {elapsed:.1f}s", flush=True)

    return total_loss / max(total_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float, float, str]:
    """
    Đánh giá mô hình trên tập Validation hoặc Test:
    - Tính Loss CRF
    - Giải mã Viterbi lấy chuỗi nhãn tối ưu
    - Tính Precision, Recall, F1 qua thư viện seqeval
    """
    model.eval()
    total_loss = 0.0
    total_batches = len(dataloader)

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_indices = batch["word_indices"].to(device)
            word_mask = batch["word_mask"].to(device)
            labels = batch["labels"].to(device)

            # 1. Tính Loss và Emissions trong cùng 1 pass duy nhất
            loss, emissions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_indices=word_indices,
                word_mask=word_mask,
                labels=labels
            )
            total_loss += loss.item()

            # 2. Giải mã Viterbi trực tiếp từ emissions (tiết kiệm 50% thời gian đánh giá)
            mask_crf = word_mask.bool()
            crf_layer = getattr(model, "crf", None)
            if crf_layer is not None:
                predictions = crf_layer.decode(emissions, mask=mask_crf)
            else:
                _, predictions = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_indices=word_indices,
                    word_mask=word_mask,
                    labels=None
                )

            # 3. Lọc token hợp lệ và chuyển đổi sang tag seqeval
            batch_labels = labels.cpu().tolist()
            batch_masks = word_mask.cpu().tolist()

            for i in range(len(predictions)):
                pred_seq = predictions[i]
                mask_seq = batch_masks[i]
                label_seq = batch_labels[i]

                valid_len = sum(1 for m in mask_seq if m)
                true_ids = label_seq[:valid_len]
                pred_ids = pred_seq[:valid_len]

                y_true = [id_to_seqeval_tag(t) for t in true_ids]
                y_pred = [id_to_seqeval_tag(p) for p in pred_ids]

                y_true_all.append(y_true)
                y_pred_all.append(y_pred)

    avg_loss = float(total_loss / max(total_batches, 1))
    prec = float(precision_score(y_true_all, y_pred_all, zero_division="0"))  # type: ignore
    rec = float(recall_score(y_true_all, y_pred_all, zero_division="0"))  # type: ignore
    f1 = float(f1_score(y_true_all, y_pred_all, zero_division="0"))  # type: ignore
    report = str(classification_report(y_true_all, y_pred_all, digits=4, zero_division="0"))

    return avg_loss, prec, rec, f1, report


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình LegalPhoBERT_CRF cho bài toán Legal NER")
    parser.add_argument("--data_path", type=str, default="dataset/silver_dataset.jsonl", help="Đường dẫn file silver_dataset.jsonl")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base", help="Pretrained model tên HuggingFace")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Thư mục lưu mô hình tốt nhất")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch huấn luyện")
    parser.add_argument("--lr_backbone", type=float, default=2e-5, help="Tốc độ học cho PhoBERT backbone")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Tốc độ học cho Classifier & CRF")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Hệ số suy giảm trọng số Weight Decay")
    parser.add_argument("--max_subwords", type=int, default=256, help="Chiều dài subword tối đa của PhoBERT")
    parser.add_argument("--max_samples", type=int, default=None, help="Giới hạn số mẫu (dùng cho dry run/thử nghiệm nhanh)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    parser.add_argument("--log_interval", type=int, default=20, help="Số batch in log tiến độ")
    parser.add_argument("--include_gold", action="store_true", default=True, help="Tích hợp dữ liệu Vàng (HITL) vào tập Train")
    parser.add_argument("--gold_path", type=str, default="dataset/reviewed_gold.jsonl", help="Đường dẫn file dữ liệu Vàng")
    parser.add_argument("--pretrained_checkpoint", type=str, default="checkpoints/best_model_v1.pt", help="Checkpoint nạp trước để fine-tune (Warm Start)")

    args = parser.parse_args()

    # Thiết lập môi trường và thiết bị
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(os.cpu_count() or 4, 8))
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  QUY TRÌNH TÁI HUẤN LUYỆN (RETRAIN) LegalPhoBERT-CRF (BƯỚC 4 HITL)")
    print("=" * 70)
    print(f"Thiết bị tính toán (Device)  : {device}")
    print(f"Pretrained Model             : {args.model_name}")
    print(f"Số lượng nhãn BIO (Classes)  : {len(BIO_LABELS)}")
    print(f"Learning Rate (Backbone/Head): {args.lr_backbone} / {args.lr_head}")
    print(f"Batch Size / Epochs          : {args.batch_size} / {args.epochs}")
    print(f"Max Samples (Giới hạn)       : {args.max_samples if args.max_samples else 'Toàn bộ dữ liệu'}")
    print(f"Tích hợp Dữ liệu Vàng (HITL) : {args.include_gold} ({args.gold_path})")
    print(f"Warm Start từ Checkpoint cũ  : {args.pretrained_checkpoint if args.pretrained_checkpoint else 'Không'}")

    # 1. Khởi tạo Tokenizer
    print("\n[1/5] Đang nạp PhoBERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 2. Phân chia tập dữ liệu
    print("[2/5] Đang đọc và phân chia tập dữ liệu (80% Train - 10% Val - 10% Test)...")
    train_samples, val_samples, test_samples = load_and_split_dataset(
        jsonl_path=args.data_path,
        include_gold=args.include_gold,
        gold_path=args.gold_path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed
    )

    if args.max_samples:
        silver_part = [s for s in train_samples if not str(s.get("id", "")).startswith("gold_")]
        gold_part = [s for s in train_samples if str(s.get("id", "")).startswith("gold_")]
        train_samples = silver_part[:args.max_samples] + gold_part
        val_samples = val_samples[:max(int(args.max_samples * 0.125), 16)]
        test_samples = test_samples[:max(int(args.max_samples * 0.125), 16)]

    print(f"  -> Tập Huấn luyện (Train) : {len(train_samples):,} mẫu")
    print(f"  -> Tập Đánh giá (Val)     : {len(val_samples):,} mẫu")
    print(f"  -> Tập Kiểm thử (Test)    : {len(test_samples):,} mẫu")

    train_dataset = LegalNERDataset(train_samples, tokenizer, max_subwords=args.max_subwords)
    val_dataset = LegalNERDataset(val_samples, tokenizer, max_subwords=args.max_subwords)
    test_dataset = LegalNERDataset(test_samples, tokenizer, max_subwords=args.max_subwords)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Khởi tạo Mô hình & Optimizer
    print("\n[3/5] Đang khởi tạo mô hình LegalPhoBERT_CRF và cấu hình Differential LR...")
    model = LegalPhoBERT_CRF(
        model_name=args.model_name,
        num_labels=len(BIO_LABELS),
        dropout_prob=0.15
    ).to(device)

    # Nạp trọng số khởi đầu nếu có pretrained_checkpoint (Warm Start)
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        print(f"[*] Nạp trọng số từ Checkpoint cũ: {args.pretrained_checkpoint} (Warm Start)...")
        old_ckpt = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
        model.load_compatible_state_dict(
            old_ckpt["model_state_dict"],
            old_label2id=old_ckpt.get("label2id")
        )
        print(f"    -> Đã mở rộng ma trận thành công lên {len(BIO_LABELS)} nhãn và kế thừa 100% trọng số cũ!")

    optimizer_grouped_parameters = model.get_optimizer_grouped_parameters(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay
    )
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    total_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_training_steps * 0.1)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    # 4. Vòng lặp Huấn luyện (Training Loop)
    print("\n[4/5] Bắt đầu vòng lặp huấn luyện:")
    best_val_f1 = 0.0
    history = []
    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            log_interval=args.log_interval
        )

        val_loss, val_prec, val_rec, val_f1, _ = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} Kết quả: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | "
              f"Val Precision = {val_prec*100:.2f}% | Val Recall = {val_rec*100:.2f}% | Val F1 = {val_f1*100:.2f}%")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"  >>> Cải thiện F1! Đang lưu Best Checkpoint tại: {checkpoint_path}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "num_labels": len(BIO_LABELS),
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "args": vars(args)
            }, checkpoint_path)

    # Lưu lịch sử huấn luyện
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu lịch sử huấn luyện tại: {history_path}")

    # 5. Đánh giá cuối cùng trên Tập Kiểm thử (Test Set)
    print("\n[5/5] Đánh giá mô hình tốt nhất trên Tập Kiểm thử độc lập (Test Set)...")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Đã nạp checkpoint tốt nhất tại Epoch {checkpoint['epoch']} (Val F1: {checkpoint['val_f1']*100:.2f}%)")

    test_loss, test_prec, test_rec, test_f1, test_report = evaluate(model, test_loader, device)

    print("\n" + "=" * 70)
    print("                KẾT QUẢ TRÊN TẬP KIỂM THỬ (TEST SET)")
    print("=" * 70)
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Test Precision : {test_prec*100:.2f}%")
    print(f"Test Recall    : {test_rec*100:.2f}%")
    print(f"Test F1-Score  : {test_f1*100:.2f}%")
    print("\nChi tiết báo cáo phân loại theo từng loại thực thể (Classification Report):")
    print(test_report)
    print("=" * 70)


if __name__ == "__main__":
    main()
