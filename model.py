# -*- coding: utf-8 -*-
"""
model.py
Định nghĩa kiến trúc mạng nơ-ron học sâu LegalPhoBERT_CRF phục vụ Task 4 (NER cấp token):
Kiến trúc kết hợp 3 tầng xử lý:
  1. Pre-trained Contextual Encoder: PhoBERT-base (768 chiều) trích xuất vector ngữ cảnh tiếng Việt
  2. Dropout + Linear Projection (Emission Layer): Chiếu từ chiều ẩn 768 sang điểm số phát xạ nhãn BIO
  3. Conditional Random Field (CRF Layer): Mô hình hóa xác suất chuyển tiếp giữa các nhãn BIO và giải mã Viterbi
"""

import torch
import torch.nn as nn
from transformers import AutoModel  # type: ignore
from torchcrf import CRF  # type: ignore
from typing import List, Dict, Any, Optional, Tuple

from labels import BIO_LABELS


class LegalPhoBERT_CRF(nn.Module):
    """
    Kiến trúc kết hợp PhoBERT và CRF cho bài toán nhận diện thực thể pháp lý tiếng Việt (Legal NER).
    Sử dụng cơ chế gom biểu diễn từ (First-subword Pooling) trước khi đưa vào CRF
    để tối ưu hóa ranh giới nhãn BIO cấp từ và tránh lỗi logic chuyển tiếp nhãn.
    """
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_labels: int = len(BIO_LABELS),
        dropout_prob: float = 0.15
    ):
        super(LegalPhoBERT_CRF, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name

        # 1. Tầng Pretrained Language Model (PhoBERT)
        self.phobert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.phobert.config.hidden_size

        # 2. Tầng Regularization & Dropout
        self.dropout = nn.Dropout(dropout_prob)

        # 3. Tầng chiếu tuyến tính phát xạ (Emission Layer)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        # 4. Tầng Trường ngẫu nhiên điều kiện (CRF Layer)
        # batch_first=True quy định định dạng tensor là [batch_size, seq_len, num_tags]
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_indices: torch.Tensor,
        word_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Any]:
        """
        Forward pass của mô hình:
        Args:
            input_ids: [batch_size, subword_len] - Mã token subwords của PhoBERT
            attention_mask: [batch_size, subword_len] - Mặt nạ attention của PhoBERT
            word_indices: [batch_size, max_words] - Chỉ số subword đầu tiên của mỗi từ trong câu
            word_mask: [batch_size, max_words] - Mặt nạ đánh dấu từ hợp lệ (True) và padding (False)
            labels: [batch_size, max_words] - Nhãn ID chuẩn của các từ (chỉ có khi huấn luyện)
        Returns:
            (loss, emissions) khi huấn luyện, hoặc (None, predictions) khi suy luận
        """
        # Bước 1: Trích xuất vector ngữ cảnh qua PhoBERT
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, subword_len, 768]
        sequence_output = self.dropout(sequence_output)

        # Bước 2: Gom biểu diễn subword đầu tiên của từng từ (First-subword Representation)
        batch_size, max_words = word_indices.shape
        batch_idx = torch.arange(batch_size, device=word_indices.device).unsqueeze(1).expand(-1, max_words)
        word_representations = sequence_output[batch_idx, word_indices]  # [batch_size, max_words, 768]

        # Bước 3: Tính điểm số phát xạ nhãn (Emission Scores)
        emissions = self.classifier(word_representations)  # [batch_size, max_words, num_labels]

        # Bước 4: Xử lý qua tầng CRF
        if labels is not None:
            # Khi huấn luyện: Tính Negative Log-Likelihood Loss qua CRF
            # Đảm bảo word_mask ở dạng Byte/Bool tensor cho CRF
            mask_crf = word_mask.bool()
            # Tránh giá trị -100 gây lỗi index của CRF
            clean_labels = labels.clone()
            clean_labels[clean_labels == -100] = 0
            
            loss = -self.crf(emissions, clean_labels, mask=mask_crf, reduction="mean")  # type: ignore
            return loss, emissions
        else:
            # Khi suy luận: Giải mã chuỗi nhãn tối ưu bằng thuật toán Viterbi
            mask_crf = word_mask.bool()
            predictions: List[List[int]] = self.crf.decode(emissions, mask=mask_crf)  # type: ignore
            return None, predictions

    def get_optimizer_grouped_parameters(
        self,
        lr_backbone: float = 2e-5,
        lr_head: float = 1e-3,
        weight_decay: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Chiến lược phân tầng Learning Rate (Differential Learning Rate):
        - PhoBERT (Pretrained backbone): học chậm với lr nhỏ (2e-5) để không phá vỡ tri thức cũ
        - Classifier & CRF (Head ngẫu nhiên): học nhanh với lr lớn (1e-3) để nhanh chóng hội tụ ma trận chuyển tiếp
        - Áp dụng Weight Decay chỉ cho các trọng số, loại bỏ bias và LayerNorm
        """
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        optimizer_grouped_parameters = [
            # 1. PhoBERT backbone - có weight decay
            {
                "params": [
                    p for n, p in self.phobert.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "lr": lr_backbone,
                "weight_decay": weight_decay,
            },
            # 2. PhoBERT backbone - không weight decay (bias, LayerNorm)
            {
                "params": [
                    p for n, p in self.phobert.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "lr": lr_backbone,
                "weight_decay": 0.0,
            },
            # 3. Classifier & CRF layer (Tầng đầu ra)
            {
                "params": [
                    p for n, p in list(self.classifier.named_parameters()) + list(self.crf.named_parameters())
                    if p.requires_grad
                ],
                "lr": lr_head,
                "weight_decay": weight_decay,
            },
        ]
        return optimizer_grouped_parameters

    def load_compatible_state_dict(
        self,
        state_dict: Dict[str, Any],
        old_label2id: Optional[Dict[str, int]] = None,
        new_label2id: Optional[Dict[str, int]] = None
    ):
        """
        Nạp state_dict có khả năng tương thích ngược (Backward Compatibility)
        khi số lượng nhãn được mở rộng từ 37 lên 39 (ví dụ bổ sung tieu_muc).
        Ánh xạ trọng số dựa trên TÊN NHÃN (Label Names) thay vì chỉ số thô
        để bảo toàn 100% độ chính xác cho các nhãn đã học trước đó.
        """
        current_state = self.state_dict()
        new_state = {}

        if new_label2id is None:
            from labels import LABEL2ID
            new_label2id = LABEL2ID

        if old_label2id is None:
            from labels import THUOC_TINH_LABELS, LUOC_DO_ENTITY_LABELS
            old_muc_luc = ["phan", "chuong", "muc", "dieu", "khoan", "diem"]
            old_entities = THUOC_TINH_LABELS + old_muc_luc + LUOC_DO_ENTITY_LABELS
            old_bio = ["o"]
            for ent in old_entities:
                old_bio.append(f"b-{ent}")
                old_bio.append(f"i-{ent}")
            old_label2id = {lbl: idx for idx, lbl in enumerate(old_bio)}

        for k, v in state_dict.items():
            if k not in current_state:
                new_state[k] = v
                continue

            if v.shape == current_state[k].shape:
                new_state[k] = v
            else:
                target = current_state[k].clone()
                if k == "classifier.weight":
                    for lbl_name, old_id in old_label2id.items():
                        if lbl_name in new_label2id:
                            new_id = new_label2id[lbl_name]
                            target[new_id] = v[old_id]
                elif k == "classifier.bias":
                    for lbl_name, old_id in old_label2id.items():
                        if lbl_name in new_label2id:
                            new_id = new_label2id[lbl_name]
                            target[new_id] = v[old_id]
                elif k == "crf.transitions":
                    for lbl_a, old_a in old_label2id.items():
                        if lbl_a in new_label2id:
                            new_a = new_label2id[lbl_a]
                            for lbl_b, old_b in old_label2id.items():
                                if lbl_b in new_label2id:
                                    new_b = new_label2id[lbl_b]
                                    target[new_a, new_b] = v[old_a, old_b]
                elif k in ["crf.start_transitions", "crf.end_transitions"]:
                    for lbl_name, old_id in old_label2id.items():
                        if lbl_name in new_label2id:
                            new_id = new_label2id[lbl_name]
                            target[new_id] = v[old_id]
                else:
                    if v.dim() == 1:
                        target[:v.shape[0]] = v
                    elif v.dim() == 2:
                        target[:v.shape[0], :v.shape[1]] = v
                new_state[k] = target

        self.load_state_dict(new_state)

