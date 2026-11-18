# -*- coding: utf-8 -*-
"""
dataset.py
Định nghĩa Data Pipeline chuẩn cho huấn luyện LegalPhoBERT_CRF:
1. LegalNERDataset: Lớp Dataset kế thừa torch.utils.data.Dataset
2. align_words_to_subwords: Trích xuất subwords và vị trí subword đầu tiên của từng từ
3. collate_fn: Hàm gom batch với Dynamic Padding (đệm động) tối ưu bộ nhớ
4. load_and_split_dataset: Phân chia tập dữ liệu 80% Train - 10% Val - 10% Test
"""

import os
import sys
import json
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer  # type: ignore

try:
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    getattr(sys.stderr, "reconfigure")(encoding="utf-8")
except Exception:
    pass

from labels import LABEL2ID


class LegalNERDataset(Dataset[Dict[str, Any]]):
    """
    Dataset cho bài toán nhận diện thực thể pháp lý (NER).
    Mỗi mẫu lưu trữ:
      - input_ids: chuỗi mã subwords của PhoBERT (bắt đầu bằng <s>, kết thúc bằng </s>)
      - attention_mask: mặt nạ 1 cho token hợp lệ
      - word_indices: chỉ số vị trí subword đầu tiên của mỗi từ trong câu
      - word_labels: nhãn ID tương ứng của từng từ (0 đến len(BIO_LABELS)-1)
      - raw_words: danh sách từ gốc (phục vụ đánh giá seqeval)
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer,
        max_subwords: int = 256
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_subwords = max_subwords
        self.bos_token = tokenizer.bos_token or "<s>"
        self.eos_token = tokenizer.eos_token or "</s>"
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

        self.features = [self._process_sample(s) for s in samples]

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        words = sample["words"]
        if "ner_tags" in sample:
            ner_ids = [LABEL2ID.get(t, 0) for t in sample["ner_tags"]]
        else:
            ner_ids = sample.get("ner_ids", [0] * len(words))

        subwords = [self.bos_token]
        word_indices = []
        valid_labels = []
        valid_words = []

        for word, label_id in zip(words, ner_ids):
            w_subwords = self.tokenizer.tokenize(word)
            if not w_subwords:
                continue

            # Giới hạn chiều dài subwords (nhường chỗ cho </s> ở cuối)
            if len(subwords) + len(w_subwords) >= self.max_subwords:
                break

            # Lưu vị trí của subword đầu tiên đại diện cho từ
            first_idx = len(subwords)
            word_indices.append(first_idx)
            valid_labels.append(label_id)
            valid_words.append(word)

            subwords.extend(w_subwords)

        subwords.append(self.eos_token)
        input_ids = self.tokenizer.convert_tokens_to_ids(subwords)
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_indices": word_indices,
            "word_labels": valid_labels,
            "raw_words": valid_words,
            "num_words": len(valid_labels)
        }

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.features[index]


def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 1) -> Dict[str, Any]:
    """
    Hàm gom batch với cơ chế Dynamic Padding (đệm động theo chiều dài lớn nhất trong batch):
    - input_ids & attention_mask: đệm theo max_subword_len
    - word_indices, word_mask, labels: đệm theo max_words_len
    """
    max_subword_len = max(len(item["input_ids"]) for item in batch)
    max_words_len = max(max(item["num_words"], 1) for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_word_indices = []
    batch_word_mask = []
    batch_labels = []
    raw_words_list = []

    for item in batch:
        sub_len = len(item["input_ids"])
        w_len = item["num_words"]

        # 1. Đệm subwords (PhoBERT input)
        pad_sub = max_subword_len - sub_len
        batch_input_ids.append(item["input_ids"] + [pad_token_id] * pad_sub)
        batch_attention_mask.append(item["attention_mask"] + [0] * pad_sub)

        # 2. Đệm word level (CRF input)
        pad_w = max_words_len - w_len
        # Vị trí đệm của word_indices gán an toàn là 0 (đã có word_mask che chắn)
        batch_word_indices.append(item["word_indices"] + [0] * pad_w)
        batch_word_mask.append([True] * w_len + [False] * pad_w)
        batch_labels.append(item["word_labels"] + [0] * pad_w)
        raw_words_list.append(item["raw_words"])

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "word_indices": torch.tensor(batch_word_indices, dtype=torch.long),
        "word_mask": torch.tensor(batch_word_mask, dtype=torch.bool),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "raw_words": raw_words_list
    }


def load_and_split_dataset(
    jsonl_path: str = "dataset/silver_dataset.jsonl",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    include_gold: bool = False,
    gold_path: str = "dataset/reviewed_gold.jsonl"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Nạp dữ liệu thực thể trực tiếp từ file JSONL chuẩn:
    - Nạp từ dataset/silver_dataset.jsonl và phân chia Train (80%) - Val (10%) - Test (10%).
    - Tự động tích hợp dữ liệu Vàng từ Human-in-the-Loop nếu include_gold=True.
    """

    samples = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

    # Xáo trộn ngẫu nhiên với seed cố định
    rng = random.Random(seed)
    rng.shuffle(samples)

    total = len(samples)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_data = samples[:n_train]
    val_data = samples[n_train:n_train + n_val]
    test_data = samples[n_train + n_val:]

    # Tích hợp dữ liệu Vàng từ Human-in-the-Loop vào tập Train
    if include_gold and os.path.exists(gold_path):
        gold_samples = []
        with open(gold_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    gold_samples.append(json.loads(line))
        if gold_samples:
            print(f"[*] Đã tích hợp {len(gold_samples)} mẫu Dữ liệu Vàng (Human-in-the-Loop) vào tập Train!")
            train_data.extend(gold_samples)

    return train_data, val_data, test_data
