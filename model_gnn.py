# -*- coding: utf-8 -*-
"""
model_gnn.py
Kiến trúc Mô hình Lai GNN + Transformer Tối ưu (Legal RGCN + Gated Feature Fusion) phục vụ:
Trích xuất quan hệ thực thể (Relation Extraction) và Dự đoán liên kết (Link Prediction)
trong đồ thị tri thức pháp lý đa quan hệ cấp độ tài liệu (DocRE).

Các cải tiến cốt lõi:
  1. PhoBERT Contextual Node Features:
     - Biểu diễn đặc trưng ngữ cảnh 768 chiều trực tiếp từ mô hình ngôn ngữ tiếng Việt PhoBERT.
  2. Relational Message Passing (R-GCN 2 tầng với Basis Decomposition):
     - Lan truyền thông tin đa quan hệ có hướng trên 22 quan hệ (1 cấu trúc THUOC + 21 quan hệ pháp lý liên văn bản).
  3. Gated Feature Fusion Layer:
     - Tự động cân bằng giữa biểu diễn ngữ nghĩa PhoBERT (cục bộ) và biểu diễn cấu trúc R-GCN (toàn cục).
  4. Unified Relation Classifier (23 lớp):
     - Phân loại trực tiếp 22 quan hệ chuẩn hóa + 1 nhãn âm (no_relation).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from torch_geometric.nn import RGCNConv  # type: ignore

from labels import (
    ALL_RELATIONS,
    METADATA_RELATIONS,
    STRUCTURAL_RELATIONS,
    LEGAL_RELATIONS,
    REL2ID,
    ID2REL
)

# Tổng số quan hệ: 22 quan hệ thực (0..21) + 1 nhãn âm no_relation (22) = 23 lớp
NUM_RELATION_CLASSES = len(ALL_RELATIONS) + 1
NO_RELATION_LABEL = len(ALL_RELATIONS)

# Hằng số tương thích ngược
NUM_LEGAL_CLASSES = NUM_RELATION_CLASSES
NO_LEGAL_LABEL = NO_RELATION_LABEL
NUM_TEMP_CLASSES = 0
NO_TEMP_LABEL = -1
NUM_LEVELS = 0
LEVEL_LIST = []
LEVEL2ID = {}
ID2LEVEL = {}


class GatedFusion(nn.Module):
    """
    Cơ chế cổng hợp nhất thích ứng (Gated Feature Fusion):
    h_fused = gate * h_sem + (1 - gate) * h_topo
    """
    def __init__(self, hidden_dim: int):
        super(GatedFusion, self).__init__()
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_sem: torch.Tensor, h_topo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([h_sem, h_topo], dim=-1)
        gate = self.sigmoid(self.gate_linear(concat))
        h_fused = gate * h_sem + (1.0 - gate) * h_topo
        return h_fused, gate


class LegalRGCN(nn.Module):
    """
    Mô hình Lai GNN + Transformer (Legal RGCN + PhoBERT + Gated Fusion)
    cho bài toán trích xuất quan hệ trên đồ thị tri thức pháp luật.
    """
    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 128,
        num_relations: int = len(ALL_RELATIONS),
        num_bases: int = 8,
        num_rgcn_layers: int = 2,
        dropout: float = 0.2,
        level_dim: int = 0  # Tương thích ngược
    ):
        super(LegalRGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_relations = num_relations

        # 1. Chiếu đặc trưng ngữ nghĩa PhoBERT (768 -> hidden_channels)
        self.proj_semantic = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )

        # 2. Các tầng R-GCN đa quan hệ (Relational Message Passing)
        self.rgcn_layers = nn.ModuleList()
        self.rgcn_norms = nn.ModuleList()
        for _ in range(num_rgcn_layers):
            self.rgcn_layers.append(
                RGCNConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    num_relations=num_relations,
                    num_bases=num_bases
                )
            )
            self.rgcn_norms.append(nn.LayerNorm(hidden_channels))

        self.rgcn_dropout = nn.Dropout(dropout)
        self.post_dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(0.1)

        # 3. Tầng Hợp nhất Gated Fusion
        self.fusion = GatedFusion(hidden_dim=hidden_channels)

        # 4. Bộ phân loại Quan hệ Pháp lý & Cấu trúc (Unified Relation Classifier - 23 lớp)
        pair_in_dim = hidden_channels * 4
        mlp_dim = hidden_channels * 2
        self.relation_classifier = nn.Sequential(
            nn.Linear(pair_in_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, NUM_RELATION_CLASSES)
        )
        self.legal_classifier = self.relation_classifier

    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        hierarchy_levels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trích xuất biểu diễn đỉnh kết hợp giữa ngữ nghĩa PhoBERT và tương tác R-GCN.
        """
        h_sem = self.proj_semantic(x)

        h_topo = h_sem
        if edge_index.numel() > 0:
            for conv, norm in zip(self.rgcn_layers, self.rgcn_norms):
                h_next = conv(h_topo, edge_index, edge_type)
                h_topo = norm(self.leaky_relu(h_next))
                h_topo = self.rgcn_dropout(h_topo)

        h_fused, gate = self.fusion(h_sem, h_topo)
        h_fused = self.post_dropout(h_fused)  # áp dụng dropout sau fusion để giảm overfit
        return h_fused, gate

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dự đoán quan hệ cho các cặp đỉnh ứng viên.
        Hỗ trợ linh hoạt mọi cách gọi:
          - model(x, edge_index, edge_type, candidate_pairs)
          - model(x, hierarchy_levels, edge_index, edge_type, candidate_pairs)
          - model(x, edge_index, edge_type, heads, tails)
          - model(x, hierarchy_levels, edge_index, edge_type, heads, tails)
        Trả về:
          - logits: [E_cand, NUM_RELATION_CLASSES] (23 lớp)
          - h_fused: [N, hidden_channels] (biểu diễn nút sau hợp nhất)
        """
        # Phân giải đối số linh hoạt
        tail_indices = kwargs.get("tail_indices", None)

        if len(args) == 4:
            # (x, edge_index, edge_type, candidate_pairs)
            x, edge_index, edge_type, candidate_pairs = args
        elif len(args) == 5:
            # Kiểm tra xem args[1] là hierarchy_levels (1D) hay edge_index (2D)
            if args[1] is None or (torch.is_tensor(args[1]) and args[1].dim() == 1):
                # (x, hierarchy_levels, edge_index, edge_type, candidate_pairs)
                x, _, edge_index, edge_type, candidate_pairs = args
            else:
                # (x, edge_index, edge_type, heads, tails)
                x, edge_index, edge_type, candidate_pairs, tail_indices = args
        elif len(args) >= 6:
            # (x, hierarchy_levels, edge_index, edge_type, heads, tails)
            x, _, edge_index, edge_type, candidate_pairs, tail_indices = args[:6]
        elif len(args) == 1 and isinstance(args[0], tuple):
            return self.forward(*args[0], **kwargs)
        else:
            raise ValueError(f"Số lượng đối số không hợp lệ trong forward: {len(args)}")

        h_fused, _ = self.encode_nodes(x, edge_index, edge_type)

        if tail_indices is not None:
            src_indices = candidate_pairs
            dst_indices = tail_indices
        elif candidate_pairs.numel() > 0 and candidate_pairs.dim() == 2 and candidate_pairs.size(0) == 2:
            src_indices = candidate_pairs[0]
            dst_indices = candidate_pairs[1]
        elif candidate_pairs.numel() == 0 or candidate_pairs.size(-1) == 0:
            device = x.device
            return torch.empty((0, NUM_RELATION_CLASSES), device=device), h_fused
        else:
            raise ValueError(f"Dạng candidate_pairs không hợp lệ: shape={candidate_pairs.shape}")

        if src_indices.numel() == 0:
            device = x.device
            return torch.empty((0, NUM_RELATION_CLASSES), device=device), h_fused

        h_u = h_fused[src_indices]
        h_v = h_fused[dst_indices]

        # Biểu diễn cặp đa góc nhìn: [h_u, h_v, |h_u - h_v|, h_u * h_v]
        pair_feat = torch.cat([
            h_u,
            h_v,
            torch.abs(h_u - h_v),
            h_u * h_v
        ], dim=-1)

        logits = self.relation_classifier(pair_feat)
        return logits, h_fused


# Alias tương thích ngược
DualHeadLegalGNN = LegalRGCN
LegalGNNTransformerRelationExtractor = LegalRGCN


if __name__ == "__main__":
    import sys
    try:
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    except Exception:
        pass

    print("=== KIỂM THỬ KHỞI TẠO KIẾN TRÚC MÔ HÌNH LAI (model_gnn.py) ===")
    num_nodes = 50
    num_edges = 120
    num_pairs = 30

    x_dummy = torch.randn(num_nodes, 768)
    edge_index_dummy = torch.randint(0, num_nodes, (2, num_edges))
    edge_type_dummy = torch.randint(0, len(ALL_RELATIONS), (num_edges,))
    candidate_pairs_dummy = torch.randint(0, num_nodes, (2, num_pairs))

    model = LegalRGCN()
    # Test gọi 4 đối số
    logits, h_fused = model(x_dummy, edge_index_dummy, edge_type_dummy, candidate_pairs_dummy)

    print(f"✅ Khởi tạo thành công mô hình LegalRGCN (Thuần túy PhoBERT + RGCN + Gated Fusion)!")
    print(f"   - Số quan hệ đồ thị (num_relations): {len(ALL_RELATIONS)}")
    print(f"   - Số lớp phân loại (num_classes): {NUM_RELATION_CLASSES} (22 quan hệ + 1 nhãn âm)")
    print(f"   - Logits output shape: {logits.shape}")
    print(f"   - h_fused output shape: {h_fused.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Tổng số tham số học được (Trainable Params): {total_params:,}")
