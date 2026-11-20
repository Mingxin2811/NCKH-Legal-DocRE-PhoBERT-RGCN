"""Shared training/inference models. GNN edges are observable, never gold labels."""

import torch
from torch import nn
from torch_geometric.nn import RGCNConv
from labels import ALL_ENTITIES, ALL_RELATIONS

FEATURE_DIM = 772
NUM_TOPOLOGY_RELATIONS = 4


class BoundaryNER(nn.Module):
    """Independent boundaries permit overlapping entity types, unlike flat BIO."""

    def __init__(self, hidden=128, dropout=0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * len(ALL_ENTITIES)),
        )

    def forward(self, x):
        return self.head(x)


class LegalRGCN(nn.Module):
    def __init__(self, hidden_channels=128, dropout=0.2, use_gnn=True):
        super().__init__()
        self.use_gnn = use_gnn
        self.node_proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )
        self.convs = (
            nn.ModuleList(
                [
                    RGCNConv(hidden_channels, hidden_channels, 4, num_bases=4)
                    for _ in range(2)
                ]
            )
            if use_gnn
            else nn.ModuleList()
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in self.convs])
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_channels * 2, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, len(ALL_RELATIONS)),
        )

    def encode_nodes(self, x, edge_index, edge_type):
        semantic = self.node_proj(x)
        h = semantic
        for conv, norm in zip(self.convs, self.norms):
            h = self.dropout(
                torch.nn.functional.gelu(norm(conv(h, edge_index, edge_type)))
            )
        if self.use_gnn:
            gate = torch.sigmoid(self.gate(torch.cat([semantic, h], dim=-1)))
            h = gate * semantic + (1 - gate) * h
        return h

    def classify(self, h, pairs):
        u, v = h[pairs[0]], h[pairs[1]]
        return self.classifier(torch.cat([u, v, abs(u - v), u * v], dim=-1))

    def forward(self, x, edge_index, edge_type, pairs):
        return self.classify(self.encode_nodes(x, edge_index, edge_type), pairs)
