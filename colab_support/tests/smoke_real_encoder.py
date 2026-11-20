"""Small real-PhoBERT CPU check, not an experiment or accuracy benchmark."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from context_encoder import ContextEncoder
from data_pipeline import body_text
from model_gnn import BoundaryNER, LegalRGCN
from training import boundary_targets, make_graph, sample_training_pairs

p = argparse.ArgumentParser()
p.add_argument("--json-file", required=True)
p.add_argument("--encoder", default="vinai/phobert-base")
p.add_argument("--local-files-only", action="store_true")
args = p.parse_args()
raw = json.loads(Path(args.json_file).read_text(encoding="utf-8-sig"))
text = body_text(raw["full_text"])[:1800]
entities = [
    e
    for e in raw["entities"]
    if e["end"] <= len(text) and text[e["start"] : e["end"]] == e["text"]
]
ids = {e["id"] for e in entities}
doc = {
    "id": "smoke",
    "text": text,
    "entities": entities,
    "relations": [
        r for r in raw["relations"] if r["from_id"] in ids and r["to_id"] in ids
    ],
}
encoder = ContextEncoder(
    args.encoder, device="cpu", batch_size=2, local_files_only=args.local_files_only
)
feature = encoder.encode(text)
assert torch.isfinite(feature["x"]).all()
ner = BoundaryNER(16, 0.0)
y, _ = boundary_targets(doc, feature["offsets"])
loss = torch.nn.functional.binary_cross_entropy_with_logits(
    ner(feature["x"].float()), y
)
loss.backward()
graph = make_graph(text, entities, feature)
pairs, targets = sample_training_pairs(graph, doc, 42, 2.0)
relation = LegalRGCN(16, 0.0, True)
loss_re = torch.nn.functional.binary_cross_entropy_with_logits(
    relation(graph["x"], graph["edge_index"], graph["edge_type"], pairs), targets
)
loss_re.backward()
assert torch.isfinite(loss) and torch.isfinite(loss_re)
print(
    f"REAL_ENCODER_SMOKE_OK: units={len(feature['offsets'])}, entities={len(entities)}, pairs={len(targets)}, NER_loss={loss.item():.4f}, RE_loss={loss_re.item():.4f}"
)
