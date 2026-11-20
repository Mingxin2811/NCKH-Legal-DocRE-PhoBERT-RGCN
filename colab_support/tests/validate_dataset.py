"""Validate real corpus segmentation/alignment without downloading model weights."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_encoder import segmented_words, aligned_units
from data_pipeline import load_documents, save_json

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()
docs, _ = load_documents(args.dataset)
failures = []
unalignable = {}
for i, doc in enumerate(docs):
    try:
        words = segmented_words(doc["text"])
        offsets = aligned_units(doc["text"], words)
        j = 0
        for a, b in offsets:
            while j + 1 < len(words) and words[j][2] <= a:
                j += 1
            if not words[j][1] <= a < b <= words[j][2]:
                raise ValueError(f"Atom {(a, b)} crosses segmented word {words[j]}")
        starts = {a for a, b in offsets}
        ends = {b for a, b in offsets}
        missing = [
            e["id"]
            for e in doc["entities"]
            if e["start"] not in starts or e["end"] not in ends
        ]
        if missing:
            unalignable[doc["id"]] = missing
    except ValueError as exc:
        failures.append({"id": doc["id"], "error": str(exc)})
    if (i + 1) % 20 == 0:
        print(f"Checked {i + 1}/{len(docs)}; failures={len(failures)}", flush=True)
save_json(
    args.output,
    {
        "documents": len(docs),
        "segmentation_failures": failures,
        "unalignable_entity_boundaries": unalignable,
    },
)
print(
    f"Finished {len(docs)} documents; failures={len(failures)}; report={args.output}",
    flush=True,
)
if failures:
    sys.exit(1)
