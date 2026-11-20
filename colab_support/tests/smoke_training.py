"""One-epoch integration check on three real tiny documents, NOT research results."""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from data_pipeline import load_documents
from training import parser, run
from test_thu_cong import Extractor

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
args = p.parse_args()
torch.set_num_threads(2)
documents, _ = load_documents(args.dataset)
small = sorted(documents, key=lambda d: len(d["text"]))[:3]
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for i, d in enumerate(small):
        shutil.copy2(Path(args.dataset) / d["id"], root / f"{i}_ket_qua.json")
    cfg = parser().parse_args(
        [
            "--dataset",
            str(root),
            "--output",
            str(root / "run"),
            "--encoder",
            "vinai/phobert-base",
            "--local-files-only",
            "--device",
            "cpu",
            "--ner-epochs",
            "1",
            "--relation-epochs",
            "1",
            "--hidden",
            "16",
            "--encoder-batch",
            "2",
        ]
    )
    run(cfg)
    for name in [
        "ner_test.json",
        "test_semantic.json",
        "test_rgcn.json",
        "comparison.csv",
    ]:
        assert (root / "run" / name).is_file(), name
    extractor = Extractor(root / "run", device="cpu", local_files_only=True)
    result = extractor.predict(small[0]["text"])
    assert result["full_text"] == small[0]["text"]
    assert isinstance(result["entities"], list) and isinstance(
        result["relations"], list
    )
    print("REAL_DATA_TRAIN_INFER_SMOKE_OK (3 documents; temporary artifacts removed)")
