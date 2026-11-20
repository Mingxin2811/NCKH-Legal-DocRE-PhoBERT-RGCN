"""Predict entities and relations from raw text using the same trained architectures."""

import argparse
import json
import time
from pathlib import Path

import torch
from context_encoder import ContextEncoder
from data_pipeline import body_text, save_json, normalized_body_hash
from document_io import read_input_document
from labels import ALL_RELATIONS
from model_gnn import BoundaryNER, LegalRGCN
from training import graph_for, load_checkpoint, predict_entities


class Extractor:
    def __init__(self, model_dir, variant="rgcn", device=None, local_files_only=False):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model_dir = Path(model_dir)
        self.model_dir = model_dir
        ner_ckpt = load_checkpoint(model_dir / "checkpoints" / "ner.pt", self.device)
        re_ckpt = load_checkpoint(
            model_dir / "checkpoints" / f"relation_{variant}.pt", self.device
        )
        self.config = ner_ckpt["config"]
        if re_ckpt["config"] != self.config or re_ckpt.get("variant") != variant:
            raise ValueError("NER and relation checkpoints must belong to the same run")
        self.ner = BoundaryNER(self.config["hidden"], self.config["dropout"]).to(
            self.device
        )
        self.relation = LegalRGCN(
            self.config["hidden"], self.config["dropout"], variant == "rgcn"
        ).to(self.device)
        self.ner.load_state_dict(ner_ckpt["model_state_dict"], strict=True)
        self.relation.load_state_dict(re_ckpt["model_state_dict"], strict=True)
        self.ner.eval()
        self.relation.eval()
        self.ner_threshold = ner_ckpt["threshold"]
        self.relation_threshold = re_ckpt["threshold"]
        ec = self.config["encoder_config"]
        self.encoder = ContextEncoder(
            ec["model_name"],
            ec["window"],
            ec["stride"],
            self.config["encoder_batch"],
            self.device,
            ec["revision"],
            local_files_only,
        )
        if self.encoder.config != ec:
            raise ValueError(
                "Encoder/segmentation versions differ from training; use the recorded environment"
            )

    @torch.no_grad()
    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Văn bản đầu vào không được rỗng.")
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        text = body_text(text)
        if not text.strip():
            raise ValueError("Không có phần thân văn bản để nhận diện.")
        feature = self.encoder.encode(text)
        entities = predict_entities(
            self.ner, feature, text, self.device, self.ner_threshold
        )
        if len(entities) > 10000:
            raise ValueError(
                f"NER predicted {len(entities)} entities, exceeding the 10,000-node resource guard. "
                "Inspect NER training/threshold and input quality, or test a shorter document. "
                "No predictions were silently truncated."
            )
        graph = graph_for({"text": text}, feature, self.config, entities)
        relations = []
        if graph["pairs"]:
            h = self.relation.encode_nodes(
                graph["x"].to(self.device),
                graph["edge_index"].to(self.device),
                graph["edge_type"].to(self.device),
            )
            for first in range(0, len(graph["pairs"]), 2048):
                pairs = graph["pairs"][first : first + 2048]
                probabilities = (
                    self.relation.classify(
                        h, torch.tensor(pairs, device=self.device).t()
                    )
                    .sigmoid()
                    .cpu()
                )
                for i, r in (
                    (probabilities >= self.relation_threshold).nonzero().tolist()
                ):
                    u, v = pairs[i]
                    relations.append(
                        {
                            "from_id": entities[u]["id"],
                            "to_id": entities[v]["id"],
                            "code": ALL_RELATIONS[r],
                            "score": float(probabilities[i, r]),
                        }
                    )
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        return {
            "full_text": text,
            "entities": entities,
            "relations": relations,
            "inference": {
                "elapsed_seconds": time.perf_counter()-started,
                "characters":len(text), "lexical_units":len(feature["offsets"]),
                "candidate_pairs":len(graph["pairs"]),
                "ner_threshold":self.ner_threshold,"relation_threshold":self.relation_threshold,
                "device":str(self.device),"timing_scope":"Encoding + NER + RE; excludes model loading, DOCX reading and visualization.",
            },
            "note": "Scores are uncalibrated model outputs; absent candidate pairs are not scored.",
        }

    def predict_file(self, path):
        text, info = read_input_document(path)
        predicted = self.predict(text)
        matches=[]
        manifest_path=self.model_dir/"split_manifest.json"
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
            value=normalized_body_hash(predicted["full_text"])
            ids={i for i,h in manifest.get("body_hashes",{}).items() if h==value}
            matches=[name for name,items in manifest["splits"].items() if ids.intersection(items)]
        info["matching_splits_by_exact_normalized_body"]=matches
        info["evaluation_note"]=(
            "Nội dung trùng với tập "+", ".join(matches)+". Chỉ dùng minh họa; không coi là kiểm tra độc lập mới."
            if matches else "Chưa phát hiện trùng nội dung chuẩn hóa với manifest. Cần tự kiểm tra bản gần trùng; chưa có nhãn chuẩn để chấm điểm."
        )
        info["inference"]=predicted["inference"]
        return predicted, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--text-file")
    source.add_argument("--input-file", help="Read DOCX/TXT or JSON full_text; ignore annotations")
    source.add_argument(
        "--json-file", help="Read full_text only; ignore all supplied annotations"
    )
    source.add_argument("--text")
    p.add_argument("--variant", choices=["semantic", "rgcn"], default="rgcn")
    p.add_argument("--output", default="prediction.json")
    p.add_argument("--device", default=None)
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()
    extractor=Extractor(args.model_dir, args.variant, args.device, args.local_files_only)
    info={}
    if args.input_file:
        result,info=extractor.predict_file(args.input_file)
    elif args.json_file:
        raw = json.loads(Path(args.json_file).read_text(encoding="utf-8-sig"))
        if isinstance(raw, list):
            if len(raw) != 1:
                raise ValueError("Expected a single document")
            raw = raw[0]
        text = raw["full_text"]
    elif args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8-sig")
    else:
        text = args.text
    if not args.input_file:
        result=extractor.predict(text)
    save_json(args.output, result)
    from prediction_view import export_prediction
    report=export_prediction(result,Path(args.output).with_suffix(""),info)
    print(f"Interactive report: {report}")
    print(
        f"Saved {len(result['entities'])} entities, {len(result['relations'])} relations to {args.output}"
    )


if __name__ == "__main__":
    main()
