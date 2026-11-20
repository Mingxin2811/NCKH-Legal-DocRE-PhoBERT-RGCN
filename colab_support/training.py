"""Reproducible frozen-PhoBERT NER + relation training on raw legal documents.

Run --audit-only before training. Test labels are used only for scoring.
"""

import argparse
import bisect
import csv
import json
import random
import importlib.metadata
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_pipeline import (
    PIPELINE_VERSION,
    candidate_pairs,
    load_documents,
    relation_gold,
    save_json,
    set_scores,
    span_key,
    split_documents,
    topology_edges,
)
from labels import ALL_ENTITIES, ALL_RELATIONS
from model_gnn import BoundaryNER, LegalRGCN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_features(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def boundary_targets(doc, offsets):
    y = torch.zeros(len(offsets), 2 * len(ALL_ENTITIES))
    starts = {a: i for i, (a, b) in enumerate(offsets)}
    ends = {b: i for i, (a, b) in enumerate(offsets)}
    unsupported = []
    for e in doc["entities"]:
        if e["start"] not in starts or e["end"] not in ends:
            unsupported.append(e["id"])
            continue
        code = ALL_ENTITIES.index(e["code"])
        y[starts[e["start"]], code] = 1
        y[ends[e["end"]], code + len(ALL_ENTITIES)] = 1
    return y, unsupported


def decode_entities(probabilities, offsets, text, threshold=0.5):
    """Nearest following end for each start/type; can cross encoder windows.

    Cross-type overlap is supported. Same-type nested spans sharing boundaries
    cannot be represented unambiguously; exact-span evaluation exposes this limit.
    """
    entities, seen = [], set()
    ntypes = len(ALL_ENTITIES)
    for code_idx, code in enumerate(ALL_ENTITIES):
        starts = torch.where(probabilities[:, code_idx] >= threshold)[0].tolist()
        ends = torch.where(probabilities[:, code_idx + ntypes] >= threshold)[0].tolist()
        for start in starts:
            loc = bisect.bisect_left(ends, start)
            if loc == len(ends):
                continue
            end = ends[loc]
            a, b = offsets[start][0], offsets[end][1]
            if (a, b, code) not in seen:
                seen.add((a, b, code))
                entities.append(
                    {
                        "start": a,
                        "end": b,
                        "code": code,
                        "text": text[a:b],
                        "score": float(
                            min(
                                probabilities[start, code_idx],
                                probabilities[end, code_idx + ntypes],
                            )
                        ),
                    }
                )
    entities.sort(key=lambda e: (e["start"], e["end"], e["code"]))
    for i, e in enumerate(entities):
        e["id"] = f"P{i}"
    return entities


@torch.no_grad()
def predict_entities(model, feature, text, device, threshold, batch_size=4096):
    model.eval()
    x = feature["x"]
    probs = [
        model(x[i : i + batch_size].float().to(device)).sigmoid().cpu()
        for i in range(0, len(x), batch_size)
    ]
    return decode_entities(
        torch.cat(probs) if probs else torch.empty(0, 2 * len(ALL_ENTITIES)),
        feature["offsets"],
        text,
        threshold,
    )


@torch.no_grad()
def evaluate_ner(model, docs, features, device, threshold):
    gold, pred = set(), set()
    for doc in docs:
        feature = read_features(features[doc["id"]])
        inferred = predict_entities(model, feature, doc["text"], device, threshold)
        gold.update((doc["id"], *span_key(e)) for e in doc["entities"])
        pred.update((doc["id"], *span_key(e)) for e in inferred)
    scores = set_scores(gold, pred)
    scores["per_type"] = {
        code: set_scores(
            {e for e in gold if e[-1] == code}, {e for e in pred if e[-1] == code}
        )
        for code in ALL_ENTITIES
    }
    return scores


def checkpoint_payload(model, config, **extra):
    return {
        "version": PIPELINE_VERSION,
        "model_state_dict": model.state_dict(),
        "config": config,
        "entity_labels": ALL_ENTITIES,
        "relation_labels": ALL_RELATIONS,
        **extra,
    }


def train_ner(splits, features, config, output, device):
    set_seed(config["seed"])
    model = BoundaryNER(config["hidden"], config["dropout"]).to(device)
    counts = torch.zeros(2 * len(ALL_ENTITIES))
    total = 0
    alignment = {}
    train_ids = {doc["id"] for doc in splits["train"]}
    for doc in sum(splits.values(), []):
        feature = read_features(features[doc["id"]])
        y, unsupported = boundary_targets(doc, feature["offsets"])
        if unsupported:
            alignment[doc["id"]] = unsupported
        if doc["id"] in train_ids:
            counts += y.sum(0)
            total += len(y)
    save_json(output / "boundary_alignment_issues.json", alignment)
    # Weights depend only on training data. Missing types remain explicit in reports.
    pos_weight = ((total - counts) / counts.clamp_min(1)).clamp(1, 50).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=0.01
    )
    best, stale, history = -1.0, 0, []
    best_path = output / "checkpoints" / "ner.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(config["ner_epochs"]):
        model.train()
        order = list(splits["train"])
        random.Random(config["seed"] + epoch).shuffle(order)
        total_loss, n = 0.0, 0
        for doc in order:
            feature = read_features(features[doc["id"]])
            y, _ = boundary_targets(doc, feature["offsets"])
            indices = torch.randperm(len(y))
            for start in range(0, len(indices), config["token_batch"]):
                idx = indices[start : start + config["token_batch"]]
                x = feature["x"][idx].float().to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y[idx].to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(idx)
                n += len(idx)
        validation = [
            (threshold, evaluate_ner(model, splits["val"], features, device, threshold))
            for threshold in config["thresholds"]
        ]
        threshold, scores = max(validation, key=lambda item: item[1]["f1"])
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / max(1, n),
                "val_f1": scores["f1"],
                "threshold": threshold,
            }
        )
        print(
            f"NER {epoch + 1}: loss={history[-1]['loss']:.4f}, val exact-span F1={scores['f1']:.4f}",
            flush=True,
        )
        if scores["f1"] > best:
            best, stale = scores["f1"], 0
            torch.save(
                checkpoint_payload(
                    model,
                    config,
                    threshold=threshold,
                    epoch=epoch + 1,
                    validation=scores,
                    kind="ner",
                ),
                best_path,
            )
        else:
            stale += 1
        save_json(output / "ner_history.json", history)
        if stale >= config["patience"]:
            break
    ckpt = load_checkpoint(best_path, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval(), ckpt["threshold"]


def make_graph(text, entities, feature, neighbors=8, anchors=12, enclosing=0):
    """Pool contextual span+surrounding text; graph only sees observable geometry."""
    offsets = feature["offsets"]
    starts, ends = [a for a, b in offsets], [b for a, b in offsets]
    x = torch.zeros(len(entities), 772)
    for i, entity in enumerate(entities):
        a = bisect.bisect_right(ends, entity["start"])
        b = bisect.bisect_left(starts, entity["end"])
        if a >= b:
            raise ValueError(f"Entity has no lexical units: {entity}")
        # Both span content and nearby full-document context, not entity text alone.
        span = feature["x"][a:b, :768].float().mean(0)
        context = (
            feature["x"][max(0, a - 16) : min(len(offsets), b + 16), :768]
            .float()
            .mean(0)
        )
        x[i, :768] = (span + context) / 2
        x[i, 768:] = torch.tensor(
            [
                entity["start"] / max(1, len(text)),
                entity["end"] / max(1, len(text)),
                min(b - a, 1000) / 1000,
                min(a, 1000) / 1000,
            ]
        )
    edges = topology_edges(entities)
    ei = (
        torch.tensor([[u, v] for u, v, r in edges], dtype=torch.long).t().contiguous()
        if edges
        else torch.empty(2, 0, dtype=torch.long)
    )
    et = torch.tensor([r for u, v, r in edges], dtype=torch.long)
    return {
        "x": x,
        "edge_index": ei,
        "edge_type": et,
        "entities": entities,
        "pairs": candidate_pairs(entities, neighbors, anchors, enclosing),
    }


def gold_pair_labels(doc, entities):
    indices = {span_key(e): i for i, e in enumerate(entities)}
    labels = {}
    for u, v, r in relation_gold(doc):
        if u in indices and v in indices:
            labels.setdefault((indices[u], indices[v]), set()).add(r)
    return labels


def sample_training_pairs(graph, doc, seed, neg_ratio):
    positive = gold_pair_labels(doc, graph["entities"])
    negative = [pair for pair in graph["pairs"] if pair not in positive]
    rng = random.Random(seed)
    # Unique pairs; gold positives may be added for TRAINING only.
    rng.shuffle(negative)
    pairs = (
        sorted(positive) + negative[: max(1, int(max(1, len(positive)) * neg_ratio))]
    )
    rng.shuffle(pairs)
    y = torch.zeros(len(pairs), len(ALL_RELATIONS))
    for i, pair in enumerate(pairs):
        for label in positive.get(pair, set()):
            y[i, label] = 1
    return torch.tensor(pairs, dtype=torch.long).t().reshape(2, -1), y


def relation_metrics(tp, fp, support, correct, candidates, covered, gold_pairs_covered):
    p = tp / (tp + fp).clip(min=1)
    r = tp / support.clip(min=1)
    f1 = 2 * p * r / (p + r).clip(min=1e-12)
    active = support > 0
    micro_p = float(tp.sum() / max(1, (tp + fp).sum()))
    micro_r = float(tp.sum() / max(1, support.sum()))
    return {
        "accuracy_candidates_exact_set": correct / max(1, candidates),
        "precision_macro_22": float(p.mean()),
        "recall_macro_22": float(r.mean()),
        "f1_macro_22": float(f1.mean()),
        "f1_macro_supported": float(f1[active].mean()) if active.any() else 0.0,
        "precision_micro_positive": micro_p,
        "recall_micro_positive": micro_r,
        "f1_micro_positive": 2 * micro_p * micro_r / (micro_p + micro_r)
        if micro_p + micro_r
        else 0.0,
        "candidate_recall": covered / max(1, int(support.sum())),
        "candidate_count": candidates,
        "no_relation_baseline_accuracy": (candidates - gold_pairs_covered)
        / max(1, candidates),
        "supported_classes": int(active.sum()),
        "per_class": {
            code: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
            }
            for i, code in enumerate(ALL_RELATIONS)
        },
    }


@torch.no_grad()
def evaluate_relations(model, docs, graphs, device, threshold, batch_size=2048):
    """All deterministic candidates, no sampled test negatives; missed gold counts as FN."""
    model.eval()
    tp, fp, support = (np.zeros(len(ALL_RELATIONS), dtype=np.int64) for _ in range(3))
    correct = candidates = covered = gold_pairs_covered = 0
    for doc in docs:
        graph = graphs[doc["id"]]
        labels = gold_pair_labels(doc, graph["entities"])
        for u, v, r in relation_gold(doc):
            support[r] += 1
        pair_set = set(graph["pairs"])
        covered += sum(len(rs) for pair, rs in labels.items() if pair in pair_set)
        gold_pairs_covered += sum(pair in pair_set for pair in labels)
        if not graph["pairs"]:
            continue
        h = model.encode_nodes(
            graph["x"].to(device),
            graph["edge_index"].to(device),
            graph["edge_type"].to(device),
        )
        for first in range(0, len(graph["pairs"]), batch_size):
            pairs = graph["pairs"][first : first + batch_size]
            pair_tensor = torch.tensor(pairs, dtype=torch.long, device=device).t()
            pred = (model.classify(h, pair_tensor).sigmoid() >= threshold).cpu().numpy()
            target = np.zeros_like(pred)
            for i, pair in enumerate(pairs):
                for r in labels.get(pair, set()):
                    target[i, r] = True
            tp += (pred & target).sum(0)
            fp += (pred & ~target).sum(0)
            correct += int((pred == target).all(1).sum())
            candidates += len(pairs)
    return relation_metrics(
        tp, fp, support, correct, candidates, covered, gold_pairs_covered
    )


def train_relations(splits, graphs, config, output, device, variant, validation_graphs=None):
    set_seed(config["seed"])
    model = LegalRGCN(config["hidden"], config["dropout"], variant == "rgcn").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=0.01
    )
    counts = Counter(r["code"] for d in splits["train"] for r in d["relations"])
    total = sum(counts.values())
    # Equal class contribution, tempered weights for rare positives. No test statistics.
    weights = torch.tensor(
        [
            min(20.0, max(1.0, total / max(1, counts[c]) / len(ALL_RELATIONS)))
            for c in ALL_RELATIONS
        ],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    best, stale, history = -1.0, 0, []
    best_path = output / "checkpoints" / f"relation_{variant}.pt"
    for epoch in range(config["relation_epochs"]):
        model.train()
        order = list(splits["train"])
        random.Random(config["seed"] + epoch).shuffle(order)
        losses = []
        for i, doc in enumerate(order):
            graph = graphs[doc["id"]]
            pairs, y = sample_training_pairs(
                graph, doc, config["seed"] + epoch * len(order) + i, config["neg_ratio"]
            )
            if not len(y):
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                graph["x"].to(device),
                graph["edge_index"].to(device),
                graph["edge_type"].to(device),
                pairs.to(device),
            )
            loss = criterion(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        validation = [
            (t, evaluate_relations(model, splits["val"], validation_graphs if validation_graphs is not None else graphs, device, t))
            for t in config["thresholds"]
        ]
        selection_metric = config.get("selection_metric", "f1_macro_22")
        threshold, scores = max(validation, key=lambda x: x[1][selection_metric])
        history.append(
            {
                "epoch": epoch + 1,
                "loss": sum(losses) / max(1, len(losses)),
                "val_f1_macro_22": scores["f1_macro_22"],
                "val_f1_micro_positive": scores["f1_micro_positive"],
                "val_precision_micro": scores["precision_micro_positive"],
                "val_recall_micro": scores["recall_micro_positive"],
                "selection_metric": selection_metric,
                "threshold": threshold,
            }
        )
        print(
            f"RE {variant} {epoch + 1}: loss={history[-1]['loss']:.4f}, val {selection_metric}={scores[selection_metric]:.4f}",
            flush=True,
        )
        if scores[selection_metric] > best:
            best, stale = scores[selection_metric], 0
            torch.save(
                checkpoint_payload(
                    model,
                    config,
                    threshold=threshold,
                    epoch=epoch + 1,
                    validation=scores,
                    kind="relation",
                    variant=variant,
                ),
                best_path,
            )
        else:
            stale += 1
        save_json(output / f"relation_{variant}_history.json", history)
        if stale >= config["patience"]:
            break
    ckpt = load_checkpoint(best_path, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval(), ckpt["threshold"]


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (
        checkpoint.get("version") != PIPELINE_VERSION
        or checkpoint.get("entity_labels") != ALL_ENTITIES
        or checkpoint.get("relation_labels") != ALL_RELATIONS
    ):
        raise ValueError("Incompatible checkpoint/schema; retrain using this pipeline.")
    return checkpoint


def graph_for(doc, feature, config, entities=None):
    return make_graph(
        doc["text"],
        doc["entities"] if entities is None else entities,
        feature,
        config["neighbors"],
        config["anchors"],
        config.get("enclosing", 0),
    )


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_config.json").exists():
        raise ValueError("Output already contains a training run. Choose a new OUTPUT_DIR; old results stay unchanged.")
    started = time.perf_counter()
    documents, audit = load_documents(args.dataset)
    save_json(output / "data_audit.json", audit)
    print(json.dumps(audit["totals"], ensure_ascii=False, indent=2), flush=True)
    if args.audit_only:
        return
    save_json(output / "run_status.json", {"status": "training", "version": PIPELINE_VERSION})
    if (
        min(
            args.ner_epochs,
            args.relation_epochs,
            args.hidden,
            args.token_batch,
            args.encoder_batch,
            args.patience,
        )
        < 1
    ):
        raise ValueError("Epochs, dimensions, batches and patience must be positive")
    if (
        args.neg_ratio <= 0
        or min(args.neighbors, args.anchors, args.enclosing) < 0
        or not 0 <= args.dropout < 1
        or args.lr <= 0
    ):
        raise ValueError("Invalid sampling/model hyperparameters")
    if not args.variants or any(
        v not in {"semantic", "rgcn"} for v in args.variants.split(",")
    ):
        raise ValueError("Variants must be semantic,rgcn (or either one)")
    splits = split_documents(documents, output / "split_manifest.json", args.seed)
    support = {
        name: dict(Counter(r["code"] for d in docs for r in d["relations"]))
        for name, docs in splits.items()
    }
    save_json(output / "split_relation_support.json", support)
    for name in ("val", "test"):
        active = len(support[name])
        print(f"{name}: {len(splits[name])} documents; {active}/22 supported relations; macro-F1 ceiling <= {active/22:.2%}", flush=True)
        if not active:
            print(f"WARNING: {name} has no positive relations. This run can check execution but cannot establish RE quality.", flush=True)
    save_json(output / "split_summary.json", {
        "counts": {name:len(docs) for name,docs in splits.items()},
        "ratios_requested": {"train":0.8,"val":0.1,"test":0.1},
        "seed":args.seed,
        "note":"Grouped identical normalized bodies; not label-stratified. Review rare-label support and near-duplicate versions manually.",
    })
    save_json(
        output / "environment.json",
        {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "torch-geometric",
                "transformers",
                "pyvi",
                "numpy",
                "scikit-learn",
            ]
        },
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(args.seed)
    from context_encoder import ContextEncoder

    encoder = ContextEncoder(
        args.encoder,
        window=args.window,
        stride=args.stride,
        batch_size=args.encoder_batch,
        device=device,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    config = vars(args).copy()
    config["encoder_config"] = encoder.config
    config["thresholds"] = [0.2, 0.35, 0.5, 0.65, 0.8]
    config["relation_validation_setting"] = "predicted_entities_end_to_end"
    config["feature_note"] = "RE positions: relative start/end, clipped span unit count, clipped START unit index (not document length)."
    save_json(output / "run_config.json", config)
    features = {}
    for i, doc in enumerate(documents):
        encoded = encoder.encode(doc["text"], output / "cache")
        features[doc["id"]] = str(output / "cache" / f"{encoded['cache_key']}.pt")
        print(
            f"Embedding {i + 1}/{len(documents)}: {doc['id']} ({len(encoded['offsets'])} units)",
            flush=True,
        )
    del encoded, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ner, ner_threshold = train_ner(splits, features, config, output, device)
    ner_test = evaluate_ner(ner, splits["test"], features, device, ner_threshold)
    save_json(output / "ner_test.json", ner_test)
    graphs = {
        doc["id"]: graph_for(doc, read_features(features[doc["id"]]), config)
        for doc in documents
    }
    inferred_graphs = {}
    for doc in splits["val"] + splits["test"]:
        feature = read_features(features[doc["id"]])
        inferred = predict_entities(ner, feature, doc["text"], device, ner_threshold)
        inferred_graphs[doc["id"]] = graph_for(doc, feature, config, inferred)
    results = []
    for variant in args.variants.split(","):
        if variant not in {"semantic", "rgcn"}:
            raise ValueError(f"Unknown variant: {variant}")
        relation, threshold = train_relations(
            splits, graphs, config, output, device, variant, validation_graphs=inferred_graphs
        )
        oracle = evaluate_relations(relation, splits["test"], graphs, device, threshold)
        end_to_end = evaluate_relations(
            relation, splits["test"], inferred_graphs, device, threshold
        )
        save_json(
            output / f"test_{variant}.json",
            {"gold_entities": oracle, "predicted_entities_end_to_end": end_to_end},
        )
        for setting, metrics in [
            ("gold_entities", oracle),
            ("predicted_entities_end_to_end", end_to_end),
        ]:
            results.append(
                {
                    "variant": variant,
                    "setting": setting,
                    **{k: v for k, v in metrics.items() if k != "per_class"},
                }
            )
    if results:
        with (output / "comparison.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    selected = {}
    for name in ["ner"] + [f"relation_{v}" for v in args.variants.split(",")]:
        saved = load_checkpoint(output/"checkpoints"/f"{name}.pt", "cpu")
        selected[name] = {"epoch":saved["epoch"], "threshold":saved["threshold"],
                         "validation":saved["validation"]}
    save_json(output/"selected_checkpoints.json", selected)
    save_json(output / "run_status.json", {"status":"complete", "version":PIPELINE_VERSION,
        "elapsed_seconds": time.perf_counter()-started, "device":str(device),
        "device_name":torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU"})
    print(f"Finished. Actual measured results: {output}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="output_results/raw_doc_v1")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--encoder", default="vinai/phobert-base-v2")
    p.add_argument("--revision", default="main")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--encoder-batch", type=int, default=8)
    p.add_argument("--token-batch", type=int, default=2048)
    p.add_argument("--ner-epochs", type=int, default=20)
    p.add_argument("--relation-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--neg-ratio", type=float, default=5.0)
    p.add_argument("--neighbors", type=int, default=8)
    p.add_argument("--anchors", type=int, default=12)
    p.add_argument("--enclosing", type=int, default=2)
    p.add_argument("--selection-metric", choices=["f1_micro_positive", "f1_macro_22"], default="f1_micro_positive")
    p.add_argument("--variants", default="semantic,rgcn")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
