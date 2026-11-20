"""Validated raw-document inputs; labels are never graph features."""

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from labels import ALL_ENTITIES, ALL_RELATIONS, RELATION_ALIAS_MAP

FOOTER_MARKER = "--- THÔNG TIN THUỘC TÍNH & LƯỢC ĐỒ TỪ FILE JSON ---"
PIPELINE_VERSION = "raw-doc-v2"


def normalized_body_hash(text):
    return digest(" ".join(unicodedata.normalize("NFC", body_text(text)).split()))


def body_text(text):
    # Do not strip trailing whitespace: annotations may include the final newline.
    return text.split(FOOTER_MARKER, 1)[0]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_documents(root):
    """Require JSON full_text; CSV duplicates are not separate documents.

    Keep exact original offsets. Invalid/ambiguous spans are reported, not guessed.
    """
    root = Path(root)
    paths = sorted(root.rglob("*_ket_qua.json"))
    if not paths:
        raise ValueError(f"No *_ket_qua.json with full_text in {root}")
    documents, issues, totals = [], [], Counter()
    entity_counts, relation_counts, relation_docs = Counter(), Counter(), Counter()
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read dataset JSON {path}: {exc}") from exc
        if isinstance(raw, list):
            if len(raw) != 1:
                raise ValueError(f"Expected one document: {path}")
            raw = raw[0]
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a document object: {path}")
        if not isinstance(raw.get("entities", []), list) or not isinstance(raw.get("relations", []), list):
            raise ValueError(f"entities and relations must be lists: {path}")
        if not isinstance(raw.get("full_text"), str) or not raw["full_text"].strip():
            raise ValueError(f"Missing full_text: {path}")
        original = raw["full_text"]
        text = body_text(original)
        if not text.strip():
            raise ValueError(f"Missing original document body: {path}")
        totals["documents"] += 1
        totals["documents_with_synthetic_footer"] += FOOTER_MARKER in original
        entities, ids = [], set()
        for entity in raw.get("entities", []):
            totals["input_entities"] += 1
            if not isinstance(entity, dict) or not entity.get("id"):
                raise ValueError(f"Entity missing ID or invalid entity object: {path}")
            eid = entity["id"]
            if eid in ids:
                raise ValueError(f"Duplicate entity ID {eid}: {path}")
            ids.add(eid)
            code = entity.get("code")
            if code not in ALL_ENTITIES:
                raise ValueError(f"Unknown entity code {code}: {path}")
            start, end = entity.get("start"), entity.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                reason = "invalid_offsets"
            elif start >= len(text) and start < len(original):
                reason = "synthetic_footer"
            elif not 0 <= start < end <= len(text):
                reason = "invalid_or_cross_footer_offsets"
            elif text[start:end] != entity.get("text"):
                reason = "text_offset_mismatch"
            elif not entity["text"].strip():
                reason = "empty_entity_text"
            else:
                # Canonical boundaries ignore surrounding whitespace, consistently
                # in training and exact-span scoring; preserve the original dataset.
                literal = entity["text"]
                clean_start = start + len(literal) - len(literal.lstrip())
                clean_end = end - (len(literal) - len(literal.rstrip()))
                if (clean_start, clean_end) != (start, end):
                    totals["entities_whitespace_boundary_trimmed"] += 1
                    issues.append(
                        {
                            "file": str(path.relative_to(root)),
                            "entity_id": eid,
                            "reason": "normalized_surrounding_whitespace",
                            "original_span": [start, end],
                            "normalized_span": [clean_start, clean_end],
                        }
                    )
                entities.append(
                    dict(
                        entity,
                        start=clean_start,
                        end=clean_end,
                        text=text[clean_start:clean_end],
                    )
                )
                entity_counts[code] += 1
                continue
            totals[f"excluded_entities_{reason}"] += 1
            issues.append(
                {
                    "file": str(path.relative_to(root)),
                    "entity_id": eid,
                    "reason": reason,
                }
            )
        kept_ids = {e["id"] for e in entities}
        relations, seen = [], set()
        for relation in raw.get("relations", []):
            totals["input_relations"] += 1
            if not isinstance(relation, dict):
                raise ValueError(f"Invalid relation object: {path}")
            code = (
                str(relation.get("code") or relation.get("label") or "").strip().upper()
            )
            if code not in RELATION_ALIAS_MAP:
                raise ValueError(f"Unknown relation code {code}: {path}")
            canonical = RELATION_ALIAS_MAP[code]
            key = (relation.get("from_id"), relation.get("to_id"), canonical)
            if key[0] not in ids or key[1] not in ids:
                raise ValueError(f"Dangling relation {key}: {path}")
            if key[0] not in kept_ids or key[1] not in kept_ids:
                totals["excluded_relations_missing_body_entity"] += 1
                continue
            if key in seen:
                totals["duplicate_relations_removed"] += 1
                continue
            seen.add(key)
            relations.append({"from_id": key[0], "to_id": key[1], "code": canonical})
            relation_counts[canonical] += 1
        relation_docs.update({r["code"] for r in relations})
        documents.append(
            {
                "id": path.relative_to(root).as_posix(),
                "text": text,
                "entities": entities,
                "relations": relations,
            }
        )
    totals["retained_entities"] = sum(entity_counts.values())
    totals["retained_relations"] = sum(relation_counts.values())
    report = {
        "version": PIPELINE_VERSION,
        "totals": dict(totals),
        "entity_counts": {k: entity_counts[k] for k in ALL_ENTITIES},
        "relation_counts": {k: relation_counts[k] for k in ALL_RELATIONS},
        "relation_document_support": {k: relation_docs[k] for k in ALL_RELATIONS},
        "issues": issues,
        "note": "Supplied annotations are not certified as human-reviewed gold.",
    }
    return documents, report


def split_documents(documents, output, seed=42):
    """Persist splits; identical normalized bodies stay together; reject stale manifests."""
    output = Path(output)
    ids = [d["id"] for d in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs: use relative paths, not basenames.")
    fingerprints = {
        d["id"]: digest(json.dumps(d, sort_keys=True, ensure_ascii=False))
        for d in documents
    }
    if output.exists():
        manifest = json.loads(output.read_text(encoding="utf-8"))
        if manifest.get("version") != PIPELINE_VERSION:
            raise ValueError("Old pipeline manifest: choose a new output directory.")
        if manifest["fingerprints"] != fingerprints or manifest["seed"] != seed:
            raise ValueError("Dataset/seed changed: choose a new output directory.")
    else:
        groups = defaultdict(list)
        for doc in documents:
            groups[
                digest(" ".join(unicodedata.normalize("NFC", doc["text"]).split()))
            ].append(doc["id"])
        keys = sorted(groups)
        if len(keys) < 3:
            raise ValueError(
                "Need >=3 distinct bodies for train/val/test; sample_data is smoke-test only."
            )
        random.Random(seed).shuffle(keys)
        nv = max(1, int(len(keys) * 0.1))
        nt = max(1, int(len(keys) * 0.1))
        partitions = {
            "val": keys[:nv],
            "test": keys[nv : nv + nt],
            "train": keys[nv + nt :],
        }
        manifest = {
            "version": PIPELINE_VERSION,
            "seed": seed,
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "body_hashes": {d["id"]: normalized_body_hash(d["text"]) for d in documents},
            "fingerprints": fingerprints,
            "splits": {
                k: [i for group in v for i in groups[group]]
                for k, v in partitions.items()
            },
        }
        save_json(output, manifest)
    by_id = {d["id"]: d for d in documents}
    partitions = manifest["splits"]
    if set(partitions) != {"train", "val", "test"}:
        raise ValueError("Invalid split manifest: require train/val/test")
    assigned = [i for part in partitions.values() for i in part]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(by_id):
        raise ValueError("Split overlap/missing documents: choose a new output directory.")
    hashes = []
    for name in ("train", "val", "test"):
        if not partitions[name]:
            raise ValueError("Each split must contain at least one document.")
        hashes.append({normalized_body_hash(by_id[i]["text"]) for i in partitions[name]})
    if any(a & b for n, a in enumerate(hashes) for b in hashes[n + 1:]):
        raise ValueError("Identical document bodies occur across splits.")
    return {k: [by_id[i] for i in ids] for k, ids in manifest["splits"].items()}


def atoms(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE)]


def span_key(entity):
    return entity["start"], entity["end"], entity["code"]


def relation_gold(doc):
    by_id = {e["id"]: e for e in doc["entities"]}
    return {
        (
            span_key(by_id[r["from_id"]]),
            span_key(by_id[r["to_id"]]),
            ALL_RELATIONS.index(r["code"]),
        )
        for r in doc["relations"]
    }


def candidate_pairs(entities, neighbors=8, anchors=12, enclosing=0):
    """Offset-only directed neighbors and earliest anchors; O(N*(neighbors+anchors)).

    No target labels, IDs, or relation-encoded entity types are consulted.
    Gold relations outside this candidate set must still count as false negatives.
    """
    order = sorted(
        range(len(entities)),
        key=lambda i: (entities[i]["start"], entities[i]["end"], i),
    )
    pairs = set()
    for rank, i in enumerate(order):
        for j in order[max(0, rank - neighbors) : rank + neighbors + 1]:
            if i != j:
                pairs.add((i, j))
    for i in order[:anchors]:
        for j in order:
            if i != j:
                pairs.update(((i, j), (j, i)))
    if enclosing:
        # Geometry only: nearest enclosing spans are observable at inference too.
        active = []
        containment_order = sorted(order, key=lambda i: (entities[i]["start"], -entities[i]["end"]))
        for i in containment_order:
            e = entities[i]
            active = [j for j in active if entities[j]["end"] >= e["end"] or entities[j]["end"] > e["start"]]
            parents = [j for j in active if entities[j]["end"] >= e["end"]
                       and (entities[j]["start"], entities[j]["end"]) != (e["start"], e["end"])]
            parents.sort(key=lambda j: (entities[j]["end"]-entities[j]["start"], j))
            for j in parents[:enclosing]:
                pairs.update(((i,j),(j,i)))
            active.append(i)
    return sorted(pairs)


def topology_edges(entities):
    """Observable types: adjacency forward/backward and overlap forward/backward."""
    order = sorted(
        range(len(entities)),
        key=lambda i: (entities[i]["start"], entities[i]["end"], i),
    )
    edges, active = set(), []
    for rank, i in enumerate(order):
        if rank:
            j = order[rank - 1]
            edges.update(((j, i, 0), (i, j, 1)))
        active = [j for j in active if entities[j]["end"] > entities[i]["start"]]
        for j in active:
            edges.update(((j, i, 2), (i, j, 3)))
            if len(edges) > 500000:
                raise ValueError("Graph exceeds 500,000 positional edges. Inspect excessive/overlapping NER spans or use shorter documents; edges are not silently dropped.")
        active.append(i)
    return sorted(edges)


def set_scores(gold, predicted):
    tp = len(gold & predicted)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "tp": tp,
        "fp": len(predicted) - tp,
        "fn": len(gold) - tp,
    }
