import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from data_pipeline import (
    FOOTER_MARKER,
    atoms,
    load_documents,
    save_json,
    split_documents,
)
from labels import ALL_RELATIONS
from model_gnn import LegalRGCN
from training import (
    boundary_targets,
    checkpoint_payload,
    decode_entities,
    evaluate_relations,
    load_checkpoint,
    make_graph,
    sample_training_pairs,
)


def doc(text="A B C"):
    return {
        "id": "d",
        "text": text,
        "entities": [
            {"id": "a", "start": 0, "end": 1, "code": "SO_HIEU", "text": text[:1]},
            {"id": "b", "start": 2, "end": 3, "code": "DIEU", "text": text[2:3]},
        ],
        "relations": [{"from_id": "a", "to_id": "b", "code": "THUOC"}],
    }


class PipelineTests(unittest.TestCase):
    def test_footer_and_invalid_offsets_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = doc()
            raw["full_text"] = "A B C\n" + FOOTER_MARKER + "\nX"
            raw["entities"].append(
                {
                    "id": "f",
                    "start": len(raw["full_text"]) - 1,
                    "end": len(raw["full_text"]),
                    "text": "X",
                    "code": "VB_AP_DUNG",
                }
            )
            raw["relations"].append({"from_id": "a", "to_id": "f", "code": "AP_DUNG"})
            save_json(Path(tmp) / "a_ket_qua.json", raw)
            docs, audit = load_documents(tmp)
            self.assertEqual(docs[0]["text"], "A B C\n")
            self.assertEqual(len(docs[0]["relations"]), 1)
            self.assertEqual(audit["totals"]["excluded_entities_synthetic_footer"], 1)
            raw["entities"][0]["text"] = "WRONG"
            save_json(Path(tmp) / "a_ket_qua.json", raw)
            docs, audit = load_documents(tmp)
            self.assertEqual(
                audit["totals"]["excluded_entities_text_offset_mismatch"], 1
            )
            self.assertFalse(docs[0]["relations"])

    def test_unknown_relation_fails_instead_of_becoming_thuoc(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = doc()
            raw["full_text"] = raw["text"]
            raw["relations"][0]["code"] = "TYPO"
            save_json(Path(tmp) / "a_ket_qua.json", raw)
            with self.assertRaisesRegex(ValueError, "Unknown relation"):
                load_documents(tmp)

    def test_surrounding_whitespace_has_consistent_canonical_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = doc()
            raw["full_text"] = " A B C\n"
            raw["entities"] = [
                {"id": "a", "start": 0, "end": 7, "text": " A B C\n", "code": "DIEU"}
            ]
            raw["relations"] = []
            save_json(Path(tmp) / "a_ket_qua.json", raw)
            docs, audit = load_documents(tmp)
            self.assertEqual(
                (docs[0]["entities"][0]["start"], docs[0]["entities"][0]["end"]), (1, 6)
            )
            self.assertEqual(audit["totals"]["entities_whitespace_boundary_trimmed"], 1)

    def test_split_is_disjoint_persistent_and_groups_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = [dict(doc(f"A B C {i}"), id=str(i)) for i in range(10)]
            docs.append(dict(docs[0], id="duplicate"))
            manifest = Path(tmp) / "split.json"
            a = split_documents(docs, manifest)
            b = split_documents(list(reversed(docs)), manifest)
            self.assertEqual(a, b)
            groups = [{d["id"] for d in v} for v in a.values()]
            self.assertTrue(
                all(not x & y for i, x in enumerate(groups) for y in groups[i + 1 :])
            )
            self.assertTrue(any({"0", "duplicate"} <= s for s in groups))
            with self.assertRaisesRegex(ValueError, "Dataset/seed changed"):
                split_documents(docs[:-1], manifest)
            with self.assertRaisesRegex(ValueError, "distinct bodies"):
                split_documents(docs[:2], Path(tmp) / "small.json")

    def test_observable_graph_does_not_use_types_or_gold_relations(self):
        d = doc()
        feature = {"x": torch.randn(3, 772), "offsets": atoms(d["text"])}
        a = make_graph(d["text"], d["entities"], feature)
        altered = [dict(e, code="VB_GIAI_THICH", id="changed") for e in d["entities"]]
        b = make_graph(d["text"], altered, feature)
        for key in ["x", "edge_index", "edge_type"]:
            self.assertTrue(torch.equal(a[key], b[key]))
        self.assertEqual(a["pairs"], b["pairs"])
        self.assertTrue(set(a["edge_type"].tolist()) <= {0, 1, 2, 3})

    def test_multilabel_pairs_and_negative_uniqueness(self):
        d = doc()
        d["relations"].append({"from_id": "a", "to_id": "b", "code": "DAN_CHIEU"})
        feature = {"x": torch.randn(3, 772), "offsets": atoms(d["text"])}
        graph = make_graph(d["text"], d["entities"], feature)
        pairs, y = sample_training_pairs(graph, d, 42, 2.0)
        self.assertEqual(len(set(map(tuple, pairs.t().tolist()))), pairs.shape[1])
        index = pairs.t().tolist().index([0, 1])
        self.assertEqual(int(y[index].sum()), 2)
        self.assertTrue(torch.equal(y, sample_training_pairs(graph, d, 42, 2.0)[1]))

    def test_missing_candidates_still_count_as_false_negatives(self):
        class NegativeModel:
            def eval(self):
                return self

            def encode_nodes(self, x, *args):
                return x

            def classify(self, h, pairs):
                return torch.full((pairs.shape[1], len(ALL_RELATIONS)), -20.0)

        d = doc()
        feature = {"x": torch.randn(3, 772), "offsets": atoms(d["text"])}
        graph = make_graph(d["text"], d["entities"], feature)
        graph["pairs"] = [(1, 0)]
        scores = evaluate_relations(NegativeModel(), [d], {"d": graph}, "cpu", 0.5)
        self.assertEqual(scores["candidate_recall"], 0.0)
        self.assertEqual(scores["per_class"]["THUOC"]["support"], 1)
        self.assertEqual(scores["recall_micro_positive"], 0.0)
        self.assertEqual(scores["accuracy_candidates_exact_set"], 1.0)

    def test_overlapping_types_and_long_span_decode(self):
        text = "A B C D E"
        offsets = atoms(text)
        d = doc(text)
        d["entities"] = [
            {"id": "a", "start": 0, "end": len(text), "code": "TEN_VAN_BAN"},
            {"id": "b", "start": 0, "end": 1, "code": "LOAI_VAN_BAN"},
        ]
        targets, unsupported = boundary_targets(d, offsets)
        self.assertFalse(unsupported)
        decoded = decode_entities(targets, offsets, text, 0.5)
        self.assertEqual(
            {(e["start"], e["end"], e["code"]) for e in decoded},
            {(0, len(text), "TEN_VAN_BAN"), (0, 1, "LOAI_VAN_BAN")},
        )

    def test_shared_checkpoint_and_empty_graph(self):
        torch.manual_seed(42)
        config = {"hidden": 16, "dropout": 0.0}
        for use_gnn in [False, True]:
            model = LegalRGCN(16, 0.0, use_gnn).eval()
            x = torch.randn(2, 772)
            ei = torch.empty(2, 0, dtype=torch.long)
            et = torch.empty(0, dtype=torch.long)
            pairs = torch.tensor([[0], [1]])
            expected = model(x, ei, et, pairs)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.pt"
                torch.save(checkpoint_payload(model, config), path)
                other = LegalRGCN(16, 0.0, use_gnn).eval()
                other.load_state_dict(
                    load_checkpoint(path)["model_state_dict"], strict=True
                )
                self.assertTrue(torch.allclose(expected, other(x, ei, et, pairs)))
            self.assertEqual(
                model(torch.empty(0, 772), ei, et, ei).shape, (0, len(ALL_RELATIONS))
            )

    def test_subword_windows_cover_entire_document_without_truncation(self):
        from context_encoder import ContextEncoder

        class Tokenizer:
            pad_token_id = 1
            bos_token_id = 0
            eos_token_id = 2
            unk_token_id = 3

            def encode(self, word, add_special_tokens=False):
                return [int(word)] * 3

        class Encoder:
            def __call__(self, input_ids, attention_mask):
                self.lengths = getattr(self, "lengths", []) + [input_ids.shape[1]]
                return SimpleNamespace(
                    last_hidden_state=input_ids.float().unsqueeze(-1).repeat(1, 1, 768)
                )

        encoder = ContextEncoder.__new__(ContextEncoder)
        encoder.device = torch.device("cpu")
        encoder.config = {"test": True}
        encoder.window = 8
        encoder.stride = 3
        encoder.batch_size = 2
        encoder.tokenizer = Tokenizer()
        encoder.model = Encoder()
        text = "10 20 30 40 50"
        with patch(
            "context_encoder.segmented_words",
            return_value=[(text[a:b], a, b) for a, b in atoms(text)],
        ):
            feature = encoder.encode(text)
        self.assertTrue(all(n <= 8 for n in encoder.model.lengths))
        self.assertEqual(
            feature["x"][:, :768].float().mean(1).tolist(),
            [10.0, 20.0, 30.0, 40.0, 50.0],
        )

    def test_lexical_units_split_at_segmentation_boundaries(self):
        from context_encoder import aligned_units, segmented_words

        self.assertEqual(
            aligned_units("2,5a", [("2,5", 0, 3), ("a", 3, 4)]),
            [(0, 1), (1, 2), (2, 3), (3, 4)],
        )
        self.assertEqual(segmented_words("____"), [("____", 0, 4)])


if __name__ == "__main__":
    unittest.main()
