"""Dataset-format regression tests; no model download or GPU required."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline import load_documents
from labels import ENTITY_TAXONOMY, RELATION_TAXONOMY


class DatasetSchemaTests(unittest.TestCase):
    def load(self, entity, relation=None):
        with tempfile.TemporaryDirectory() as tmp:
            raw = {"full_text": "A B", "entities": [dict(id="a", start=0, end=1, text="A", **entity)], "relations": []}
            if relation is not None:
                raw["entities"].append(dict(id="b", start=2, end=3, text="B", code="DIEU"))
                raw["relations"] = [dict(from_id="a", to_id="b", **relation)]
            path = Path(tmp)/"drive_document_ket_qua.json"
            path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            # A CSV export is not a second training document.
            path.with_suffix(".csv").write_text("ENTITY,a,SO_HIEU", encoding="utf-8")
            return load_documents(tmp)

    def test_code_and_display_label_remain_compatible(self):
        docs, audit = self.load(dict(code="SO_HIEU", label="Display name"))
        self.assertEqual(docs[0]["entities"][0]["code"], "SO_HIEU")
        self.assertEqual(audit["totals"]["documents"], 1)

    def test_missing_code_accepts_label_code_and_human_name(self):
        for value in ("SO_HIEU", ENTITY_TAXONOMY["SO_HIEU"][1]):
            with self.subTest(value=value):
                docs, audit = self.load(dict(label=value))
                self.assertEqual(docs[0]["entities"][0]["code"], "SO_HIEU")
                self.assertEqual(audit["totals"]["entities_code_from_label"], 1)

    def test_type_and_single_labels_list_are_supported(self):
        for entity in (dict(type="SO_HIEU"), dict(labels=["SO_HIEU"]), dict(code=None, label="SO_HIEU")):
            with self.subTest(entity=entity):
                docs, _ = self.load(entity)
                self.assertEqual(docs[0]["entities"][0]["code"], "SO_HIEU")

    def test_never_guesses_missing_unknown_or_conflicting_labels(self):
        for entity in ({}, dict(label="NOT_A_LABEL"), dict(code="TYPO", label="SO_HIEU"), dict(label="SO_HIEU", type="DIEU"), dict(labels=["DIEU", "SO_HIEU"])):
            with self.subTest(entity=entity):
                with self.assertRaisesRegex(ValueError, "drive_document_ket_qua.json"):
                    self.load(entity)

    def test_relation_human_label_and_alias_are_canonicalized(self):
        for relation in (dict(label=RELATION_TAXONOMY["THUOC"][1]), dict(type="THUOC"), dict(code="THUOC")):
            with self.subTest(relation=relation):
                docs, _ = self.load(dict(type="SO_HIEU"), relation)
                self.assertEqual(docs[0]["relations"][0]["code"], "THUOC")


if __name__ == "__main__":
    unittest.main()
