"""Verify the delivered ipynb works without any of the development .py files."""
import ast
import contextlib
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[2]/"google_colab"/"NCKH_PhoBERT_RGCN_Training.ipynb"


def notebook_cells():
    notebook=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {c["metadata"]["tags"][0]:"".join(c["source"]) for c in notebook["cells"] if c["cell_type"]=="code"}


def notebook_namespace():
    cells=notebook_cells()
    ns={"__name__":"notebook_test"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(cells["labels"],"notebook:labels","exec"),ns)
        exec(compile(cells["pipeline"].split("# AUDIT DATASET")[0],"notebook:pipeline","exec"),ns)
        exec(compile(cells["models"],"notebook:models","exec"),ns)
        exec(compile(cells["training"].split("# RUN TRAINING")[0],"notebook:training","exec"),ns)
    return ns


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells=notebook_cells()
        cls.ns=notebook_namespace()

    def test_preserves_eleven_steps_and_all_cells_compile(self):
        notebook=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        headings="\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"]=="markdown")
        self.assertEqual([int(n) for n in re.findall(r"\[BƯỚC (\d+)\]",headings)],list(range(1,12)))
        self.assertIn("export_prediction",self.cells["inference"])
        for source in self.cells.values():
            compile(source,"notebook","exec")

    def test_no_project_imports_or_subprocess_data_processing(self):
        local={"labels","data_pipeline","context_encoder","training","model_gnn","document_io","prediction_view","reporting"}
        for source in self.cells.values():
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.ImportFrom):
                    self.assertNotIn(node.module,local)
                if isinstance(node,ast.Import):
                    self.assertFalse(any(n.name in local|{"subprocess"} for n in node.names))
            self.assertNotIn("PROJECT_DIR",source)
            self.assertNotIn("build_document_graph.py",source)

    def test_audit_runs_from_empty_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=root/"data";data.mkdir()
            raw={"full_text":"A B\n","entities":[{"id":"a","start":0,"end":1,"text":"A","code":"SO_HIEU"},
                                                          {"id":"b","start":2,"end":3,"text":"B","code":"DIEU"}],
                 "relations":[{"from_id":"a","to_id":"b","code":"THUOC"}]}
            (data/"a_ket_qua.json").write_text(json.dumps(raw),encoding="utf-8")
            for i in range(2):
                another=dict(raw,full_text=f"A B {i}")
                (data/f"extra_{i}_ket_qua.json").write_text(json.dumps(another),encoding="utf-8")
            ns=dict(self.ns,DATASET_DIR=data,OUTPUT_DIR=root/"output")
            old=Path.cwd()
            try:
                os.chdir(root)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(self.cells["pipeline"].split("# AUDIT DATASET")[1],ns)
            finally:
                os.chdir(old)
            self.assertEqual(ns["audit"]["totals"]["documents"],3)
            self.assertTrue((root/"output"/"data_audit.json").is_file())

    def test_data_errors_include_the_filename_without_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"broken_ket_qua.json"
            file.write_text("{invalid",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"broken_ket_qua.json"):
                self.ns["load_documents"](tmp)
            file.write_text("{}",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Missing full_text.*broken_ket_qua.json"):
                self.ns["load_documents"](tmp)

    def test_inference_only_mode_does_not_read_dataset_or_retrain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            ns=dict(self.ns,TRAIN_NEW_MODEL=False,OUTPUT_DIR=root,DATASET_DIR=root/'missing')
            with contextlib.redirect_stdout(io.StringIO()):
                exec(self.cells['pipeline'].split('# AUDIT DATASET')[1],ns)
                exec(self.cells['training'],ns)
            self.assertFalse((root/'run_config.json').exists())
            self.assertFalse((root/'data_audit.json').exists())

    def test_path_checks_explain_missing_or_zipped_dataset(self):
        tree=ast.parse(self.cells["paths"])
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef))
        ns={"Path":Path}
        exec(compile(ast.Module(body=[function],type_ignores=[]),"paths","exec"),ns)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError,"ZIP"):
                ns["check_dataset_path"](tmp)
            with self.assertRaisesRegex(FileNotFoundError,"Bước 3"):
                ns["check_dataset_path"](Path(tmp)/"missing")


if __name__=="__main__":
    unittest.main()
