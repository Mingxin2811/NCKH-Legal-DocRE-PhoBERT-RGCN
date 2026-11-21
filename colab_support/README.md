# Colab development support (not required on Google Drive)

The deliverable is `../google_colab/NCKH_PhoBERT_RGCN_Training.ipynb`.
It contains the runtime implementation directly, retaining the original 11-step presentation.
No development Python modules or tests need to accompany the notebook on Colab.

Training reads `*_ket_qua.json` because it contains `full_text`, entities and relations.
CSV exports of the same document are not loaded a second time. Step 3 displays both file counts.
The loader accepts entity `code`, or an exact schema code/display name in `label` or `type`;
it records conversions in the audit and never assigns a default type to missing/unknown labels.
After replacing the notebook on Colab, restart the session and rerun the definitions so the
old in-memory `load_documents` is replaced. The notebook remains self-contained.

- `rebuild_notebook.py`: generates the standalone notebook from these source modules.
  It removes project imports and CLI entry points with AST transformations.
- `training.py`, `context_encoder.py`, `data_pipeline.py`, `model_gnn.py`, `labels.py`,
  `test_thu_cong.py`: maintainable runtime source. CLI use is still supported here.
- `tests/`: unit tests plus real-encoder/training/standalone-notebook smoke checks.
- `document_io.py`: ordered DOCX body/table extraction and TXT/JSON reading.
- `reporting.py`: measured NER/RE summaries and six training/test PNG figures.
- `prediction_view.py`: full CSV/JSON and offline interactive relation graph/source view.
- `sample_data/`: original sample documents, preserved from the former upload folder.
- `output_results/`: local verification audits, not full experiment results.

From the NCKH project root:

```bash
python colab_support/rebuild_notebook.py
python -m unittest discover -s colab_support/tests -v
python -X utf8 colab_support/tests/smoke_notebook.py --dataset dataset --docx bo_luat/Khong_so_10427/noi_dung.docx
```

The last command needs cached `vinai/phobert-base-v2` weights, uses three small actual documents,
runs one epoch plus plotting/export, reads the real DOCX in full and tests inference on an
explicit 12-character excerpt (one-epoch NER is not trained enough for full-document use).
It removes temporary artifacts unless `--keep-output` is supplied.
It verifies execution, not model quality. Full experiments run on Colab using the delivered notebook.

Pipeline v2 rejects overlapping/old manifests and reused training output directories. RE checkpoint
selection uses end-to-end validation micro-F1; no test-based tuning. `--enclosing 2` adds observable
containment candidates; negative sampling defaults to 5:1. These are new experimental settings,
not a guarantee of higher scores. Exact body grouping does not detect all near-duplicate versions.
The fourth RE positional feature is clipped start-unit index, not document length.
