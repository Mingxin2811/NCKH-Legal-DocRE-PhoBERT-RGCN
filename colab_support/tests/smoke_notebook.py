"""Execute actual standalone notebook definitions, training, plots and raw inference."""
import argparse
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import torch
from test_notebook import notebook_cells,notebook_namespace

p=argparse.ArgumentParser()
p.add_argument("--dataset",required=True)
p.add_argument("--docx",required=True)
p.add_argument("--keep-output",default=None)
p.add_argument("--docx-excerpt-chars",type=int,default=12,
               help="One-epoch untrained NER is dense; smoke inference uses an explicitly labelled excerpt, not full quality evaluation")
args=p.parse_args()
torch.set_num_threads(2)
warnings.filterwarnings("ignore",message="Glyph .* missing from font",category=UserWarning)
cells=notebook_cells()
ns=notebook_namespace()
documents,_=ns["load_documents"](args.dataset)
small=sorted(documents,key=lambda d:len(d["text"]))[:3]
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);data=root/"dataset";data.mkdir()
    for i,d in enumerate(small):
        shutil.copy2(Path(args.dataset)/d["id"],data/f"{i}_ket_qua.json")
    ns.update(DATASET_DIR=data,OUTPUT_DIR=root/"output",IS_COLAB=False,LOCAL_FILES_ONLY=True)
    exec(cells["pipeline"].split("# AUDIT DATASET")[1],ns)
    train_args=ns["parser"]().parse_args(["--dataset",str(data),"--output",str(root/"output"),
                                        "--encoder","vinai/phobert-base-v2","--device","cpu","--local-files-only",
                                        "--ner-epochs","1","--relation-epochs","1","--hidden","16"])
    ns["run"](train_args)
    for tag in ["curves","evaluation","comparison","inference","export"]:
        print(f"NOTEBOOK_CELL: {tag}",flush=True)
        source=cells[tag]
        if tag=="inference":
            source=source.replace("INPUT_FILE = None",f"INPUT_FILE = {str(Path(args.docx).resolve())!r}")
            # Read the real DOCX in full, then keep a bounded excerpt for the one-epoch
            # smoke model. Production notebook never applies this test-only wrapper.
            marker="if INPUT_FILE is None:"
            wrapper=f'''_real_docx_reader = read_input_document
def _smoke_docx_reader(path):
    text, info = _real_docx_reader(path)
    info['smoke_full_document_characters'] = len(text)
    info['smoke_excerpt_characters'] = {args.docx_excerpt_chars}
    return text[:{args.docx_excerpt_chars}], info
read_input_document = _smoke_docx_reader
'''
            source=source.replace(marker,wrapper+'\n'+marker,1)
        exec(compile(source,f"notebook:{tag}","exec"),ns)
    for name in ["figures/ner_training.png","figures/re_training.png",
                 "report_ner.csv","report_relations.csv","run_status.json"]:
        assert (root/"output"/name).is_file(),name
    assert ns['html_path'].is_file()
    assert (ns['prediction_dir']/'entities.csv').is_file()
    assert (ns['prediction_dir']/'relation_graph.png').is_file()
    if args.keep_output:
        shutil.copytree(root/'output',args.keep_output,dirs_exist_ok=True,ignore=shutil.ignore_patterns('cache'))
    print("STANDALONE_NOTEBOOK_SMOKE_OK: PhoBERT-base-v2; 1 epoch/3 tiny docs; plots/checkpoint reload; full DOCX reading then bounded excerpt inference/HTML. NOT research-quality training or full-document inference benchmark.")
