"""Audit the raw-document dataset. Training uses data_pipeline + training.make_graph."""

import argparse
import json
from data_pipeline import load_documents, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="output_results/data_audit.json")
    args = parser.parse_args()
    _, report = load_documents(args.dataset)
    save_json(args.output, report)
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
