"""Measured evaluation reports; separate NER/RE tasks and preserve raw metrics."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from labels import ALL_ENTITIES, ALL_RELATIONS


def generate_training_report(output):
    output = Path(output)
    def read(name):
        return json.loads((output / name).read_text(encoding="utf-8"))
    ner = read("ner_test.json")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    saved = []
    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(figures / name))
    history = read("ner_history.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [h["epoch"] for h in history]
    axes[0].plot(epochs, [h["loss"] for h in history], marker=".")
    axes[0].set(title="NER: training loss", xlabel="Epoch", ylabel="Weighted BCE")
    axes[1].plot(epochs, [100*h["val_f1"] for h in history], marker=".")
    axes[1].set(title="NER: validation exact-span/type F1", xlabel="Epoch", ylabel="F1 (%)", ylim=(0,100))
    for ax in axes: ax.grid(alpha=.25)
    finish(fig, "ner_training.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rows = []
    for variant in ("semantic", "rgcn"):
        file = output / f"test_{variant}.json"
        if not file.exists(): continue
        h = read(f"relation_{variant}_history.json")
        x = [a["epoch"] for a in h]
        axes[0].plot(x, [a["loss"] for a in h], label=variant, marker=".")
        metric = "val_f1_micro_positive" if "val_f1_micro_positive" in h[0] else "val_f1_macro_22"
        axes[1].plot(x, [100*a[metric] for a in h], label=f"{variant}: {metric}", marker=".")
        for setting, scores in read(file.name).items():
            rows.append({"model": variant, "setting": setting,
                "precision_micro_pct": 100*scores["precision_micro_positive"],
                "recall_micro_pct": 100*scores["recall_micro_positive"],
                "f1_micro_pct": 100*scores["f1_micro_positive"],
                "f1_macro_22_pct": 100*scores["f1_macro_22"],
                "candidate_recall_pct": 100*scores["candidate_recall"],
                "supported_relations": scores["supported_classes"]})
            per = pd.DataFrame(scores["per_class"]).T
            per["fn"] = per["support"] - per["tp"]
            per.to_csv(output/f"relations_{variant}_{setting}.csv", encoding="utf-8-sig", index_label="code")
    axes[0].set(title="RE: training loss", xlabel="Epoch", ylabel="Weighted BCE")
    axes[1].set(title="RE: validation selection score", xlabel="Epoch", ylabel="F1 (%)", ylim=(0,100))
    for ax in axes: ax.grid(alpha=.25); ax.legend(fontsize=7)
    finish(fig, "re_training.png")
    main = pd.DataFrame(rows)
    main.to_csv(output/"report_relations.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([{"task":"NER exact-span/type", "precision_pct":100*ner["precision"],
        "recall_pct":100*ner["recall"], "f1_pct":100*ner["f1"], "tp":ner["tp"], "fp":ner["fp"], "fn":ner["fn"]}])
    summary.to_csv(output/"report_ner.csv", index=False, encoding="utf-8-sig")
    entities = pd.DataFrame(ner["per_type"]).T
    entities["support"] = entities["tp"] + entities["fn"]
    entities.to_csv(output/"entities_per_type.csv", encoding="utf-8-sig", index_label="code")
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(["Precision","Recall","F1"], [100*ner[k] for k in ("precision","recall","f1")], color=["#2563eb","#0d9488","#7c3aed"])
    ax.set(title="NER test: exact-span AND entity type", ylim=(0,100), ylabel="%")
    finish(fig, "ner_test.png")
    fig, ax = plt.subplots(figsize=(11,5))
    end = main[main.setting=="predicted_entities_end_to_end"]
    x = np.arange(len(end)); width=.2
    for j,col in enumerate(["precision_micro_pct","recall_micro_pct","f1_micro_pct","f1_macro_22_pct"]):
        ax.bar(x+(j-1.5)*width, end[col], width, label=col.replace("_pct", ""))
    ax.set_xticks(x, end.model)
    ax.set(title="RE test: predicted entities (end-to-end)", ylabel="%", ylim=(0,100))
    ax.legend(fontsize=8)
    finish(fig, "re_test_comparison.png")
    fig, ax = plt.subplots(figsize=(12,11))
    ax.barh(entities.index[::-1], 100*entities.f1.iloc[::-1], color="#0d9488")
    ax.set(title="NER test F1 by entity type (zero includes unsupported types)", xlabel="F1 (%)", xlim=(0,100))
    ax.tick_params(axis="y", labelsize=7)
    finish(fig, "ner_per_type.png")
    preferred = "rgcn" if (output/"test_rgcn.json").exists() else "semantic"
    scores = read(f"test_{preferred}.json")["predicted_entities_end_to_end"]
    per = pd.DataFrame(scores["per_class"]).T
    per["fn"] = per.support-per.tp
    matrix = per[["tp","fp","fn"]].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(13,9))
    ax.imshow(np.log1p(matrix), cmap="Blues", aspect="auto")
    ax.set_xticks(range(3),["TP","FP","FN"])
    ax.set_yticks(range(len(per)), per.index, fontsize=7)
    for i in range(len(per)):
        for j in range(3):
            color = "white" if np.log1p(matrix[i,j]) > np.log1p(matrix.max())*.55 else "#172033"
            ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=8,color=color)
    ax.set_title(f"RE {preferred} end-to-end: counts by relation (log color scale)")
    finish(fig, "re_errors.png")
    note = """# Kết quả thực đo\n\nNER: đúng đồng thời ranh giới và loại. RE: quan hệ dương có hướng; kết quả chính dùng thực thể NER dự đoán.\n\nMacro-F1 lấy đủ 22 lớp, lớp không có mẫu có F1=0. Đọc cùng support; micro-F1 có thể bị THUOC chi phối.\n\nLoss là loss huấn luyện có trọng số, không phải phần trăm chính xác. Các biểu đồ không chứng minh khả năng khái quát nếu nhãn test chưa được người kiểm tra.\n\n"""
    (output/"REPORT.md").write_text(note+summary.round(2).to_markdown(index=False)+"\n\n"+main.round(2).to_markdown(index=False),encoding="utf-8")
    return {"ner":summary, "relations":main, "figures":saved}
