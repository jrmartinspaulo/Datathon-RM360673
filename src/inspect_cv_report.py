# src/inspect_cv_report.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path

REPORT = Path("data/processed/train_cv_report.json")

def main():
    j = json.loads(REPORT.read_text(encoding="utf-8"))
    print("\n[MÉDIAS CV]")
    for k, v in j["means"].items():
        print(f"{k}: {v:.4f}")
    print("\n[FOLDS]")
    for m in j["fold_metrics"]:
        print(f"fold={m['fold']} | roc_auc={m['roc_auc']:.4f} | pr_auc={m['pr_auc']:.4f} | f1@0.5={m['f1_at_0.5']:.4f}")
    bt = j["best_threshold"]
    print(f"\n[THRESHOLD ÓTIMO] t={bt['threshold']:.2f} | F1={bt['f1']:.4f} | precision={bt['precision']:.4f} | recall={bt['recall']:.4f}")

if __name__ == "__main__":
    main()
