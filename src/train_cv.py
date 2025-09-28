# src/train_cv.py
from __future__ import annotations

# --- garante o pacote top-level 'src' no sys.path quando rodar como script ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

# Reaproveita utilitários e pipeline do treino baseline
from src.train_baseline import (
    find_file,
    load_json,
    first_key,
    flatten_text_from_subdicts,
    label_from_text,
    score_tecnico,
    make_pipeline,
    choose_threshold,
    JOB_SUBDICT_KEYS,
    APPLICANT_SUBDICT_KEYS,
    APPLICANT_CV_KEYS,
)

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_CV_FILE = MODELS_DIR / "metrics_cv.json"

# Ajuste para o seu formato (mesmo do baseline)
CANDIDATE_KEY = ["codigo"]
STATUS_KEYS = ["situacao_candidado", "situacao", "status"]
COMMENT_KEYS = ["comentario"]


def build_dataset() -> pd.DataFrame:
    jobs = load_json(find_file("Jobs.json"))
    prospects = load_json(find_file("Prospects.json"))
    applicants = load_json(find_file("Applicants.json"))

    rows: List[Dict[str, Any]] = []

    for vaga_code, blob in prospects.items():
        if not isinstance(blob, dict):
            continue
        plist = blob.get("prospects") or blob.get("prospeccoes") or []
        if not isinstance(plist, list):
            continue

        job_obj = jobs.get(str(vaga_code), {}) if isinstance(jobs, dict) else {}
        job_text = flatten_text_from_subdicts(job_obj, JOB_SUBDICT_KEYS)

        for it in plist:
            if not isinstance(it, dict):
                continue

            cand_key = first_key(it, CANDIDATE_KEY)
            cand_code = str(it.get(cand_key)) if cand_key else ""

            status_key = first_key(it, STATUS_KEYS)
            raw_status = str(it.get(status_key) or "")
            y = label_from_text(raw_status)
            if y is None:
                ckey = first_key(it, COMMENT_KEYS)
                if ckey:
                    y = label_from_text(str(it.get(ckey) or ""))

            cand_obj = applicants.get(cand_code, {}) if isinstance(applicants, dict) else {}
            cand_text = flatten_text_from_subdicts(
                cand_obj, APPLICANT_SUBDICT_KEYS, APPLICANT_CV_KEYS
            )

            st = score_tecnico(job_text, cand_text)

            rows.append(
                {
                    "vaga_code": str(vaga_code),
                    "candidato_code": cand_code,
                    "job_text": job_text,
                    "cand_text": cand_text,
                    "situacao_norm": raw_status,
                    "score_tecnico": st,
                    "y": y,
                }
            )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)
    df = df.drop_duplicates(subset=["vaga_code", "candidato_code"], keep="last")
    if df["y"].nunique() < 2:
        raise RuntimeError("Dataset tem uma única classe — ajuste rótulos POS/NEG.")
    return df


def main():
    df = build_dataset()
    X = df[["job_text", "cand_text", "situacao_norm", "score_tecnico"]].reset_index(drop=True)
    y = df["y"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs, accs, f1s, precs, recs = [], [], [], [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        pipe = make_pipeline().fit(X_tr, y_tr)

        # threshold escolhido no treino
        proba_tr = pipe.predict_proba(X_tr)[:, 1]
        thr = choose_threshold(proba_tr, y_tr)

        proba_te = pipe.predict_proba(X_te)[:, 1]
        yhat_te = (proba_te >= thr).astype(int)

        auc = float(roc_auc_score(y_te, proba_te))
        acc = float(accuracy_score(y_te, yhat_te))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, yhat_te, average="binary", zero_division=0
        )
        aucs.append(auc)
        accs.append(acc)
        f1s.append(float(f1))
        precs.append(float(prec))
        recs.append(float(rec))

        print(
            f"[CV fold {fold}] AUC={auc:.3f} | Acc={acc:.3f} | F1={f1:.3f} | "
            f"Prec={prec:.3f} | Rec={rec:.3f} | thr={thr:.3f}"
        )

    metrics_cv = {
        "n_total": int(len(df)),
        "n_splits": 5,
        "pos_rate_total": float(y.mean()),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s, ddof=1)),
        "precision_mean": float(np.mean(precs)),
        "precision_std": float(np.std(precs, ddof=1)),
        "recall_mean": float(np.mean(recs)),
        "recall_std": float(np.std(recs, ddof=1)),
    }

    METRICS_CV_FILE.write_text(
        json.dumps(metrics_cv, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] métricas CV salvas em {METRICS_CV_FILE}")


if __name__ == "__main__":
    main()
