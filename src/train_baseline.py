# src/train_baseline.py
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List, Iterable
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------
# Caminhos e constantes
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = [ROOT / "data", ROOT / "data" / "raw"]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD_FILE = MODELS_DIR / "decision_threshold.json"
METRICS_FILE = MODELS_DIR / "metrics.json"
DEFAULT_THRESHOLD = 0.59

# === Ajustado ao seu formato (conforme peek/inspect) ===
# Prospects.json: dict[vaga_code] -> { titulo, modalidade, prospects: [ { nome, codigo, situacao_candidado, comentario, ... } ] }
CANDIDATE_KEY = ["codigo"]  # id do candidato dentro de cada item de prospects
STATUS_KEYS = ["situacao_candidado", "situacao", "status"]  # pode estar vazio
COMMENT_KEYS = ["comentario"]  # fallback para inferir rótulo pelo texto

# Jobs.json: dict por vaga com subdicts
JOB_SUBDICT_KEYS = ["informacoes_basicas", "perfil_vaga", "beneficios"]

# Applicants.json: dict por candidato com subdicts + CVs
APPLICANT_SUBDICT_KEYS = [
    "infos_basicas",
    "informacoes_pessoais",
    "informacoes_profissionais",
    "formacao_e_idiomas",
    "cargo_atual",
]
APPLICANT_CV_KEYS = ["cv_pt", "cv_en"]

# Rótulos por substring (ajuste se necessário conforme seus comentários reais)
POS_KEYS = {
    "aprovado",
    "aprovacao",
    "aprovada",
    "aprovado cliente",
    "aprovado entrevista",
    "contratado",
    "oferta aceita",
    "finalista",
    "hired",
}
NEG_KEYS = {
    "reprovado",
    "reprova",
    "descartado",
    "nao avancou",
    "nao selecionado",
    "rejeitado",
    "desistiu",
    "sem perfil",
    "bloqueado",
    "fail",
    "sem interesse",
}

# Weak labels: percentis do score_tecnico (extremos ganham rótulo; meio é descartado)
PCTL_POS, PCTL_NEG = 0.70, 0.30


# --------------------------------------------------------------------------------------
# IO e utilitários
# --------------------------------------------------------------------------------------
def find_file(name: str) -> Path:
    for base in DATA_DIRS:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Não encontrei {name} em {DATA_DIRS}")

def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    return (
        s.replace("ã", "a")
        .replace("â", "a")
        .replace("á", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )

def first_key(d: dict, candidates: Iterable[str]) -> str | None:
    low = {k.lower(): k for k in d.keys()}
    for c in candidates:
        if c in low:
            return low[c]
    return None

def label_from_text(s: str) -> int | None:
    s = norm(s)
    if any(k in s for k in POS_KEYS):
        return 1
    if any(k in s for k in NEG_KEYS):
        return 0
    return None

def flatten_text_from_subdicts(
    obj: dict, subkeys: List[str], extra_text_keys: List[str] | None = None
) -> str:
    parts: List[str] = []
    if extra_text_keys:
        for k in extra_text_keys:
            v = obj.get(k)
            if isinstance(v, list):
                v = " ".join(map(str, v))
            if v:
                parts.append(str(v))
    for sk in subkeys:
        sub = obj.get(sk)
        if isinstance(sub, dict):
            for v in sub.values():
                if isinstance(v, list):
                    parts.append(" ".join(map(str, v)))
                elif isinstance(v, (str, int, float)):
                    parts.append(str(v))
                elif isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, list):
                            parts.append(" ".join(map(str, vv)))
                        elif isinstance(vv, (str, int, float)):
                            parts.append(str(vv))
        elif isinstance(sub, list):
            parts.append(" ".join(map(str, sub)))
    return " ".join(parts).strip()

def tokenize(t: str) -> set[str]:
    return {w for w in str(t).lower().replace("/", " ").replace(",", " ").split() if w}

def score_tecnico(job_text: str, cand_text: str) -> float:
    a, b = tokenize(job_text), tokenize(cand_text)
    if not a or not b:
        return 0.0
    inter, uni = len(a & b), len(a | b)
    return float(inter / uni)


# --------------------------------------------------------------------------------------
# --- helpers pickláveis (nível de módulo) ---
# --------------------------------------------------------------------------------------
def concat_cols_df(X):
    # mesma lógica usada na sua API
    return (
        "[JOB]" + X["job_text"].astype(str) + " "
        + "[CAND]" + X["cand_text"].astype(str) + " "
        + "[SIT]" + X["situacao_norm"].astype(str) + " "
        + "[SCORE]" + X["score_tecnico"].astype(str)
    )

def select_score_df(X):
    # seleciona a coluna numérica para o ramo "score"
    return X[["score_tecnico"]]


# --------------------------------------------------------------------------------------
# Pipeline (sem lambdas/closures, para ser picklável pelo joblib)
# --------------------------------------------------------------------------------------
def make_pipeline() -> Pipeline:
    text_maker = FunctionTransformer(concat_cols_df, validate=False)
    col = ColumnTransformer(
        [
            (
                "text",
                Pipeline(
                    [
                        ("concat", text_maker),
                        ("tfidf", TfidfVectorizer()),
                    ]
                ),
                ["job_text", "cand_text", "situacao_norm", "score_tecnico"],
            ),
            (
                "score",
                Pipeline(
                    [
                        ("sel", FunctionTransformer(select_score_df, validate=False)),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                ["score_tecnico"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("prep", col),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


# --------------------------------------------------------------------------------------
# Threshold (Youden simples) + fallback
# --------------------------------------------------------------------------------------
def choose_threshold(proba: np.ndarray, y: np.ndarray) -> float:
    try:
        cand = np.linspace(0.2, 0.8, 61)
        auc = roc_auc_score(y, proba)
        best_t, best_val = None, None
        for t in cand:
            yhat = (proba >= t).astype(int)
            acc = (yhat == y).mean()
            youden = acc + auc - 1.0
            if (best_val is None) or (youden > best_val):
                best_t, best_val = t, youden
        return float(best_t) if best_t is not None else DEFAULT_THRESHOLD
    except Exception:
        return DEFAULT_THRESHOLD


# --------------------------------------------------------------------------------------
# Main (com HOLDOUT + metrics.json)
# --------------------------------------------------------------------------------------
def main():
    jobs = load_json(find_file("Jobs.json"))            # dict[str -> job_obj]
    prospects = load_json(find_file("Prospects.json"))  # dict[str -> {titulo, modalidade, prospects:[...] }]
    applicants = load_json(find_file("Applicants.json"))# dict[str -> applicant_obj]

    rows: List[Dict[str, Any]] = []

    # Varre cada vaga e sua lista de prospects
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

            # Código do candidato
            cand_key = first_key(it, CANDIDATE_KEY)
            cand_code = str(it.get(cand_key)) if cand_key else ""

            # Status e/ou comentário para rótulo explícito
            status_key = first_key(it, STATUS_KEYS)
            raw_status = str(it.get(status_key) or "")
            y = label_from_text(raw_status)
            if y is None:
                comment_key = first_key(it, COMMENT_KEYS)
                if comment_key:
                    y = label_from_text(str(it.get(comment_key) or ""))

            # Texto do candidato (flatten dos subdicts + CVs)
            cand_obj = applicants.get(cand_code, {}) if isinstance(applicants, dict) else {}
            cand_text = flatten_text_from_subdicts(cand_obj, APPLICANT_SUBDICT_KEYS, APPLICANT_CV_KEYS)

            st = score_tecnico(job_text, cand_text)

            rows.append(
                {
                    "vaga_code": str(vaga_code),
                    "candidato_code": cand_code,
                    "job_text": job_text,
                    "cand_text": cand_text,
                    "situacao_norm": raw_status,  # pode estar vazio
                    "score_tecnico": st,
                    "y": y,  # pode ser None
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Sem pares gerados. Verifique a estrutura dos JSONs.")

    # Weak labels pelos extremos do score_tecnico (se faltar rótulo explícito)
    unlabeled = df["y"].isna()
    if unlabeled.any():
        scores = df.loc[unlabeled, "score_tecnico"].values
        if len(scores) >= 10:
            t_pos = float(np.nanpercentile(scores, PCTL_POS * 100))
            t_neg = float(np.nanpercentile(scores, PCTL_NEG * 100))
            df.loc[unlabeled & (df["score_tecnico"] >= t_pos), "y"] = 1
            df.loc[unlabeled & (df["score_tecnico"] <= t_neg), "y"] = 0
            # faixa cinza é descartada
            df = df.dropna(subset=["y"])
        else:
            df = df.dropna(subset=["y"])

    df["y"] = df["y"].astype(int)

    # remove duplicatas vaga-candidato
    df = df.drop_duplicates(subset=["vaga_code", "candidato_code"], keep="last")

    if df["y"].nunique() < 2:
        raise RuntimeError(
            "Após rotulagem, só há uma classe. Ajuste POS_KEYS/NEG_KEYS ou os percentis PCTL_POS/PCTL_NEG."
        )

    X_all = df[["job_text", "cand_text", "situacao_norm", "score_tecnico"]]
    y_all = df["y"].values

    # HOLDOUT estratificado
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=42
    )

    # Treina no treino, escolhe threshold no treino
    pipe = make_pipeline().fit(X_tr, y_tr)
    proba_tr = pipe.predict_proba(X_tr)[:, 1]
    thr_tr = choose_threshold(proba_tr, y_tr)

    # Avalia na validação
    proba_val = pipe.predict_proba(X_val)[:, 1]
    yhat_val = (proba_val >= thr_tr).astype(int)

    auc_val = float(roc_auc_score(y_val, proba_val))
    acc_val = float(accuracy_score(y_val, yhat_val))
    prec_val, rec_val, f1_val, _ = precision_recall_fscore_support(
        y_val, yhat_val, average="binary", zero_division=0
    )
    f1_val = float(f1_val)
    prec_val = float(prec_val)
    rec_val = float(rec_val)

    # Re-treina em 100% e escolhe threshold final
    pipe_full = make_pipeline().fit(X_all, y_all)
    proba_full = pipe_full.predict_proba(X_all)[:, 1]
    thr_final = choose_threshold(proba_full, y_all)

    # Salva artefatos
    joblib.dump(pipe_full, MODELS_DIR / "model.joblib")
    THRESHOLD_FILE.write_text(
        json.dumps({"threshold": float(thr_final)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Salva métricas de validação
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_total": int(len(df)),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "pos_rate_total": float(y_all.mean()),
        "pos_rate_train": float(y_tr.mean()),
        "pos_rate_val": float(y_val.mean()),
        "auc_val": auc_val,
        "accuracy_val": acc_val,
        "f1_val": f1_val,
        "precision_val": prec_val,
        "recall_val": rec_val,
        "threshold_train": float(thr_tr),
        "threshold_final": float(thr_final),
    }
    METRICS_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Logs
    print(
        f"[VAL] n_total={len(df)} | n_tr={len(X_tr)} | n_val={len(X_val)} | "
        f"AUC_val={auc_val:.3f} | Acc_val={acc_val:.3f} | F1_val={f1_val:.3f} | "
        f"Prec_val={prec_val:.3f} | Rec_val={rec_val:.3f} | thr_train={thr_tr:.3f} | thr_final={thr_final:.3f}"
    )
    print(f"[OK] modelo salvo: {MODELS_DIR/'model.joblib'}")
    print(f"[OK] threshold salvo: {THRESHOLD_FILE}")
    print(f"[OK] métricas salvas: {METRICS_FILE}")


if __name__ == "__main__":
    main()
