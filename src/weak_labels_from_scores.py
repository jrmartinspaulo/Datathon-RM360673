# src/weak_labels_from_scores.py
# -*- coding: utf-8 -*-
"""
Gera rótulos fracos (weak labels) de (vaga_code, candidato_code) priorizando SCORE.
- O score (rank por vaga) é a fonte primária do y.
- 'situacao_norm' só ajusta o limiar por vaga (boost/penalty).
- Salvaguarda: se todos os scores da vaga forem <= 0, ninguém vira 1 pelo score.
- Defaults (equilíbrio escolhido): top_k=2, min_score=0.02, quantile=0.85
- Leitura opcional de configs/weak_labels.yaml para sobrescrever parâmetros.
- Salva metadados em data/processed/labels_meta.json para auditoria.

Entradas:
- data/interim/prospects.csv
- (opcional) data/processed/scores.csv  -> colunas: vaga_code, candidato_code, score_tecnico

Saídas:
- data/processed/labels_by_candidato_vaga.csv
- data/processed/labels_meta.json
"""

from __future__ import annotations
import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import pandas as pd

# -----------------------
# Caminhos
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"

PROSPECTS_CSV = INTERIM_DIR / "prospects.csv"
SCORES_CSV = PROCESSED_DIR / "scores.csv"   # opcional
OUT_CSV = PROCESSED_DIR / "labels_by_candidato_vaga.csv"
META_JSON = PROCESSED_DIR / "labels_meta.json"
CONFIG_YAML = CONFIGS_DIR / "weak_labels.yaml"

# -----------------------
# Defaults (equilíbrio escolhido)
# -----------------------
DEFAULT_TOP_K = 2
DEFAULT_MIN_SCORE = 0.02
DEFAULT_QUANTILE = 0.85

# -----------------------
# Regras de status (ajuste de limiar)
# -----------------------
BOOST_STATUSES = {
    "encaminhado ao requisitante",
    "entrevista técnica",
    "entrevista tecnica",
    "prospect",
    "inscrito",
}
PENALTY_STATUSES = {
    "desistiu",
    "desistencia",
    "não aprovado",
    "nao aprovado",
    "não aprovado pelo cliente",
    "nao aprovado pelo cliente",
    "não aprovado pelo rh",
    "nao aprovado pelo rh",
    "não aprovado pelo requisitante",
    "nao aprovado pelo requisitante",
    "sem interesse nesta vaga",
}

# Regras fortes (desabilitadas para o score decidir)
HARD_POS_STATUSES = set()
HARD_NEG_STATUSES = set()

# -----------------------
# Utilitários
# -----------------------
def _read_yaml_config(path: Path) -> Dict[str, Any]:
    """
    Lê YAML se existir e PyYAML estiver instalado; senão retorna {}.
    Chaves aceitas: top_k, min_score, quantile.
    """
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        print(f"[INFO] PyYAML não instalado; ignorando {path.name}.")
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            return {}
        # Sanitiza tipos
        out: Dict[str, Any] = {}
        if "top_k" in cfg: out["top_k"] = int(cfg["top_k"])
        if "min_score" in cfg: out["min_score"] = float(cfg["min_score"])
        if "quantile" in cfg: out["quantile"] = float(cfg["quantile"])
        return out
    except Exception as e:
        print(f"[INFO] Falha ao ler {path.name}: {e}. Seguindo com defaults/CLI.")
        return {}

def _md5_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def read_prospects() -> pd.DataFrame:
    if not PROSPECTS_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {PROSPECTS_CSV}")
    df = pd.read_csv(
        PROSPECTS_CSV,
        dtype={"vaga_code": str, "candidato_code": str},
        encoding="utf-8",
        low_memory=False,
    )
    for col in ["vaga_code", "candidato_code", "nome", "situacao", "situacao_norm", "score_tecnico"]:
        if col not in df.columns:
            df[col] = None
    df["vaga_code"] = df["vaga_code"].astype(str)
    df["candidato_code"] = df["candidato_code"].astype(str)
    df["situacao_norm"] = df["situacao_norm"].astype(str).str.strip().str.lower()
    return df

def attach_scores(base: pd.DataFrame) -> pd.DataFrame:
    if SCORES_CSV.exists():
        scores = pd.read_csv(
            SCORES_CSV,
            dtype={"vaga_code": str, "candidato_code": str},
            encoding="utf-8",
            low_memory=False,
        )
        score_col = None
        for c in ["score_tecnico", "score", "score_modelo", "score_similarity"]:
            if c in scores.columns:
                score_col = c
                break
        if score_col is None:
            scores["score_tecnico"] = 0.0
            score_col = "score_tecnico"

        scores = scores[["vaga_code", "candidato_code", score_col]].rename(columns={score_col: "score_tecnico"})
        merged = base.merge(scores, on=["vaga_code", "candidato_code"], how="left", suffixes=("", "_sc"))
        merged["score_tecnico"] = pd.to_numeric(merged["score_tecnico"], errors="coerce")
        merged["score_tecnico"] = merged["score_tecnico"].fillna(pd.to_numeric(merged.get("score_tecnico_sc"), errors="coerce"))
        merged["score_tecnico"] = pd.to_numeric(merged["score_tecnico"], errors="coerce").fillna(0.0)
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_sc")], errors="ignore")
        return merged
    else:
        base["score_tecnico"] = pd.to_numeric(base.get("score_tecnico", 0), errors="coerce").fillna(0.0)
        return base

# -----------------------
# Rotulagem por vaga (SCORE como primário)
# -----------------------
def label_group(g: pd.DataFrame, top_k: int, min_score: float, quantile: float) -> pd.DataFrame:
    """
    Score é primário. Status ajusta o limiar (boost/penalty).
    Salvaguarda: se todos os scores <= 0 na vaga, ninguém vira 1 via score.
    """
    g = g.copy()
    g["score_tecnico"] = pd.to_numeric(g.get("score_tecnico", 0), errors="coerce").fillna(0.0)

    # Normaliza status
    situ = g.get("situacao_norm", "")
    if situ is None:
        situ = ""
    situ = situ.astype(str).str.strip().str.lower()

    # Ordena por score DESC
    g = g.sort_values("score_tecnico", ascending=False)

    # Salvaguarda: sem sinal (>0) -> ninguém vira 1 por score
    if g["score_tecnico"].max() <= 0:
        g["y"] = 0
        # Regras fortes (se quiser habilitar)
        if len(HARD_POS_STATUSES) > 0:
            g.loc[situ.isin(HARD_POS_STATUSES), "y"] = 1
        if len(HARD_NEG_STATUSES) > 0:
            g.loc[situ.isin(HARD_NEG_STATUSES), "y"] = 0
        return g

    # Limiar base: quantil por vaga + piso absoluto
    thr_q = g["score_tecnico"].quantile(quantile) if g["score_tecnico"].notna().any() else 0.0
    base_thr = max(min_score, thr_q)

    # Ajuste de limiar por status (+/-10%)
    boost_mask = situ.isin(BOOST_STATUSES)
    penalty_mask = situ.isin(PENALTY_STATUSES)
    thr_adj = pd.Series(base_thr, index=g.index, dtype=float)
    thr_adj.loc[boost_mask] = base_thr * 0.90
    thr_adj.loc[penalty_mask] = base_thr * 1.10

    # Seleciona top_k por score e aplica limiar ajustado
    k = max(1, min(int(top_k), len(g)))
    idx_top = g.index[:k]
    y_score = (g["score_tecnico"] >= thr_adj)
    y_score = y_score & y_score.index.isin(idx_top)

    # Regras fortes (desativadas por padrão)
    y = y_score.astype(int)
    if len(HARD_POS_STATUSES) > 0:
        y.loc[situ.isin(HARD_POS_STATUSES)] = 1
    if len(HARD_NEG_STATUSES) > 0:
        y.loc[situ.isin(HARD_NEG_STATUSES)] = 0

    # Safeguards:
    # 1) Se todos 1 e há mais de um candidato -> restringe a top_k
    if y.sum() == len(g) and len(g) > 1:
        y[:] = 0
        y.loc[idx_top] = 1
        if len(HARD_NEG_STATUSES) > 0:
            y.loc[situ.isin(HARD_NEG_STATUSES)] = 0

    # 2) Se ninguém 1 mas há score acima do limiar base -> garante top-1 (desde que não seja hard-negativo)
    if y.sum() == 0 and (g["score_tecnico"].max() > base_thr):
        top1 = g.index[0]
        if not (len(HARD_NEG_STATUSES) > 0 and situ.loc[top1] in HARD_NEG_STATUSES):
            y.loc[top1] = 1

    g["y"] = y.astype(int)
    return g

# -----------------------
# Execução
# -----------------------
def run(top_k: int, min_score: float, quantile: float, cfg_used: Dict[str, Any]) -> None:
    print(f"[INFO] Carregando prospects: {PROSPECTS_CSV}")
    prospects = read_prospects()

    # Filtra pares válidos
    mask_ok = (
        prospects["vaga_code"].notna() & (prospects["vaga_code"].str.len() > 0) &
        prospects["candidato_code"].notna() & (prospects["candidato_code"].str.len() > 0)
    )
    prospects = prospects.loc[mask_ok].copy()
    print(f"[OK] Prospects: {prospects.shape}")

    # Anexa scores
    print(f"[INFO] Anexando scores (opcional): {SCORES_CSV.name}")
    base = attach_scores(prospects)

    # Mantém colunas úteis
    keep_cols = ["vaga_code", "candidato_code", "nome", "situacao", "situacao_norm", "score_tecnico"]
    base = base[keep_cols].copy()

    # Aplica rotulagem por vaga
    labeled = (
        base
        .groupby("vaga_code", group_keys=True, sort=False)
        .apply(lambda g: label_group(g, top_k=top_k, min_score=min_score, quantile=quantile), include_groups=False)
        .reset_index(level=0)    # garante 'vaga_code' como coluna
        .reset_index(drop=True)
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    tot = len(labeled)
    pos = int(labeled["y"].sum())
    neg = int((labeled["y"] == 0).sum())
    print(f"[OK] Weak labels gerados: {OUT_CSV}")
    print(f"[INFO] Total pares vaga-candidato: {tot} | Positivos(y=1): {pos} | Negativos(y=0): {neg}")

    # Amostra
    cols_show = [c for c in ["vaga_code", "candidato_code", "nome", "score_tecnico", "situacao_norm", "y"] if c in labeled.columns]
    print("\n[Amostra Top-10 positivos (ordenado por score_tecnico desc)]")
    print(labeled[labeled["y"] == 1].sort_values("score_tecnico", ascending=False)[cols_show].head(10))

    # Salva metadados
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "top_k": top_k,
            "min_score": min_score,
            "quantile": quantile,
        },
        "config_file_used": str(CONFIG_YAML if CONFIG_YAML.exists() else None),
        "scores_csv_exists": SCORES_CSV.exists(),
        "scores_csv_md5": _md5_file(SCORES_CSV),
        "output_csv": str(OUT_CSV),
        "counts": {"total": tot, "positivos": pos, "negativos": neg},
    }
    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] Metadados salvos: {META_JSON}")

def parse_args():
    p = argparse.ArgumentParser(description="Geração de weak labels por vaga (score como primário).")
    p.add_argument("--top-k", type=int, default=None, help=f"Quantidade top-k por vaga (default={DEFAULT_TOP_K})")
    p.add_argument("--min-score", type=float, default=None, help=f"Score mínimo para considerar (default={DEFAULT_MIN_SCORE})")
    p.add_argument("--quantile", type=float, default=None, help=f"Quantil do score como limiar por vaga (default={DEFAULT_QUANTILE})")
    return p.parse_args()

if __name__ == "__main__":
    # Defaults
    params = {
        "top_k": DEFAULT_TOP_K,
        "min_score": DEFAULT_MIN_SCORE,
        "quantile": DEFAULT_QUANTILE,
    }
    # YAML opcional
    cfg = _read_yaml_config(CONFIG_YAML)
    params.update({k: v for k, v in cfg.items() if v is not None})
    # CLI override (se passado)
    args = parse_args()
    if args.top_k is not None: params["top_k"] = args.top_k
    if args.min_score is not None: params["min_score"] = args.min_score
    if args.quantile is not None: params["quantile"] = args.quantile

    print("[INFO] Parâmetros efetivos:", params)
    run(top_k=params["top_k"], min_score=params["min_score"], quantile=params["quantile"], cfg_used=cfg)
