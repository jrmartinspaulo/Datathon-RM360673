# src/make_labels.py
# -*- coding: utf-8 -*-
"""
Lê data/interim/prospects.csv e gera data/processed/labels_by_candidato_vaga.csv
com a coluna y (0/1) a partir do status/situacao do candidato na vaga.
- Normaliza acentos e caixa
- Usa lista ampliada de positivos
- Fallback: busca palavras-chave positivas em 'comentario' quando 'situacao' vier vazia/inespecífica
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import re
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROSPECTS_CSV = INTERIM_DIR / "prospects.csv"
OUT_CSV = PROCESSED_DIR / "labels_by_candidato_vaga.csv"

# Valores de 'situacao' claramente positivos
POSITIVE_SITUACOES = {
    "contratado", "contratada",
    "aprovado", "aprovada",
    "admitido", "admitida",
    "hired", "approved",
    # variações comuns
    "aprovado cliente", "aprovada cliente",
    "aprovado pelo cliente", "aprovada pelo cliente",
    "proposta aceita", "oferta aceita",
    "alocado", "alocada",
    "colocado", "colocada",
    "selecionado", "selecionada",
}

# Palavras/raizes que denotam positivo
POSITIVE_KEYWORDS = [
    r"\bcontrat",       # contratado/contratada/contratar
    r"\baprov",         # aprovado/aprovação
    r"\badmit",         # admitido
    r"\bhir",           # hire/hired/hiring
    r"\bproposta aceita",
    r"\boferta aceita",
    r"\baloc",          # alocado
    r"\bcoloc",         # colocado
    r"\bselecion",      # selecionado/seleção
]

POS_RE = re.compile("|".join(POSITIVE_KEYWORDS))

def normalize(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    # remove acentos
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s

def is_positive(situacao_norm: str, comentario_norm: str) -> int:
    # 1) match exato em situacao
    if situacao_norm in POSITIVE_SITUACOES:
        return 1
    # 2) palavras-chave na situacao
    if situacao_norm and POS_RE.search(situacao_norm):
        return 1
    # 3) fallback: palavras-chave no comentario
    if comentario_norm and POS_RE.search(comentario_norm):
        return 1
    return 0

def main():
    if not PROSPECTS_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {PROSPECTS_CSV}")

    df = pd.read_csv(PROSPECTS_CSV)

    # garante colunas
    for c in ["vaga_code", "candidato_code", "nome", "comentario", "situacao"]:
        if c not in df.columns:
            df[c] = None

    df["vaga_code"] = df["vaga_code"].astype(str)
    df["candidato_code"] = df["candidato_code"].astype(str)

    # normaliza
    df["situacao_norm"] = df["situacao"].apply(normalize)
    df["comentario_norm"] = df["comentario"].apply(normalize)

    # calcula y
    df["y"] = [is_positive(s, c) for s, c in zip(df["situacao_norm"], df["comentario_norm"])]

    # agrega por par vaga-candidato
    grp_cols = ["vaga_code", "candidato_code"]
    agg = (
        df.groupby(grp_cols, dropna=False)
          .agg({
              "y": "max",                       # qualquer ocorrência positiva vira 1
              "nome": "first",
              "situacao_norm": lambda x: ", ".join(sorted(set([v for v in x if v]))),
              "comentario_norm": lambda x: "; ".join(sorted(set([v for v in x if v]))),
          })
          .reset_index()
    )
    agg = agg.sort_values(["y", "vaga_code", "candidato_code"], ascending=[False, True, True])

    # renomeia colunas de texto agregadas
    agg.rename(columns={
        "situacao_norm": "situacoes",
        "comentario_norm": "comentarios"
    }, inplace=True)

    agg.to_csv(OUT_CSV, index=False, encoding="utf-8")

    total = len(agg)
    pos = int(agg["y"].sum())
    neg = int(total - pos)
    print(f"[OK] Labels gerados: {OUT_CSV}")
    print(f"[INFO] Total pares vaga-candidato: {total} | Positivos(y=1): {pos} | Negativos(y=0): {neg}")

    # amostra de positivos pra validação visual
    sample_pos = agg[agg["y"] == 1].head(10)
    if not sample_pos.empty:
        print("\n[AMOSTRA POSITIVOS]")
        print(sample_pos[["vaga_code", "candidato_code", "nome", "situacoes"]])

if __name__ == "__main__":
    main()
