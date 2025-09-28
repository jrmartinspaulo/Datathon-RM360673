# src/make_tfidf_scores.py
# -*- coding: utf-8 -*-
"""
Gera data/processed/scores.csv com score_tecnico (0..1) por (vaga_code, candidato_code)
usando similaridade TF-IDF entre texto da vaga e texto do candidato.

Versão ROBUSTA:
- Usa colunas preferidas; se não existirem ou ficarem vazias, faz fallback para TODAS as colunas textuais.
- Fallback também por LINHA (se uma linha ficar vazia após o join preferido).
- Combina duas vistas TF-IDF:
  (1) palavras 1-2 (word)
  (2) char_wb 3-5 (melhor para textos curtos/ruidosos)
- Similaridade final = MÁXIMO(word_sim, char_sim).

Saída:
- data/processed/scores.csv -> ["vaga_code", "candidato_code", "score_tecnico"]
"""

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

JOBS_CSV = INTERIM_DIR / "jobs.csv"
PROSPECTS_CSV = INTERIM_DIR / "prospects.csv"
APPLICANTS_CSV = INTERIM_DIR / "applicants.csv"
OUT_SCORES = PROCESSED_DIR / "scores.csv"


# -----------------------
# Utils de texto
# -----------------------
def normalize_text(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def list_object_cols(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["object"]).columns)


def ranked_cols(df: pd.DataFrame, preferred_cols: list[str] | None, regex_priority: list[str] | None) -> list[str]:
    """
    Retorna colunas textuais para join, priorizando:
      1) preferred_cols que existirem
      2) demais colunas object do DF, ordenadas por regex_priority (se fornecida)
    Evita retornar lista vazia.
    """
    all_obj = list_object_cols(df)

    chosen = []
    if preferred_cols:
        chosen = [c for c in preferred_cols if c in df.columns and df[c].dtype == "object"]

    # Se preferred ficou vazia, pegue todas as object
    if not chosen:
        chosen = all_obj.copy()

    # Se regex_priority existir, reordena chosen por match
    if regex_priority:
        def score_col(c):
            for i, pat in enumerate(regex_priority):
                if re.search(pat, c, flags=re.IGNORECASE):
                    return i
            return len(regex_priority) + 1
        chosen = sorted(chosen, key=score_col)

    # Garante não-vazio
    if not chosen:
        chosen = all_obj.copy()

    return chosen


def join_columns_with_per_row_fallback(df: pd.DataFrame, primary_cols: list[str], fallback_cols: list[str]) -> pd.Series:
    """
    Faz join dos valores TEXTUAIS das colunas primary_cols.
    Se a linha resultar vazia, refaz o join usando fallback_cols daquela linha.
    """
    def join_from_cols(x, cols):
        vals = []
        for c in cols:
            if c in x:
                v_norm = normalize_text(x[c])
                if v_norm:
                    vals.append(v_norm)
        return " ".join(vals)

    # join primário
    s = df.apply(lambda row: join_from_cols(row, primary_cols), axis=1)

    # fallback por linha quando vazio
    empty_mask = s.str.len().fillna(0).eq(0)
    if empty_mask.any():
        s.loc[empty_mask] = df[empty_mask].apply(lambda row: join_from_cols(row, fallback_cols), axis=1)

    return s.fillna("")


def compute_similarity(job_text: str, cand_texts: list[str]) -> list[float]:
    """
    Similaridade final = max(sim_word_1_2, sim_charwb_3_5)
    Com strip_accents="unicode" e min_df=1 para evitar sparsidade excessiva.
    """
    if not job_text or not any(cand_texts):
        return [0.0] * len(cand_texts)

    corpus = [job_text] + cand_texts

    # WORD 1-2
    try:
        vec_w = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=30000,
            strip_accents="unicode",
        )
        Xw = vec_w.fit_transform(corpus)
        jw = Xw[0:1]
        cw = Xw[1:]
        sim_word = cosine_similarity(jw, cw).ravel()
    except Exception:
        sim_word = [0.0] * len(cand_texts)

    # CHAR_WB 3-5
    try:
        vec_c = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=40000,
            strip_accents="unicode",
        )
        Xc = vec_c.fit_transform(corpus)
        jc = Xc[0:1]
        cc = Xc[1:]
        sim_char = cosine_similarity(jc, cc).ravel()
    except Exception:
        sim_char = [0.0] * len(cand_texts)

    sims = [float(max(a, b)) for a, b in zip(sim_word, sim_char)]
    return sims


# -----------------------
# Main
# -----------------------
def main():
    print(f"[INFO] Lendo {JOBS_CSV.name}, {PROSPECTS_CSV.name}, {APPLICANTS_CSV.name}")

    jobs = pd.read_csv(JOBS_CSV, dtype={"vaga_code": str}, encoding="utf-8", low_memory=False)
    prospects = pd.read_csv(PROSPECTS_CSV, dtype={"vaga_code": str, "candidato_code": str}, encoding="utf-8", low_memory=False)
    applicants = pd.read_csv(APPLICANTS_CSV, dtype={"candidato_code": str}, encoding="utf-8", low_memory=False)

    # Filtra pares válidos
    prospects = prospects[
        prospects["vaga_code"].notna() & (prospects["vaga_code"].str.len() > 0) &
        prospects["candidato_code"].notna() & (prospects["candidato_code"].str.len() > 0)
    ].copy()

    # ---- JOB TEXT ----
    job_pref = [
        "titulo", "titulo_vaga", "descricao", "descricao_vaga",
        "perfil", "perfil_da_vaga", "competencias", "competencias_tecnicas",
        "atividades", "responsabilidades", "requisitos", "beneficios",
    ]
    job_regex_pri = ["titulo", "perfil", "competenc", "atividades", "requisit", "descri", "benef"]

    job_primary_cols = ranked_cols(jobs, job_pref, job_regex_pri)
    job_fallback_cols = list_object_cols(jobs)  # todas as textuais

    jobs["job_text"] = join_columns_with_per_row_fallback(jobs, job_primary_cols, job_fallback_cols)
    jobs_text = jobs[["vaga_code", "job_text"]].drop_duplicates("vaga_code").copy()

    # ---- CAND TEXT ----
    cand_pref = [
        "nome", "area_atuacao", "conhecimentos_tecnicos", "skills", "competencias",
        "experiencias", "experiencia", "historico_profissional", "cv", "resumo", "objetivo",
    ]
    cand_regex_pri = ["conhec|skill|competenc", "experien", "cv|curriculo", "area", "resumo|objetivo"]

    cand_primary_cols = ranked_cols(applicants, cand_pref, cand_regex_pri)
    cand_fallback_cols = list_object_cols(applicants)

    applicants["cand_text"] = join_columns_with_per_row_fallback(applicants, cand_primary_cols, cand_fallback_cols)
    cand_text = applicants[["candidato_code", "cand_text"]].drop_duplicates("candidato_code").copy()

    # Diagnóstico de vazios
    n_job_empty = (jobs["job_text"].str.len().fillna(0) == 0).sum()
    n_cand_empty = (applicants["cand_text"].str.len().fillna(0) == 0).sum()
    print(f"[INFO] job_text vazios: {n_job_empty} | cand_text vazios: {n_cand_empty}")

    # Base de pares
    base = prospects[["vaga_code", "candidato_code"]].drop_duplicates()
    base = base.merge(jobs_text, on="vaga_code", how="left")
    base = base.merge(cand_text, on="candidato_code", how="left")

    # Normaliza por garantia
    base["job_text"] = base["job_text"].apply(normalize_text)
    base["cand_text"] = base["cand_text"].apply(normalize_text)

    print(f"[INFO] Pares únicos (vaga, candidato): {len(base)}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    scores = []
    vagas = base["vaga_code"].dropna().unique().tolist()
    total_vagas = len(vagas)
    print(f"[INFO] Vagas distintas com candidatos: {total_vagas}")

    for i, vg in enumerate(vagas, 1):
        sub = base[base["vaga_code"] == vg]
        job_text = normalize_text(sub["job_text"].iloc[0] if len(sub) else "")
        cand_texts = [normalize_text(t) for t in sub["cand_text"].tolist()]

        sims = compute_similarity(job_text, cand_texts)
        for cand, sc in zip(sub["candidato_code"].tolist(), sims):
            scores.append((vg, cand, float(sc)))

        if i % 200 == 0 or i == total_vagas:
            print(f"[INFO] {i}/{total_vagas} vagas processadas...")

    scores_df = pd.DataFrame(scores, columns=["vaga_code", "candidato_code", "score_tecnico"])
    scores_df.to_csv(OUT_SCORES, index=False, encoding="utf-8-sig")
    print(f"[OK] Scores salvos em: {OUT_SCORES} | Linhas: {len(scores_df)}")


if __name__ == "__main__":
    main()
