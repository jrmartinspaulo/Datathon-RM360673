# src/metrics.py
import os
import pandas as pd
from preprocessing_clean import basic_clean

RAW_PATH = os.path.join("data", "entrevistas.csv")
PROC_DIR = os.path.join("data", "processed")
OUT_VAGA = os.path.join(PROC_DIR, "metrics_avg_words_by_vaga.csv")
OUT_CAND = os.path.join(PROC_DIR, "metrics_avg_words_by_candidato.csv")

def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    # coluna limpa e comprimento (nº de palavras)
    df["resposta_clean"] = df["resposta"].astype(str).apply(basic_clean)
    df["resp_len"] = df["resposta_clean"].str.split().apply(len)

    # média por vaga
    avg_vaga = (
        df.groupby("vaga", as_index=False)["resp_len"]
          .mean()
          .rename(columns={"resp_len": "avg_words"})
          .sort_values("avg_words", ascending=False)
    )
    avg_vaga.to_csv(OUT_VAGA, index=False)

    # média por candidato
    avg_cand = (
        df.groupby("candidato", as_index=False)["resp_len"]
          .mean()
          .rename(columns={"resp_len": "avg_words"})
          .sort_values("avg_words", ascending=False)
    )
    avg_cand.to_csv(OUT_CAND, index=False)

    print(f"[OK] Salvo: {OUT_VAGA}")
    print(avg_vaga)
    print(f"\n[OK] Salvo: {OUT_CAND}")
    print(avg_cand)

if __name__ == "__main__":
    main()
