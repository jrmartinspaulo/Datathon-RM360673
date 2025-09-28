# src/apply_scores.py
import os
import pandas as pd
from preprocessing_clean import basic_clean
from scoring_rules import score_text

RAW = os.path.join("data", "entrevistas.csv")
PROC_DIR = os.path.join("data", "processed")
OUT_RESP = os.path.join(PROC_DIR, "scores_by_response.csv")
OUT_CAND = os.path.join(PROC_DIR, "scores_by_candidato.csv")
OUT_VAGA = os.path.join(PROC_DIR, "scores_by_vaga.csv")

def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    df = pd.read_csv(RAW)
    df["resposta_clean"] = df["resposta"].astype(str).apply(basic_clean)

    # scores por resposta
    scores = df["resposta_clean"].apply(lambda t: score_text(t))
    df[["score_tecnico","score_comunicacao","score_comportamental"]] = pd.DataFrame(scores.tolist(), index=df.index)

    # salva por resposta
    cols = ["id","candidato","vaga","score_tecnico","score_comunicacao","score_comportamental","resposta_clean"]
    df[cols].to_csv(OUT_RESP, index=False)
    print(f"[OK] Salvo: {OUT_RESP}")

    # agrega por candidato
    by_cand = (
        df.groupby("candidato", as_index=False)[["score_tecnico","score_comunicacao","score_comportamental"]]
          .mean()
          .sort_values(["score_tecnico","score_comunicacao","score_comportamental"], ascending=False)
    )
    by_cand.to_csv(OUT_CAND, index=False)
    print(f"[OK] Salvo: {OUT_CAND}")
    print(by_cand)

    # agrega por vaga
    by_vaga = (
        df.groupby("vaga", as_index=False)[["score_tecnico","score_comunicacao","score_comportamental"]]
          .mean()
          .sort_values(["score_tecnico","score_comunicacao","score_comportamental"], ascending=False)
    )
    by_vaga.to_csv(OUT_VAGA, index=False)
    print(f"[OK] Salvo: {OUT_VAGA}")
    print(by_vaga)

if __name__ == "__main__":
    main()
