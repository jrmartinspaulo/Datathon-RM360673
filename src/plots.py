# src/plots.py
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem interface gráfica
import matplotlib.pyplot as plt

PROC_DIR = os.path.join("data", "processed")
IN_VAGA  = os.path.join(PROC_DIR, "metrics_avg_words_by_vaga.csv")
IN_CAND  = os.path.join(PROC_DIR, "metrics_avg_words_by_candidato.csv")

OUT_DIR      = os.path.join("docs", "media")
OUT_PNG_VAGA = os.path.join(OUT_DIR, "avg_words_by_vaga.png")
OUT_PNG_CAND = os.path.join(OUT_DIR, "avg_words_by_candidato.png")

def make_bar(x, y, title, xlabel, ylabel, out_png):
    plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"[OK] Gráfico salvo em: {out_png}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # VAGA
    df_v = pd.read_csv(IN_VAGA)
    make_bar(
        x=df_v["vaga"],
        y=df_v["avg_words"],
        title="Média de palavras por vaga",
        xlabel="Vaga",
        ylabel="Média de palavras",
        out_png=OUT_PNG_VAGA
    )

    # CANDIDATO (se existir)
    if os.path.exists(IN_CAND):
        df_c = pd.read_csv(IN_CAND)
        make_bar(
            x=df_c["candidato"],
            y=df_c["avg_words"],
            title="Média de palavras por candidato",
            xlabel="Candidato",
            ylabel="Média de palavras",
            out_png=OUT_PNG_CAND
        )

if __name__ == "__main__":
    main()
