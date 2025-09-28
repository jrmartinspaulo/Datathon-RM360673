# src/save_clean.py
import os
import pandas as pd
from preprocessing_clean import basic_clean

def main():
    in_path  = os.path.join("data", "entrevistas.csv")
    out_dir  = os.path.join("data", "processed")
    out_path = os.path.join(out_dir, "entrevistas_clean.csv")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_path)
    df["resposta_clean"] = df["resposta"].astype(str).apply(basic_clean)

    df.to_csv(out_path, index=False)
    print(f"[OK] Arquivo salvo em: {out_path}")

    # extra: uma estatística simples, só para ver algo útil
    df["resp_len"] = df["resposta_clean"].str.split().apply(len)
    print("\n=== Tamanho (nº de palavras) por entrevista ===")
    print(df[["id", "candidato", "resp_len"]])

if __name__ == "__main__":
    main()
