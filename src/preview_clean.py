# src/preview_clean.py
import os
import pandas as pd
from preprocessing_clean import basic_clean

def main():
    csv_path = os.path.join("data", "entrevistas.csv")
    df = pd.read_csv(csv_path)

    # cria uma coluna limpa a partir de 'resposta'
    df["resposta_clean"] = df["resposta"].astype(str).apply(basic_clean)

    print("\n=== Amostra (originais) ===")
    print(df[["id", "candidato", "resposta"]].head())

    print("\n=== Amostra (limpas) ===")
    print(df[["id", "resposta_clean"]].head())

if __name__ == "__main__":
    main()
