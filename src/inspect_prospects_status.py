# src/inspect_prospects_status.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "interim" / "prospects.csv")

print("\n=== Amostra de colunas ===")
print(df.columns.tolist())

print("\n=== Top 30 valores em 'situacao' (normalizados) ===")
s = df["situacao"].fillna("").str.strip().str.lower()
print(s.value_counts().head(30))

print("\nVazios em 'situacao':", (s=="").sum())
print("Total linhas:", len(df))
