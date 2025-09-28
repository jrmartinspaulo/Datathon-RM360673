# src/inspect_comment_keywords.py
from pathlib import Path
import pandas as pd
import re
import unicodedata
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "interim" / "prospects.csv")

def normalize(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s

df["comentario_norm"] = df["comentario"].apply(normalize)
non_empty = df[df["comentario_norm"] != ""]

print("Total com comentário não vazio:", len(non_empty))

print("\n=== Top comentários idênticos (até 30) ===")
print(non_empty["comentario_norm"].value_counts().head(30))

# Top bigrams / trigrams simples
tokens = []
for txt in non_empty["comentario_norm"]:
    words = re.findall(r"[a-z0-9]+", txt)
    for i in range(len(words)-1):
        tokens.append(words[i] + " " + words[i+1])
    for i in range(len(words)-2):
        tokens.append(words[i] + " " + words[i+1] + " " + words[i+2])

print("\n=== Top n-grams (bigrams/trigrams) (até 40) ===")
for text, cnt in Counter(tokens).most_common(40):
    print(cnt, text)
