# src/term_freq_by_candidato.py
import os
import re
import pandas as pd
from collections import Counter
from preprocessing_clean import basic_clean

STOPWORDS = {
    "a","o","os","as","de","da","do","das","dos","e","em","um","uma","uns","umas",
    "para","por","com","sem","no","na","nos","nas","ao","aos","que","se",
    "seu","sua","seus","suas","meu","minha","meus","minhas","teu","tua","teus","tuas",
    "nosso","nossa","nossos","nossas","eu","voce","você","ele","ela","eles","elas",
    "este","esta","isto","esse","essa","isso","aquele","aquela","aquilo",
    "mais","menos","tambem","também","ja","já","quando","onde","como","porque",
    "entre","sobre","ate","até","ser","ter","haver","fazer","vai","vou","foi","era",
    "sao","são","sera","será","tem","tinha","seja","sendo","depois","antes"
}

def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())

def top_terms_by_group(df: pd.DataFrame, group_col: str, text_col: str, top_k: int = 10):
    rows = []
    for group, gdf in df.groupby(group_col):
        tokens = []
        for txt in gdf[text_col].fillna(""):
            toks = [t for t in tokenize(txt) if t not in STOPWORDS and len(t) > 2 and not t.isdigit()]
            tokens.extend(toks)
        for term, cnt in Counter(tokens).most_common(top_k):
            rows.append({group_col: group, "term": term, "count": cnt})
    return pd.DataFrame(rows)

def main():
    raw_path  = os.path.join("data", "entrevistas.csv")
    proc_dir  = os.path.join("data", "processed")
    proc_path = os.path.join(proc_dir, "entrevistas_clean.csv")
    out_path  = os.path.join(proc_dir, "top_terms_by_candidato.csv")

    os.makedirs(proc_dir, exist_ok=True)

    # Lê processado se existir; senão cria coluna limpa
    if os.path.exists(proc_path):
        df = pd.read_csv(proc_path)
        if "resposta_clean" not in df.columns and "resposta" in df.columns:
            df["resposta_clean"] = df["resposta"].astype(str).apply(basic_clean)
    else:
        df = pd.read_csv(raw_path)
        df["resposta_clean"] = df["resposta"].astype(str).apply(basic_clean)

    if "candidato" not in df.columns:
        raise ValueError("CSV precisa ter a coluna 'candidato'.")

    out_df = top_terms_by_group(df, group_col="candidato", text_col="resposta_clean", top_k=10)
    out_df.to_csv(out_path, index=False)

    print("\n=== Top termos por CANDIDATO ===")
    for cand, gdf in out_df.groupby("candidato"):
        print(f"\n[Candidato] {cand}")
        for _, row in gdf.iterrows():
            print(f"{row['term']:15s} {row['count']}")
    print(f"\n[OK] Salvo: {out_path}")

if __name__ == "__main__":
    main()
