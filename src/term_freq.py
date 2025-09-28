# src/term_freq.py
import os
import re
import pandas as pd
from collections import Counter
from preprocessing_clean import basic_clean

# Stopwords simples (compatível com texto já "sem acento" e minúsculo)
STOPWORDS = {
    "a","o","os","as","de","da","do","das","dos","e","em","um","uma","uns","umas",
    "para","por","com","sem","no","na","nos","nas","ao","aos","à","às","que","se",
    "seu","sua","seus","suas","meu","minha","meus","minhas","teu","tua","teus","tuas",
    "nosso","nossa","nossos","nossas","eu","voce","você","ele","ela","eles","elas",
    "este","esta","isto","esse","essa","isso","aquele","aquela","aquilo",
    "mais","menos","tambem","também","ja","já","quando","onde","como","porque",
    "entre","sobre","ate","até","ser","ter","haver","fazer","vai","vou","foi","era",
    "sao","são","sera","será","tem","tinha","seja","sendo","um","uma","depois","antes"
}

def tokenize(text: str):
    # pega só letras/números (texto já limpo)
    return re.findall(r"[a-z0-9]+", text.lower())

def main():
    raw_path  = os.path.join("data", "entrevistas.csv")
    proc_dir  = os.path.join("data", "processed")
    proc_path = os.path.join(proc_dir, "entrevistas_clean.csv")
    out_terms = os.path.join(proc_dir, "top_terms.csv")

    os.makedirs(proc_dir, exist_ok=True)

    # Lê o arquivo processado se existir; senão, limpa on-the-fly
    if os.path.exists(proc_path):
        df = pd.read_csv(proc_path)
        if "resposta_clean" not in df.columns and "resposta" in df.columns:
            df["resposta_clean"] = df["resposta"].astype(str).apply(basic_clean)
    else:
        df = pd.read_csv(raw_path)
        df["resposta_clean"] = df["resposta"].astype(str).apply(basic_clean)

    # Tokeniza e remove stopwords
    all_tokens = []
    for txt in df["resposta_clean"].fillna(""):
        toks = [t for t in tokenize(txt) if t not in STOPWORDS and len(t) > 2 and not t.isdigit()]
        all_tokens.extend(toks)

    counts = Counter(all_tokens)
    top = counts.most_common(15)

    print("\n=== Top termos (unigramas) ===")
    for term, cnt in top:
        print(f"{term:15s} {cnt}")

    pd.DataFrame(top, columns=["term", "count"]).to_csv(out_terms, index=False)
    print(f"\n[OK] Salvo: {out_terms}")

if __name__ == "__main__":
    main()
