# src/preprocessing_clean.py
import re
import unicodedata

def basic_clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    # normaliza acentos
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8", "ignore")
    # remove urls
    s = re.sub(r"http\S+|www\.\S+", "", s)
    # remove caracteres que nao sao letras, numeros, espaco ou pontuacao simples
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\-\:;]", " ", s)
    # colapsa espacos
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()
