from __future__ import annotations
from pathlib import Path
import json
from collections import Counter
from itertools import islice

ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = [ROOT/"data", ROOT/"data"/"raw"]

def find_file(name: str) -> Path:
    for base in DATA_DIRS:
        p = base/name
        if p.exists():
            return p
    raise FileNotFoundError(f"Não encontrei {name} em {DATA_DIRS}")

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    return (s.replace("ã","a").replace("â","a").replace("á","a")
             .replace("é","e").replace("ê","e").replace("í","i")
             .replace("ó","o").replace("ô","o").replace("õ","o")
             .replace("ú","u").replace("ç","c"))

def main():
    pros = load_json(find_file("Prospects.json"))
    # pros: dict[vaga_code] -> {"titulo":..., "modalidade":..., "prospects":[ {...}, ... ]}
    keys_seen = Counter()
    status_counter = Counter()
    sample_items = []

    for vaga_code, blob in islice(pros.items(), 0, 50):  # amostra de até 50 vagas
        if not isinstance(blob, dict): 
            continue
        lst = blob.get("prospects") or blob.get("prospeccoes") or []
        if not isinstance(lst, list): 
            continue
        for it in lst[:5]:  # pega alguns por vaga
            if not isinstance(it, dict):
                continue
            for k in it.keys():
                keys_seen[k] += 1
            # tenta status em nomes comuns
            raw = it.get("situacao_norm") or it.get("situacao") or it.get("status") or ""
            status_counter[norm(str(raw))] += 1
            if len(sample_items) < 5:
                sample_items.append(it)

    print("== Chaves mais comuns nos itens de prospects ==")
    for k, v in keys_seen.most_common(30):
        print(f"{k}: {v}")

    print("\n== Status normalizados (top 30) ==")
    for k, v in status_counter.most_common(30):
        print(f"{k!r}: {v}")

    print("\n== Amostras de itens ==")
    for i, it in enumerate(sample_items, 1):
        print(f"[{i}] { {k: type(v).__name__ for k, v in it.items()} }")
        print(f"    raw status: {it.get('situacao_norm') or it.get('situacao') or it.get('status')}\n")

if __name__ == "__main__":
    main()
