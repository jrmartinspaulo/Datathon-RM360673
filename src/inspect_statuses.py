# src/inspect_statuses.py
from __future__ import annotations
from pathlib import Path
import json
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = [ROOT / "data", ROOT / "data" / "raw"]

def find_file(name: str) -> Path:
    for base in DATA_DIRS:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Não encontrei {name} em {DATA_DIRS}")

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = (s.replace("ã","a").replace("â","a").replace("á","a")
           .replace("é","e").replace("ê","e")
           .replace("í","i")
           .replace("ó","o").replace("ô","o").replace("õ","o")
           .replace("ú","u")
           .replace("ç","c"))
    return s

def main():
    prospects = load_json(find_file("Prospects.json"))
    # prospects pode ser: dict[vaga_code] -> {prospeccoes: [...] }  OU lista
    items = []
    if isinstance(prospects, dict):
        for v in prospects.values():
            if isinstance(v, dict) and "prospeccoes" in v:
                items.extend(v["prospeccoes"])
            elif isinstance(v, list):
                items.extend(v)
    elif isinstance(prospects, list):
        for pr in prospects:
            if isinstance(pr, dict) and "prospeccoes" in pr:
                items.extend(pr["prospeccoes"])
            elif isinstance(pr, list):
                items.extend(pr)

    c = Counter()
    for p in items:
        raw = str(p.get("situacao_norm") or p.get("situacao") or p.get("status") or "")
        c[norm(raw)] += 1

    print("== Top status normalizados ==")
    for k, v in c.most_common(50):
        print(f"{k!r}: {v}")

if __name__ == "__main__":
    main()
