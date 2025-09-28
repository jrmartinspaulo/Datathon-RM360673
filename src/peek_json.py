# src/peek_json.py
from __future__ import annotations
from pathlib import Path
import json
from itertools import islice

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

def head_dict(d: dict, n=3):
    return {k: type(v).__name__ for k, v in islice(d.items(), n)}

def head_list(xs: list, n=1):
    return xs[:n]

def describe(name: str):
    path = find_file(name)
    obj = load_json(path)
    print(f"\n=== {name} ===")
    print(f"type: {type(obj).__name__}")
    if isinstance(obj, dict):
        print(f"len(dict): {len(obj)}")
        print(f"top-keys(types): {head_dict(obj, 10)}")
        # amostra de um valor
        val = next(iter(obj.values()))
        print(f"sample value type: {type(val).__name__}")
        if isinstance(val, dict):
            print(f"sample value keys: {list(islice(val.keys(), 20))}")
        if isinstance(val, list):
            print(f"sample list len: {len(val)}")
            if val:
                v0 = val[0]
                print(f"sample[0] type: {type(v0).__name__}")
                if isinstance(v0, dict):
                    print(f"sample[0] keys: {list(v0.keys())}")
    elif isinstance(obj, list):
        print(f"len(list): {len(obj)}")
        head = head_list(obj, 2)
        for i, it in enumerate(head):
            print(f"[{i}] type: {type(it).__name__}")
            if isinstance(it, dict):
                print(f"[{i}] keys: {list(islice(it.keys(), 30))}")
                # tenta achar sublista de prospecções
                for k, v in it.items():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        print(f"[{i}] nested list '{k}' item0 keys: {list(v[0].keys())}")

if __name__ == "__main__":
    for fname in ["Prospects.json", "Jobs.json", "Applicants.json"]:
        try:
            describe(fname)
        except Exception as e:
            print(f"{fname}: {e}")
