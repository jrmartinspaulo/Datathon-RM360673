# src/load_jsons.py
# -*- coding: utf-8 -*-
"""
Lê Jobs.json / Prospects.json / Applicants.json de data/raw,
normaliza e salva em data/interim (CSV + opcional PARQUET).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

# -----------------------
# Configuração de caminhos
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"

JOBS_PATH = RAW_DIR / "Jobs.json"
PROSPECTS_PATH = RAW_DIR / "Prospects.json"
APPLICANTS_PATH = RAW_DIR / "Applicants.json"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Utilitários
# -----------------------
def _is_list_of_primitives(v: Any) -> bool:
    return isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v)

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "__") -> Dict[str, Any]:
    """
    Achata dicionários arbitrários.
    - listas de primitivos -> string "a | b | c"
    - listas de dicts -> JSON string
    """
    items: List[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif _is_list_of_primitives(v):
            items.append((new_key, " | ".join(map(str, v))))
        elif isinstance(v, list):
            # lista complexa: mantemos como JSON para não perder informação
            items.append((new_key, json.dumps(v, ensure_ascii=False)))
        else:
            items.append((new_key, v))
    return dict(items)

def _try_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Salva PARQUET se pyarrow/fastparquet estiver disponível; caso contrário, ignora.
    """
    try:
        df.to_parquet(path, index=False)
        print(f"[OK] Parquet salvo: {path}")
    except Exception as e:
        print(f"[INFO] Não foi possível salvar PARQUET ({e}). Pulando...")

# -----------------------
# Loaders
# -----------------------
def load_jobs(path: Path) -> pd.DataFrame:
    """
    Jobs.json: dict {vaga_code: {...dados...}}
    Saída: DataFrame com coluna 'vaga_code' + colunas achatadas.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for vaga_code, payload in data.items():
        row = {"vaga_code": str(vaga_code)}
        row.update(flatten_dict(payload if isinstance(payload, dict) else {"raw": payload}))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def load_prospects(path: Path) -> pd.DataFrame:
    """
    Prospects.json (real): dict {vaga_code: {titulo, modalidade, prospects: [ {...}, ... ]}}
    Também lida com variações (lista no topo; itens string; JSON serializado).
    Saída: uma linha por par (vaga_code, candidato) — sem linhas vazias para vagas sem prospects.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Se o topo vier como lista, normaliza para um pseudo vaga_code
    if isinstance(raw, list):
        raw = {"(sem_vaga_code)": {"titulo": "", "modalidade": "", "prospects": raw}}

    rows: List[Dict[str, Any]] = []
    str_items = 0
    parsed_str_items = 0

    for vaga_code, payload in (raw or {}).items():
        titulo = ""
        modalidade = ""
        lista = []

        if isinstance(payload, dict):
            titulo = payload.get("titulo", "")
            modalidade = payload.get("modalidade", "")
            pr = payload.get("prospects", [])
            if isinstance(pr, list):
                lista = pr
            elif isinstance(pr, str):
                try:
                    lista = json.loads(pr) or []
                except Exception:
                    lista = [pr]
            else:
                # fallback: tenta achar alguma lista dentro do payload
                for k, v in payload.items():
                    if isinstance(v, list):
                        lista = v
                        break
        elif isinstance(payload, list):
            lista = payload
        else:
            lista = [payload]

        for it in (lista or []):
            item = it

            # String (comentário ou JSON serializado)
            if isinstance(item, str):
                str_items += 1
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict):
                        item = parsed
                        parsed_str_items += 1
                    else:
                        rows.append({
                            "vaga_code": str(vaga_code),
                            "titulo_vaga": titulo,
                            "modalidade": modalidade,
                            "candidato_code": None,
                            "nome": None,
                            "comentario": item,
                            "situacao": "",
                            "situacao_norm": "",
                            "recrutador": "",
                            "data_candidatura": None,
                            "ultima_atualizacao": None,
                        })
                        continue
                except Exception:
                    rows.append({
                        "vaga_code": str(vaga_code),
                        "titulo_vaga": titulo,
                        "modalidade": modalidade,
                        "candidato_code": None,
                        "nome": None,
                        "comentario": item,
                        "situacao": "",
                        "situacao_norm": "",
                        "recrutador": "",
                        "data_candidatura": None,
                        "ultima_atualizacao": None,
                    })
                    continue

            # Dict (formato esperado)
            if isinstance(item, dict):
                candidato_code = (
                    item.get("codigo")
                    or item.get("id_candidato")
                    or item.get("id")
                    or item.get("codigo_candidato")
                )
                nome = item.get("nome") or item.get("name")
                comentario = item.get("comentario") or item.get("comment") or item.get("observacao") or ""
                # Trata o typo "situacao_candidado"
                situacao_raw = (
                    item.get("situacao")
                    or item.get("situacao_candidato")
                    or item.get("situacao_candidado")
                    or item.get("status")
                    or ""
                )
                situacao_norm = str(situacao_raw).strip().lower()
                recrutador = item.get("recrutador", "")
                data_candidatura = item.get("data_candidatura")
                ultima_atualizacao = item.get("ultima_atualizacao")

                rows.append({
                    "vaga_code": str(vaga_code),
                    "titulo_vaga": titulo,
                    "modalidade": modalidade,
                    "candidato_code": str(candidato_code) if candidato_code is not None else None,
                    "nome": nome,
                    "comentario": comentario,
                    "situacao": situacao_raw,
                    "situacao_norm": situacao_norm,
                    "recrutador": recrutador,
                    "data_candidatura": data_candidatura,
                    "ultima_atualizacao": ultima_atualizacao,
                })
            else:
                rows.append({
                    "vaga_code": str(vaga_code),
                    "titulo_vaga": titulo,
                    "modalidade": modalidade,
                    "candidato_code": None,
                    "nome": None,
                    "comentario": json.dumps(item, ensure_ascii=False),
                    "situacao": "",
                    "situacao_norm": "",
                    "recrutador": "",
                    "data_candidatura": None,
                    "ultima_atualizacao": None,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["vaga_code"] = df["vaga_code"].astype(str)
        # mantém candidato_code como object para não perder zeros à esquerda
        if "candidato_code" in df.columns:
            df["candidato_code"] = df["candidato_code"].astype(object)

    print(f"[INFO] Prospects: itens string encontrados={str_items}, strings parseadas como JSON={parsed_str_items}")
    return df

def load_applicants(path: Path) -> pd.DataFrame:
    """
    Applicants.json: dict {candidato_code: {...dados...}}
    Saída: DataFrame com coluna 'candidato_code' + achatado do payload.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for cand_code, payload in data.items():
        row = {"candidato_code": str(cand_code)}
        row.update(flatten_dict(payload if isinstance(payload, dict) else {"raw": payload}))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

# -----------------------
# Execução
# -----------------------
def main() -> None:
    # Sanidade: checar arquivos
    missing = [p for p in [JOBS_PATH, PROSPECTS_PATH, APPLICANTS_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Arquivo(s) ausente(s) em data/raw: " + ", ".join(str(m) for m in missing)
        )

    print(f"[INFO] Lendo: {JOBS_PATH.name}")
    jobs_df = load_jobs(JOBS_PATH)
    print(f"[OK] Jobs carregado: {jobs_df.shape}")

    print(f"[INFO] Lendo: {PROSPECTS_PATH.name}")
    prospects_df = load_prospects(PROSPECTS_PATH)
    print(f"[OK] Prospects carregado: {prospects_df.shape}")

    print(f"[INFO] Lendo: {APPLICANTS_PATH.name}")
    applicants_df = load_applicants(APPLICANTS_PATH)
    print(f"[OK] Applicants carregado: {applicants_df.shape}")

    # Salvar CSV
    jobs_csv = INTERIM_DIR / "jobs.csv"
    prospects_csv = INTERIM_DIR / "prospects.csv"
    applicants_csv = INTERIM_DIR / "applicants.csv"

    jobs_df.to_csv(jobs_csv, index=False, encoding="utf-8")
    prospects_df.to_csv(prospects_csv, index=False, encoding="utf-8")
    applicants_df.to_csv(applicants_csv, index=False, encoding="utf-8")

    print(f"[OK] CSV salvo: {jobs_csv}")
    print(f"[OK] CSV salvo: {prospects_csv}")
    print(f"[OK] CSV salvo: {applicants_csv}")

    # Salvar PARQUET (opcional)
    _try_to_parquet(jobs_df, INTERIM_DIR / "jobs.parquet")
    _try_to_parquet(prospects_df, INTERIM_DIR / "prospects.parquet")
    _try_to_parquet(applicants_df, INTERIM_DIR / "applicants.parquet")

    print("\n[OK] Normalização concluída. Arquivos gerados em data/interim/")

if __name__ == "__main__":
    main()
