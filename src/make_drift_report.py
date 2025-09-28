# src/make_drift_report.py
from __future__ import annotations

# --- garantir pacote top-level 'src' no sys.path quando rodar como script ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------------------------------

import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# Utilitários do projeto
from src.train_baseline import (
    find_file, load_json, first_key, flatten_text_from_subdicts,
    label_from_text, score_tecnico,
    JOB_SUBDICT_KEYS, APPLICANT_SUBDICT_KEYS, APPLICANT_CV_KEYS,
)

DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML = DOCS_DIR / "drift_report.html"
OUT_JSON = DOCS_DIR / "drift_summary.json"


def build_df() -> pd.DataFrame:
    jobs = load_json(find_file("Jobs.json"))
    prospects = load_json(find_file("Prospects.json"))
    applicants = load_json(find_file("Applicants.json"))

    rows = []
    for vaga_code, blob in prospects.items():
        if not isinstance(blob, dict):
            continue
        plist = blob.get("prospects") or blob.get("prospeccoes") or []
        if not isinstance(plist, list):
            continue

        job_obj = jobs.get(str(vaga_code), {}) if isinstance(jobs, dict) else {}
        job_text = flatten_text_from_subdicts(job_obj, JOB_SUBDICT_KEYS)

        for it in plist:
            if not isinstance(it, dict):
                continue

            cand_key = first_key(it, ["codigo"])
            cand_code = str(it.get(cand_key)) if cand_key else ""

            status_key = first_key(it, ["situacao_candidado", "situacao", "status"])
            raw_status = str(it.get(status_key) or "")
            y = label_from_text(raw_status)

            cand_obj = applicants.get(cand_code, {}) if isinstance(applicants, dict) else {}
            cand_text = flatten_text_from_subdicts(cand_obj, APPLICANT_SUBDICT_KEYS, APPLICANT_CV_KEYS)

            st = score_tecnico(job_text, cand_text)

            rows.append(
                {
                    "job_text": job_text,
                    "cand_text": cand_text,
                    "situacao_norm": raw_status,
                    "score_tecnico": st,
                    "y": y,
                }
            )

    df = pd.DataFrame(rows).dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)
    return df


# ----------------------------- Plano B (Plotly) -----------------------------
def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index com bins por quantis do 'expected'."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.concatenate(([-np.inf], np.quantile(expected, qs[1:-1]), [np.inf])))
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_prop = (exp_counts + 1e-6) / (exp_counts.sum() + 1e-6 * len(exp_counts))
    act_prop = (act_counts + 1e-6) / (act_counts.sum() + 1e-6 * len(act_counts))
    psi = np.sum((act_prop - exp_prop) * np.log(act_prop / exp_prop))
    return float(psi)

def compute_ks(expected: np.ndarray, actual: np.ndarray) -> float:
    """KS 2-amostras (aprox. sem SciPy)."""
    e = np.sort(expected)
    a = np.sort(actual)
    i = j = 0
    ks = 0.0
    while i < len(e) and j < len(a):
        if e[i] <= a[j]:
            i += 1
        else:
            j += 1
        ks = max(ks, abs(i / len(e) - j / len(a)))
    return float(ks)

def render_plotly_report(ref: pd.Series, cur: pd.Series, title: str = "Drift Report (Plotly)") -> str:
    import plotly.graph_objects as go
    import plotly.offline as po

    psi = compute_psi(ref.values, cur.values, n_bins=10)
    ks = compute_ks(ref.values, cur.values)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ref, name="Referência (treino)", opacity=0.6, nbinsx=40, histnorm="probability"))
    fig.add_trace(go.Histogram(x=cur, name="Atual (validação/proxy)", opacity=0.6, nbinsx=40, histnorm="probability"))
    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis_title="score_tecnico",
        yaxis_title="Frequência relativa",
        legend=dict(orientation="h"),
        template="plotly_white",
    )

    metrics_table = pd.DataFrame({
        "métrica": ["PSI", "KS"],
        "valor": [round(psi, 4), round(ks, 4)],
        "interpretação": [
            "PSI < 0.1: baixo; 0.1–0.25: moderado; >0.25: alto",
            "KS > 0.1 costuma indicar mudança relevante",
        ],
    })

    html_plot = po.plot(fig, include_plotlyjs="cdn", output_type="div")
    html_table = metrics_table.to_html(index=False)
    html = f"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; max-width: 980px; margin: 24px auto; padding: 0 12px; }}
h1,h2 {{ margin: 0 0 12px 0; }}
.card {{ border: 1px solid #eee; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; }}
th {{ background: #fafafa; text-align: left; }}
.small {{ color:#666; font-size: 12px; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="small">Gerado em {datetime.now(timezone.utc).isoformat()}</div>
  <div class="card">
    <h2>Resumo</h2>
    {html_table}
  </div>
  <div class="card">
    <h2>Distribuições — score_tecnico</h2>
    {html_plot}
  </div>
  <div class="small">Fonte: comparação entre amostra de treino (referência) e validação (proxy de produção).</div>
</body>
</html>"""
    return html
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # 1) Dados
    df = build_df()
    from sklearn.model_selection import train_test_split
    train, current = train_test_split(df, test_size=0.2, stratify=df["y"], random_state=42)

    feat = "score_tecnico"
    ref = train[feat].reset_index(drop=True)
    cur = current[feat].reset_index(drop=True)

    # 2) Tenta Evidently primeiro (import SOMENTE AQUI, para não quebrar no topo)
    used = "plotly"
    try:
        try:
            from evidently.report import Report
        except Exception:
            from evidently.report.report import Report  # fallback
        try:
            from evidently.metric_preset import DataDriftPreset
        except Exception:
            from evidently.metrics import DataDriftPreset  # fallback

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=train[[feat]].reset_index(drop=True),
                   current_data=current[[feat]].reset_index(drop=True))
        report.save_html(str(OUT_HTML))
        used = "evidently"
    except Exception:
        # 3) Plano B: Plotly (PSI/KS)
        html = render_plotly_report(ref, cur)
        OUT_HTML.write_text(html, encoding="utf-8")

    # 4) Salva um JSON-resumo simples
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature": feat,
        "method": used,
        "n_ref": int(len(ref)),
        "n_cur": int(len(cur)),
        "note": "Se method=evidently, layout Evidently; senão, fallback Plotly com PSI e KS.",
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Drift report salvo em {OUT_HTML}  (method={used})")
    print(f"[OK] Resumo salvo em {OUT_JSON}")
