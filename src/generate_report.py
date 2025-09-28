# src/generate_report.py
import os
import pandas as pd
from datetime import datetime

PROC_DIR = os.path.join("data", "processed")
RAW_PATH = os.path.join("data", "entrevistas.csv")
OUT_MD   = os.path.join("docs", "relatorio.md")

def read_csv_safe(path, required_cols=None):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if required_cols:
        miss = [c for c in required_cols if c not in df.columns]
        if miss:
            raise ValueError(f"Faltando colunas {miss} em {path}")
    return df

def md_table(df, max_rows=15):
    df = df.copy().head(max_rows)
    header = "| " + " | ".join(df.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows   = ["| " + " | ".join(map(str, r)) + " |" for r in df.values]
    return "\n".join([header, sep] + rows)

def main():
    os.makedirs("docs", exist_ok=True)

    # Dados principais e tops
    raw = read_csv_safe(RAW_PATH, required_cols=["id","candidato","vaga","pergunta","resposta","data"])
    top_all = read_csv_safe(os.path.join(PROC_DIR, "top_terms.csv"), required_cols=["term","count"])
    top_vaga = read_csv_safe(os.path.join(PROC_DIR, "top_terms_by_vaga.csv"))
    top_cand = read_csv_safe(os.path.join(PROC_DIR, "top_terms_by_candidato.csv"))

    # Métricas
    metrics_vaga = read_csv_safe(os.path.join(PROC_DIR, "metrics_avg_words_by_vaga.csv"))
    metrics_cand = read_csv_safe(os.path.join(PROC_DIR, "metrics_avg_words_by_candidato.csv"))

    # >>> NOVO: Scores (por resposta, candidato, vaga)
    scores_resp = read_csv_safe(os.path.join(PROC_DIR, "scores_by_response.csv"))
    scores_cand = read_csv_safe(os.path.join(PROC_DIR, "scores_by_candidato.csv"))
    scores_vaga = read_csv_safe(os.path.join(PROC_DIR, "scores_by_vaga.csv"))

    lines = []
    lines.append(f"# Relatório de Análise (baseline)")
    lines.append(f"_Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    # Resumo dataset
    lines.append("## Visão Geral do Dataset")
    lines.append(f"- Total de entrevistas: **{len(raw)}**")
    if "vaga" in raw.columns:
        vagas = raw["vaga"].value_counts().to_dict()
        linhas_vagas = ", ".join([f"{k}: {v}" for k, v in vagas.items()])
        lines.append(f"- Distribuição por vaga: {linhas_vagas}")
    lines.append("")

    # Amostra
    lines.append("### Amostra de linhas")
    lines.append(md_table(raw[["id","candidato","vaga","data"]], max_rows=5))
    lines.append("")

    # Top termos (geral)
    if top_all is not None:
        lines.append("## Top termos (geral)")
        lines.append(md_table(top_all, max_rows=15))
        lines.append("")

    # Top termos por vaga
    if top_vaga is not None and not top_vaga.empty:
        lines.append("## Top termos por VAGA")
        for vaga, gdf in top_vaga.groupby("vaga"):
            lines.append(f"### {vaga}")
            lines.append(md_table(gdf[["term","count"]], max_rows=10))
            lines.append("")
    # Top termos por candidato
    if top_cand is not None and not top_cand.empty:
        lines.append("## Top termos por CANDIDATO")
        for cand, gdf in top_cand.groupby("candidato"):
            lines.append(f"### {cand}")
            lines.append(md_table(gdf[["term","count"]], max_rows=10))
            lines.append("")

    # >>> NOVO: Seções de métricas
    if metrics_vaga is not None and not metrics_vaga.empty:
        lines.append("## Métricas: média de palavras por VAGA")
        lines.append(md_table(metrics_vaga, max_rows=50))
        lines.append("")

    if metrics_cand is not None and not metrics_cand.empty:
        lines.append("## Métricas: média de palavras por CANDIDATO")
        lines.append(md_table(metrics_cand, max_rows=50))
        lines.append("")

    # >>> NOVO: Seções de scores
    if scores_cand is not None and not scores_cand.empty:
        lines.append("## Scores médios por CANDIDATO (técnico, comunicação, comportamental)")
        lines.append(md_table(scores_cand, max_rows=50))
        lines.append("")

    if scores_vaga is not None and not scores_vaga.empty:
        lines.append("## Scores médios por VAGA (técnico, comunicação, comportamental)")
        lines.append(md_table(scores_vaga, max_rows=50))
        lines.append("")

    if scores_resp is not None and not scores_resp.empty:
        lines.append("### Amostra de scores por resposta")
        cols = ["id","candidato","vaga","score_tecnico","score_comunicacao","score_comportamental"]
        cols = [c for c in cols if c in scores_resp.columns]
        lines.append(md_table(scores_resp[cols], max_rows=5))
        lines.append("")

    # Imagens
    img_path = os.path.join("docs", "media", "avg_words_by_vaga.png")
    if os.path.exists(img_path):
        lines.append("## Visualização: média de palavras por vaga")
        lines.append(f"![Média de palavras por vaga]({img_path.replace(os.sep, '/')})")
        lines.append("")

    img_path2 = os.path.join("docs", "media", "avg_words_by_candidato.png")
    if os.path.exists(img_path2):
        lines.append("## Visualização: média de palavras por candidato")
        lines.append(f"![Média de palavras por candidato]({img_path2.replace(os.sep, '/')})")
        lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Relatório gerado em: {OUT_MD}")

if __name__ == "__main__":
    main()
