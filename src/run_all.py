# src/run_all.py
import subprocess, sys

STEPS = [
    [sys.executable, "src/save_clean.py"],
    [sys.executable, "src/term_freq.py"],
    [sys.executable, "src/term_freq_by_group.py"],
    [sys.executable, "src/term_freq_by_candidato.py"],
    [sys.executable, "src/apply_scores.py"],      # << gera scores antes
    [sys.executable, "src/metrics.py"],
    [sys.executable, "src/plots.py"],
    [sys.executable, "src/generate_report.py"],   # << relatório já pega tudo
]

def run(cmd):
    print("\n$ " + " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    for c in STEPS:
        run(c)
    print("\n[OK] Pipeline concluída. Veja docs/relatorio.md")
