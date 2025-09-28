import pandas as pd
base = pd.read_csv(r"data\processed\labels_by_candidato_vaga.csv", encoding="utf-8-sig")
print("max score:", base["score_tecnico"].max())
print("qtd score>0:", (base["score_tecnico"]>0).sum())
print(base["situacao_norm"].value_counts().head(10))
