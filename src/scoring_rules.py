# src/scoring_rules.py
import re
from collections import Counter

def tokenize(txt: str):
    """Retorna tokens minúsculos e sem acentos (assumindo que o texto já veio limpo)."""
    return re.findall(r"[a-z0-9]+", (txt or "").lower())

# ----------------------------
# DICIONÁRIOS (PT/EN) — termos unitários
# ----------------------------
TECH = {
    # linguagens
    "sql","python","java","javascript","typescript","ts","go","golang","rust","ruby","php","scala","kotlin","swift","r",
    "c","cpp","c++","csharp","c#",".net","dotnet",
    # backend / web / frameworks
    "spring","springboot","django","flask","fastapi","rails","laravel","express","node","nodejs",
    "react","next","nextjs","angular","vue","svelte",
    # dados / big data / orquestração
    "pandas","numpy","sklearn","scikit","scikit-learn","matplotlib","seaborn",
    "spark","pyspark","hadoop","hdfs","hive","presto","trino",
    "databricks","snowflake","bigquery","redshift","synapse","athena","glue",
    "kafka","kinesis","airflow","dagster","prefect","dbt","etl","elt","pipeline","batch","stream","streaming",
    # ml / modelos / métricas
    "xgboost","lightgbm","catboost","lstm","cnn","rnn","transformer","bert","gpt","embedding","embeddings",
    "feature","features","featurestore","mlops","mlflow",
    "precision","recall","f1","roc","auc","mse","rmse","mae","map","ndcg","silhouette",
    # cloud
    "aws","gcp","azure","ec2","s3","lambda","iam","sqs","sns","cloudwatch","cloudrun","pubsub","gke","eks","aks",
    "dataproc","dataflow","bigtable","cloudfunctions","appengine","cloudbuild",
    # bancos / caches / search
    "mysql","postgres","postgresql","sqlserver","mssql","oracle","mongodb","mongo","redis","elasticsearch","kibana",
    "clickhouse","cassandra","dynamodb","cosmosdb",
    # devops / infra / observabilidade
    "docker","kubernetes","k8s","terraform","ansible","jenkins","github","gitlab","git","ci","cd","cicd","helm",
    "prometheus","grafana","sonarqube","artifactory","sentry",
    # testes / qa
    "pytest","unittest","tdd","bdd","integration","integracao","unitario","unitarios","testes","teste","qa",
    # apis
    "api","rest","graphql","grpc","webhook","oauth","jwt",
    # so / shell
    "linux","shell","bash","zsh","powershell","windows","macos",
    # arquit / microserviços
    "microservicos","microservices","eventos","event-driven","orquestracao","observabilidade"
}

COMMS = {
    "comunicacao","claridade","clareza","objetividade","sintese","didatica","storytelling",
    "apresentacao","apresentei","apresentar","documentacao","documentei","documentar","relatorio","relatorios",
    "alinhamento","alinhei","alinhar","facilitacao","facilitei","facilitar","mediacao","mediador",
    "moderacao","moderador","feedback","negociacao","negociar","reuniao","reunioes",
    "stakeholders","cliente","clientes","sponsor","patrocinador","briefing","requisitos","especificacao",
    "kpis","okrs","planejamento","planning","kickoff","followup","follow","roadmap",
    "userstory","user-stories","historia","historias","criterios","criterio","aceitacao","backlog","grooming","refinement",
    "prioridades","priorizacao","priorizar"
}

BEHAV = {
    "proativo","proatividade","resiliencia","resiliente","colaboracao","colaborar","colaborativo",
    "cooperacao","cooperar","teamwork","time","equipe","trabalho","ownership","accountability",
    "autonomia","protagonismo","lideranca","lider","mentoria","mentor","mentorei","ensinei","aprendi","aprendizado",
    "curiosidade","resolvi","resolucao","resolver","problema","problemas","desafio","desafios",
    "compromisso","comprometimento","motivacao","engajamento","etica","etico",
    "organizacao","organizacao","prioridade","prioridades","priorizacao",
    "planejamento","adaptabilidade","adaptavel","flexibilidade","agilidade","agil",
    "foco","pontualidade","disciplina","persistencia","iniciativa","melhoria","inovacao","criatividade",
    "empatia","escuta","humildade","confianca"
}

# ----------------------------
# FRASES-CHAVE (contadas como “bônus”); tudo sem acento e em minúsculas
# ----------------------------
PHRASES = {
    # TECH
    "machine learning": "tech",
    "aprendizado de maquina": "tech",
    "feature store": "tech",
    "k fold": "tech",
    "cloud run": "tech",
    "ci cd": "tech",
    "event driven": "tech",
    # COMMS
    "boa comunicacao": "comms",
    "gestao de stakeholders": "comms",
    "historias de usuario": "comms",
    "criterios de aceitacao": "comms",
    # BEHAV
    "trabalho em equipe": "behv",
    "melhoria continua": "behv",
    "escuta ativa": "behv",
    "sentido de dono": "behv"
}

def score_text(clean_text: str):
    """
    Calcula três scores (0..1) com base em:
    - termos unitários nos dicionários (contagem relativa aos tokens)
    - + bônus por frases-chave (peso fixo)
    """
    text = (clean_text or "").lower()
    toks = tokenize(text)
    if not toks:
        return 0.0, 0.0, 0.0

    T = len(toks)
    counts = Counter(toks)

    tech_hits = sum(cnt for t, cnt in counts.items() if t in TECH)
    comm_hits = sum(cnt for t, cnt in counts.items() if t in COMMS)
    behv_hits = sum(cnt for t, cnt in counts.items() if t in BEHAV)

    # bônus por frases (cada ocorrência soma 2 “pontos”)
    PHRASE_BONUS = 2.0
    for phrase, cat in PHRASES.items():
        if phrase in text:
            if cat == "tech":
                tech_hits += PHRASE_BONUS
            elif cat == "comms":
                comm_hits += PHRASE_BONUS
            elif cat == "behv":
                behv_hits += PHRASE_BONUS

    # normaliza pelos tokens totais e aplica leve boost
    boost = 3.0
    tech = min(1.0, (tech_hits / T) * boost)
    comm = min(1.0, (comm_hits / T) * boost)
    behv = min(1.0, (behv_hits / T) * boost)

    return tech, comm, behv
