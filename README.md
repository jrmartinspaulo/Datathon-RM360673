# Decision Match API — Datathon RM360673

Uma API simples e objetiva para predição de **match candidato–vaga**. O projeto inclui: pipeline de treino, artefatos versionáveis, API em FastAPI, testes com cobertura focada no código de produção e documentação de execução.

> **Status atual (27/09/2025):**
>
> * ✅ Modelo **real** treinado e salvo em `models/model.joblib`
> * ✅ Limite de decisão salvo em `models/decision_threshold.json`
> * ✅ API FastAPI com `/health` e `/predict`
> * ✅ Testes: **7 passando**, cobertura **88%** (somente `src/api.py`)
> * 🟨 Documentação/Docker/Deploy: ver *Roadmap*

---

## Sumário

* [Arquitetura](#arquitetura)
* [Estrutura do repositório](#estrutura-do-repositório)
* [Setup rápido](#setup-rápido)
* [Dados de entrada](#dados-de-entrada)
* [Treinamento do modelo](#treinamento-do-modelo)
* [API (FastAPI)](#api-fastapi)

  * [Executar localmente](#executar-localmente)
  * [Endpoints](#endpoints)
  * [Exemplos de requisição](#exemplos-de-requisição)
* [Testes e cobertura](#testes-e-cobertura)
* [Decisões de modelagem](#decisões-de-modelagem)
* [Docker (opcional)](#docker-opcional)
* [Roadmap / Próximos passos](#roadmap--próximos-passos)
* [Licença](#licença)

---

## Arquitetura

* **Treino** (`src/train_baseline.py`)

  * Constrói dataset a partir de `Jobs.json`, `Applicants.json`, `Prospects.json`.
  * Gera features textuais via `TF-IDF` sobre concatenação padronizada + 1 feature numérica (`score_tecnico`, similaridade de termos vaga×candidato).
  * Treina `LogisticRegression` e calcula threshold (ponto de Youden em ROC; fallback 0.59).
  * Salva artefatos em `models/`.
* **Serviço** (`src/api.py`)

  * Carrega `models/model.joblib` e `models/decision_threshold.json` ao iniciar.
  * Endpoint `/predict` recebe o contrato de produção e retorna probabilidade e rótulo binário usando o threshold.
  * `_predict_proba_flexible` aceita múltiplos formatos de entrada do pipeline (DF original; DF `{"text": ...}`; lista 1D).
* **Qualidade** (`tests/`)

  * Testes unitários e de integração (FastAPI `TestClient`).
  * Cobertura focada **apenas** no código de produção (`src/api.py`).

---

## Estrutura do repositório

```
.
├─ src/
│  ├─ api.py                # FastAPI + lógica de predição
│  ├─ train_baseline.py     # Treino do modelo (fim-a-fim)
│  └─ ...                   # Scripts auxiliares/opcionais
├─ tests/
│  ├─ test_api_internal.py  # Testes unitários da função flexível
│  └─ test_api_endpoints.py # Testes de integração (/health, /predict)
├─ data/                    # Coloque aqui os JSONs (ou em data/raw/)
│  ├─ Applicants.json
│  ├─ Jobs.json
│  └─ Prospects.json
├─ models/
│  ├─ model.joblib              # (gerado pelo treino)
│  └─ decision_threshold.json   # (gerado pelo treino)
├─ pytest.ini               # Configurações do pytest/coverage
├─ .coveragerc              # Configuração do relatório de coverage
└─ requirements.txt         # Dependências (sugestão)
```

---

## Setup rápido

Recomendado Python **3.11+** (o projeto foi validado em 3.13 também).

```bash
# Windows PowerShell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

> **Dica**: se não existir um `requirements.txt`, uma base mínima é:
>
> ```
> fastapi
> uvicorn
> scikit-learn
> pandas
> numpy
> joblib
> pytest
> pytest-cov
> pydantic
> ```

---

## Dados de entrada

Coloque os arquivos **JSON** fornecidos pelo Datathon em `data/` (ou `data/raw/`):

* `Applicants.json` (dict por candidato; inclui `infos_basicas`, `informacoes_profissionais`, `cv_pt`/`cv_en`, etc.)
* `Jobs.json` (dict por vaga; inclui `informacoes_basicas`, `perfil_vaga`, `beneficios`)
* `Prospects.json` (dict por vaga; contém lista `prospects` com itens `{ nome, codigo, situacao_candidado, comentario, ... }`)

> O script trata diferentes variações de chaves e normaliza acentos
> para mapear rótulos (positivo/negativo) a partir de `situacao_candidado`
> e, se vazio, via `comentario`. Exemplos de palavras-chave:
> **POS**: `aprovado`, `contratado`, `finalista` …
> **NEG**: `reprovado`, `descartado`, `nao selecionado` …

---

## Treinamento do modelo

Treine o modelo **fim-a-fim** (sem mocks):

```bash
python src/train_baseline.py
```

Saída típica (exemplo real do último run):

```
[OK] n=37173 | PosRate=0.585 | AUC=0.971 | thr=0.440
[OK] modelo salvo: models/model.joblib
[OK] threshold salvo: models/decision_threshold.json
```

Artefatos esperados:

* `models/model.joblib`
* `models/decision_threshold.json` com `{ "threshold": 0.44 }` (valor varia conforme dados)

> **Observação**: quando `situacao_candidado`/`comentario` não fornecem rótulo,
> o script usa *weak labels* pelos **percentis do `score_tecnico`** (padrão: ≥70% → 1; ≤30% → 0; meio é descartado),
> garantindo um dataset útil sem inventar rótulos.

---

## API (FastAPI)

### Executar localmente

```bash
uvicorn src.api:app --reload
```

* Documentação interativa: `http://127.0.0.1:8000/docs`
* Healthcheck: `http://127.0.0.1:8000/health`

### Endpoints

**GET `/health`** → status do serviço e se o modelo foi carregado.

**POST `/predict`** → corpo esperado (Pydantic `PredictRequest`):

```json
{
  "job_text": "...",
  "cand_text": "...",
  "score_tecnico": 0.25,
  "situacao_norm": "prospect",
  "mode": "raw"  
}
```

**Resposta** (`PredictResponse`):

```json
{
  "y_prob": 0.73,
  "y_pred": 1,
  "details": {
    "mode": "raw",
    "score_tecnico": 0.25,
    "situacao_norm": "prospect",
    "threshold": 0.44
  }
}
```

### Exemplos de requisição

**curl**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "job_text":"QA Selenium Java",
        "cand_text":"Selenium WebDriver, Java, testes automatizados",
        "situacao_norm":"prospect",
        "score_tecnico":0.25
      }'
```

**PowerShell**

```powershell
$body = @{
  job_text = "QA Selenium Java";
  cand_text = "Selenium WebDriver, Java, testes automatizados";
  situacao_norm = "prospect";
  score_tecnico = 0.25
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -Body $body -ContentType 'application/json'
```

---

## Testes e cobertura

Rodar toda a suíte:

```bash
pytest
```

Cobertura (relatório texto + HTML):

```bash
pytest --cov=src.api --cov-report=term-missing
pytest --cov=src.api --cov-report=html  # abre htmlcov/index.html
```

> **Configuração utilizada**
>
> **`pytest.ini`**
>
> ```ini
> [pytest]
> addopts = -q --cov=src.api --cov-report=term-missing --cov-fail-under=80
> ```
>
> **`.coveragerc`**
>
> ```ini
> [report]
> fail_under = 80
> show_missing = True
> ```

* Meta do Datathon: **≥80%** — atingida com **88%** somente em `src/api.py` (código de produção).

---

## Decisões de modelagem

* **Features**:

  * Texto concatenado padronizado: `[JOB]{job_text} [CAND]{cand_text} [SIT]{situacao_norm} [SCORE]{score_tecnico}`
  * `TF-IDF` sobre o texto concatenado.
  * `score_tecnico` como feature numérica adicional (canal paralelo com `StandardScaler`).
* **Modelo**: `LogisticRegression` (simplicidade, interpretabilidade, tempo de treino reduzido).
* **Threshold**: ponto de *Youden* em ROC, varrido em `[0.2, 0.8]` (fallback 0.59).
* **Rotulagem**:

  * Preferência por rótulos **explícitos** de `situacao_candidado` ou `comentario`.
  * *Weak labels* por percentis quando não houver rótulos explícitos — apenas extremos para reduzir ruído.
* **Resiliência**: `_predict_proba_flexible` na API tenta múltiplos formatos de entrada (DF original → DF `{"text":...}` → lista 1D).

---

## Docker (opcional)

> **Ainda não necessário** para o Datathon, mas recomendado para portabilidade.

**Exemplo de `Dockerfile`** (base slim + runtime):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY models ./models
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker run**

```bash
docker build -t decision-api .
docker run -p 8000:8000 decision-api
```

> Se quiser Compose e montagem de volumes (p/ atualizar `models/`/`data/` em tempo real), adicionar `docker-compose.yml` é trivial.

---

## Roadmap / Próximos passos

* [ ] **Documentação**: consolidar README final com prints/figuras.
* [ ] **Docker**: `Dockerfile` + (opcional) `docker-compose.yml` com volumes para `models/`.
* [ ] **Deploy local**: `docker run`/Compose; (opcional) cloud (Render/Heroku/Azure/GCP).
* [ ] **Monitoramento**: logs estruturados + verificação periódica de `/health`.
* [ ] **Entrega**: publicar no GitHub, vídeo (≤5 min) apresentando problema, solução e resultados (AUC/threshold), demo de `/predict`.

---

## Licença

Defina a licença conforme política do time/edital (ex.: MIT). Se o repositório for privado até a entrega, adicionar nota de confidencialidade.

---

## Arquivos prontos para uso

> Copie e cole **exatamente** estes conteúdos nos respectivos arquivos na raiz do projeto.

### `requirements.txt`

```txt
fastapi
uvicorn
scikit-learn
pandas
numpy
joblib
pydantic
pytest
pytest-cov
```

### `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# Dependências
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código e artefatos (modelo e threshold)
COPY src ./src
COPY models ./models

# Porta padrão
EXPOSE 8000

# Comando de execução
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `.dockerignore`

```gitignore
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
*.swp
.pytest_cache/
htmlcov/
.git/
.gitignore
.DS_Store
```

### `docker-compose.yml`

```yaml
version: "3.9"
services:
  api:
    build: .
    image: decision-api:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:rw   # atualize o modelo sem rebuild
      - ./data:/app/data:ro      # dados de referência (se necessário)
    command: uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Rodar com Docker

```bash
# build de imagem
docker build -t decision-api .

# subir container simples
docker run -p 8000:8000 decision-api

# ou via compose (com volumes)
docker compose up --build
```

---

## Métricas do último treino (27/09/2025)

* **n** = 37.173 pares
* **PosRate** ≈ 0,585
* **AUC (treino)** ≈ **0,971**

> A partir de agora, o projeto **reporta também métricas de validação** (holdout) e **K-Fold** (ver seção abaixo).

---

## Capturas sugeridas para o README (opcional)

Crie uma pasta `docs/img/` e adicione prints para enriquecer a entrega:

* `docs/img/docs-swagger.png` → tela do Swagger UI com `/predict`.
* `docs/img/health-ok.png` → retorno de `/health` com `model_loaded: true`.
* `docs/img/coverage-88.png` → recorte do relatório de cobertura.

Em seguida, referencie-as no README:

```md
![Swagger UI](docs/img/docs-swagger.png)
![Healthcheck](docs/img/health-ok.png)
![Coverage](docs/img/coverage-88.png)
```

---

## Atualização do Roadmap

* [x] **Docker**: arquivos adicionados (`Dockerfile`, `.dockerignore`, `docker-compose.yml`).
* [x] **Documentação**: README enriquecido (execução Docker, exemplos e métricas).
* [ ] **Deploy local**: testar `docker compose up` em máquina alvo.
* [ ] **GitHub público**: publicar repo com README e artefatos (sugestão: Git LFS p/ `models/*.joblib`).
* [ ] **Vídeo (≤5 min)**: roteirizar, gravar demo do `/predict` e visão geral da arquitetura.

---

# Avaliação e Confiabilidade (Validação)

O edital exige deixar claro **qual métrica** usamos e **por que o modelo é confiável**. Abaixo estão duas opções implementadas:

## 1) Holdout (validação simples)

* **O que faz**: separa automaticamente **20%** do dataset (estratificado) para validação.
* **Onde roda**: `src/train_baseline.py` (já atualizado).
* **O que salva**: `models/metrics.json` com **AUC/Accuracy/F1/Precision/Recall** no conjunto de validação.
* **Como roda**:

```bash
python src/train_baseline.py
```

* **Saídas**:

  * `models/model.joblib` e `models/decision_threshold.json` (modelo final re-treinado em 100% dos dados + threshold final)
  * `models/metrics.json` (métricas **de validação**) — exemplo de chaves:

    ```json
    {
      "n_total": 37173,
      "n_train": 29738,
      "n_val": 7435,
      "pos_rate_total": 0.585,
      "pos_rate_train": 0.585,
      "pos_rate_val": 0.585,
      "auc_val": 0.94,
      "accuracy_val": 0.88,
      "f1_val": 0.89,
      "precision_val": 0.87,
      "recall_val": 0.91,
      "threshold_train": 0.44,
      "threshold_final": 0.45,
      "timestamp": "2025-09-27T12:34:56"
    }
    ```

## 2) K-Fold (StratifiedKFold=5)

* **O que faz**: validação cruzada estratificada em 5 dobras; threshold escolhido **em cada treino** e avaliado na respectiva validação.
* **Onde roda**: `src/train_cv.py`.
* **O que salva**: `models/metrics_cv.json` com **média e desvio-padrão** de AUC/Accuracy/F1/Precision/Recall.
* **Como roda**:

```bash
python -m src.train_cv   # ou: python src/train_cv.py
```

* **Saída** (exemplo de chaves):

  ```json
  {
    "n_total": 37173,
    "n_splits": 5,
    "pos_rate_total": 0.585,
    "auc_mean": 0.94,
    "auc_std": 0.01,
    "accuracy_mean": 0.88,
    "accuracy_std": 0.01,
    "f1_mean": 0.89,
    "f1_std": 0.01,
    "precision_mean": 0.87,
    "precision_std": 0.01,
    "recall_mean": 0.91,
    "recall_std": 0.01
  }
  ```

## Por que **AUC**?

* A **AUC-ROC** mede a capacidade do modelo em **ordenar** positivos acima de negativos, **independente do threshold** — ideal quando haverá um **threshold ajustável** (como fazemos).
* Em cenários com **classes potencialmente desbalanceadas**, AUC é menos sensível a prevalência do que **accuracy**.
* Para o **uso operacional**, ainda reportamos **Accuracy/F1/Precision/Recall** considerando o **threshold escolhido no treino** (evita vazamento de informação).

> **Detalhe de projeto**: no holdout, escolhemos o threshold pelo **treino** (Youden) e avaliamos na **validação**. Depois **re-treinamos em 100%** dos dados e recalculamos o `threshold_final` para uso em produção.

---

## Resultados reais deste repositório (27/09/2025)

### Holdout (src/train_baseline.py)

* `n_total` = **37.173**, `n_train` = **29.738**, `n_val` = **7.435**
* **AUC_val** = **0,957**
* **Accuracy_val** = **0,913**
* **F1_val** = **0,927**
* **Precision_val** = **0,912**
* **Recall_val** = **0,942**
* **Threshold (treino)** = **0,440** → **Threshold final (100%)** = **0,440**

> Fonte: `models/metrics.json` gerado pelo último run.

### K-Fold 5× (src/train_cv.py)

* **AUC**: média **0,919**, desvio **0,008**
* **Accuracy**: média **0,874**, desvio **0,009**
* **F1**: média **0,919**, desvio **0,006**
* **Precision**: média **0,899**, desvio **0,006**
* **Recall**: média **0,940**, desvio **0,010**

> Fonte: `models/metrics_cv.json`. Folds individuais (log):
>
> * Fold 1 — AUC=0,925 | Acc=0,880 | F1=0,922 | Prec=0,906 | Rec=0,938 | thr=0,670
> * Fold 2 — AUC=0,907 | Acc=0,866 | F1=0,913 | Prec=0,899 | Rec=0,927 | thr=0,670
> * Fold 3 — AUC=0,927 | Acc=0,887 | F1=0,927 | Prec=0,903 | Rec=0,953 | thr=0,650
> * Fold 4 — AUC=0,922 | Acc=0,868 | F1=0,915 | Prec=0,894 | Rec=0,937 | thr=0,650
> * Fold 5 — AUC=0,914 | Acc=0,871 | F1=0,917 | Prec=0,892 | Rec=0,944 | thr=0,660

---

> **Dica**: scripts sob `src/` podem ser executados como módulo para garantir o `PYTHONPATH` correto: `python -m src.train_cv` / `python -m src.train_baseline`.

# Monitoramento & Drift

Geramos um painel de drift comparando **treino (referência)** vs **validação (proxy de produção)** para a feature `score_tecnico`.

* **Como gerar:**

  ```bash
  python -m src.make_drift_report
  ```
* **Saídas:**

  * `docs/drift_report.html` — dashboard interativo (Plotly ou Evidently, dependendo do ambiente)
  * `docs/drift_summary.json` — metadados (inclui `method` usado)

**Métricas do run atual:**

| métrica |  valor | interpretação                                       |
| ------: | :----: | --------------------------------------------------- |
|     PSI | 0.0021 | < 0.1: **baixo** · 0.1–0.25: moderado · >0.25: alto |
|      KS | 0.1580 | > 0.1 costuma indicar mudança relevante             |

> Observação: o script tenta usar **Evidently**; se indisponível na versão do Python, cai no **fallback Plotly** com **PSI** e **KS**. Abra `docs/drift_report.html` para o histograma comparativo.

---

# Exemplos rápidos da API

## Healthcheck

```bash
curl -s http://localhost:8000/health | jq
# Se a 8000 estiver ocupada e você estiver no Docker: http://localhost:8001/health
```

## Predict (payload de exemplo)

```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "job_text": "QA Selenium Java",
        "cand_text": "Selenium WebDriver, Java, testes automatizados",
        "situacao_norm": "prospect",
        "score_tecnico": 0.25,
        "mode": "raw"
      }' | jq
```

No PowerShell:

```powershell
$body = @{job_text="QA Selenium Java"; cand_text="Selenium WebDriver, Java, testes automatizados"; situacao_norm="prospect"; score_tecnico=0.25; mode="raw"} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body $body -ContentType 'application/json'
```

---

# Docker (execução local)

Build e execução:

```bash
docker build -t decision-api .
# se a porta 8000 já estiver em uso, mapeie para 8001 do host
# docker run -p 8001:8000 decision-api

docker run -p 8000:8000 decision-api
```

Acesse `http://localhost:8000/docs` (ou `:8001` se usou o mapeamento alternativo).

---

# Estado da Entrega (agora)

* ✅ **Pipeline de ML** (pré-processamento, features, treino) + modelo salvo (`models/model.joblib`).
* ✅ **Validação (Holdout)** com métricas em `models/metrics.json`.
* ✅ **Validação (K-Fold 5×)** com médias/DP em `models/metrics_cv.json`.
* ✅ **API FastAPI** (`/health`, `/predict`) + **testes** com **88%** de cobertura.
* ✅ **Docker** (imagem `decision-api`) — execução local.
* ✅ **Monitoramento**: `docs/drift_report.html` (Plotly fallback com PSI/KS).
* ⏳ **Publicação**: subir no **GitHub** (usar **Git LFS** para `models/*.joblib`) e anexar prints (`/health`, `/docs`, `/predict`).
* ⏳ **Vídeo** (≤ 5 min): contexto → solução → demo → resultados → drift.
