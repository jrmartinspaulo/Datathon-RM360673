@'
# Deploy da API (Decision)

## Pré-requisitos
- Docker Desktop instalado
- Diretórios `./models` e `./data` contendo:
  - `models/model_cv.joblib`
  - `models/decision_threshold.json`

## Rodando com Docker
```bash
docker build -t decision-api .
docker run -d -p 8000:8000 --name decision-api \
  -v "$PWD/models:/app/models" \
  -v "$PWD/data:/app/data" \
  decision-api
