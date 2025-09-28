# Usa imagem oficial do Python
FROM python:3.10-slim

# Define diretório de trabalho
WORKDIR /app

# Copia arquivos de requisitos
COPY requirements-api.txt .

# Instala dependências
RUN pip install --no-cache-dir -r requirements-api.txt

# Copia todo o código do projeto
COPY . .

# Expõe a porta da API
EXPOSE 8000

# Comando para rodar a API com Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
