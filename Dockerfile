# ==============================================================================
# 🐳 MLOPS PRODUCTION DOCKERFILE - CORROSION DETECTION
# ==============================================================================
# Imagem base oficial da NVIDIA com CUDA 12.1 e Ubuntu 22.04 para suporte a GPU
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Define variáveis de ambiente para evitar interações no terminal durante o build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AIRFLOW_HOME=/app/airflow

WORKDIR /app

# 1. Instala dependências de sistema, drivers gráficos para X11/Qt e ferramentas essenciais
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    curl \
    git \
    build-essential \
    # Dependências gráficas exigidas pelo OpenCV e Labelme (X11 do WSL2)
    libxcb-xinerama0 \
    libqt5gui5 \
    libgles2-mesa-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Configura o Python 3.11 como interpretador padrão do container
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# 2. Instala o gerenciador de alta performance 'uv' da Astral de forma global
COPY --from=astral-sh/setup-uv:latest /uv /uvx /bin/

# 3. Copia os arquivos de mapeamento de dependências travadas do projeto
COPY pyproject.toml uv.lock ./

# 4. Instala as dependências de Machine Learning e Orquestração direto no sistema do container
# O uso do --system sincroniza o cache global de pacotes do container com o lockfile do uv
RUN uv pip install --system --no-cache -r pyproject.toml

# 5. Copia o restante do código fonte do repositório para dentro do container
COPY . .

# Cria os diretórios necessários para a execução dos pipelines de dados
RUN mkdir -p /app/airflow/dags \
    && mkdir -p /app/online_data \
    && mkdir -p /app/review_data \
    && cp /app/dags/*.py /app/airflow/dags/

# Expõe as portas lógicas utilizadas pelos serviços da esteira
# 8000: FastAPI Serving | 8080: Airflow WebUI | 5000: MLflow Tracking Server
EXPOSE 8000 8080 5000

# Comando padrão por defeito (inicia a API de Serving de inferência da IA)
CMD ["python", "src/serving/app.py"]
