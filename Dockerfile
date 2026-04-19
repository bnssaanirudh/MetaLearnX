FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create data dirs
RUN mkdir -p data/raw data/processed data/benchmarks data/meta_db \
    experiments/configs experiments/runs experiments/reports \
    mlruns

ENV PYTHONPATH=/app
ENV METAX_ENV=production

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
