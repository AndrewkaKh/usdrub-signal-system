FROM python:3.11-slim

# catboost needs libgomp, scipy (via py_vollib) needs libgfortran + openblas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgfortran5 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ is mounted as a volume at runtime — create dirs so they exist
# even if the volume is empty on first start
RUN mkdir -p data/backfill data/exports

CMD ["python", "bot_runner.py", "--schedule"]
