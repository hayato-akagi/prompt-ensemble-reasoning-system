FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir huggingface-hub>=0.23.0

COPY scripts/download_model.py scripts/download_model.py
COPY config/models.json config/models.json

# data/models と config は docker-compose でボリュームマウント

ENTRYPOINT ["python", "scripts/download_model.py"]
CMD ["--list"]
