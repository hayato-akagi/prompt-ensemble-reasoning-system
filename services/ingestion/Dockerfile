FROM python:3.12-slim

WORKDIR /app

COPY services/ingestion/requirements.txt services/ingestion/requirements.txt
RUN pip install --no-cache-dir -r services/ingestion/requirements.txt

COPY services/ingestion/ services/ingestion/
COPY services/knowledge/ services/knowledge/
COPY tests/__init__.py tests/__init__.py
COPY tests/ingestion/ tests/ingestion/
COPY pytest.ini .

CMD ["pytest", "tests/ingestion/", "-v"]
