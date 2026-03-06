FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir pytest>=8.0.0

COPY services/knowledge/ services/knowledge/
COPY tests/__init__.py tests/__init__.py
COPY tests/knowledge/ tests/knowledge/
COPY pytest.ini .

CMD ["pytest", "tests/knowledge/", "-v"]
