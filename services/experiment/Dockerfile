FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir pytest>=8.0.0

COPY services/experiment/ services/experiment/
COPY tests/__init__.py tests/__init__.py
COPY tests/experiment/ tests/experiment/
COPY pytest.ini .

CMD ["pytest", "tests/experiment/", "-v"]
