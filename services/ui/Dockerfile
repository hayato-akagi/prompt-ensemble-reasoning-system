FROM python:3.12-slim

WORKDIR /app

# llama-cpp-python のビルドに必要なツール
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY services/ui/requirements.txt services/ui/requirements.txt
RUN pip install --no-cache-dir --prefer-binary -r services/ui/requirements.txt

# サービスコードをコピー（data/ と config/ はボリュームで提供）
COPY services/ services/

# /app を Python パスに追加
ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "services/ui/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
