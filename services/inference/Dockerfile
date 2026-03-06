FROM python:3.12-slim

WORKDIR /app

# llama-cpp-python のビルドに必要なツール
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements を先にコピーしてキャッシュを活用
COPY requirements.txt .
# --prefer-binary でプレビルド wheel を優先（ビルド時間短縮）
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# アプリケーションコードをコピー
COPY config/ config/
COPY services/ services/
COPY tests/__init__.py tests/__init__.py
COPY tests/inference/ tests/inference/
COPY tests/integration/ tests/integration/
COPY pytest.ini .

# data/ はボリュームマウントで提供するためコピーしない

# デフォルトはユニットテスト（LLM 不要）
# 統合テスト実行: docker compose run --rm inference pytest -m integration -v
CMD ["pytest", "tests/inference/", "-v"]
