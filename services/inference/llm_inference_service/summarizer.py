"""
SummarizationService: ローカル LLM を使ってナレッジ・ログを要約する。

- ナレッジ: 専門用語・エラーコード・数値を保持した要約
- ログ:     エラーコード・異常値・警告など重要情報を落とさない要約

コンテキスト窓を超えるテキストはチャンク分割して個別要約した後、
結合テキストが収まる場合は最終まとめ要約を行う。
"""
from __future__ import annotations

_KNOWLEDGE_PROMPT = """\
以下のテキストを要約してください。
専門用語・エラーコード・数値・手順は正確に保持してください。

[テキスト]
{text}

要約:"""

_LOG_PROMPT = """\
以下のログを要約してください。
エラーコード・エラーメッセージ・異常値・警告など重要な情報は必ず含め、省略しないでください。

[ログ]
{text}

要約:"""

_SAFETY_MARGIN = 50


class SummarizationService:
    """
    ローカル LLM を使ってテキストを要約する。LLMClient は遅延ロード。

    Parameters
    ----------
    client : LLMClient | None
        既存の LLMClient を再利用する場合に渡す。None の場合は遅延ロード。
    """

    def __init__(self, client=None) -> None:
        self._client = client

    def _get_client(self):
        if self._client is None:
            from .llm_client import LLMClient
            self._client = LLMClient()
        return self._client

    def token_count(self, text: str) -> int:
        """モデルのトークナイザーで正確なトークン数を返す。"""
        return self._get_client().token_count(text)

    def needs_summarization(self, text: str, token_limit: int) -> bool:
        return self.token_count(text) > token_limit

    def _available_chunk_tokens(self, prompt_template: str, client) -> int:
        """プロンプトテンプレートのオーバーヘッドを引いた、テキスト部分に使えるトークン数。"""
        overhead = client.token_count(prompt_template.format(text=""))
        return client.context_size() - overhead - client._max_tokens - _SAFETY_MARGIN

    def _summarize_in_chunks(self, text: str, prompt_template: str, chunk_tokens: int, client) -> str:
        """テキストをチャンク分割して要約し、必要なら最終まとめ要約を行う。"""
        text_tokens = client.token_count(text)
        chars_per_token = max(1.0, len(text) / text_tokens)
        chunk_chars = max(1, int(chunk_tokens * chars_per_token))

        chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
        partials = [
            client.generate(prompt_template.format(text=chunk)).strip()
            for chunk in chunks
        ]

        if len(partials) == 1:
            return partials[0]

        combined = "\n".join(partials)
        if client.token_count(combined) <= chunk_tokens:
            return client.generate(prompt_template.format(text=combined)).strip()

        return combined

    def summarize_knowledge(self, text: str) -> str:
        """ナレッジテキストを要約する（専門用語・数値を保持）。"""
        client = self._get_client()
        available = self._available_chunk_tokens(_KNOWLEDGE_PROMPT, client)
        if client.token_count(text) <= available:
            return client.generate(_KNOWLEDGE_PROMPT.format(text=text)).strip()
        return self._summarize_in_chunks(text, _KNOWLEDGE_PROMPT, available, client)

    def summarize_log(self, text: str) -> str:
        """ログを要約する（エラーコード・異常値を落とさない）。"""
        client = self._get_client()
        available = self._available_chunk_tokens(_LOG_PROMPT, client)
        if client.token_count(text) <= available:
            return client.generate(_LOG_PROMPT.format(text=text)).strip()
        return self._summarize_in_chunks(text, _LOG_PROMPT, available, client)

    def maybe_summarize_knowledge(self, text: str, token_limit: int) -> tuple[str, bool]:
        """
        token_limit を超える場合のみ要約する。
        Returns (result_text, was_summarized).
        """
        if self.needs_summarization(text, token_limit):
            return self.summarize_knowledge(text), True
        return text, False

    def maybe_summarize_log(self, text: str, token_limit: int) -> tuple[str, bool]:
        """
        token_limit を超える場合のみログを要約する。
        Returns (result_text, was_summarized).
        """
        if self.needs_summarization(text, token_limit):
            return self.summarize_log(text), True
        return text, False
