"""
IngestionService: ドキュメントを Markdown に変換して KnowledgeManager に保存する。

使い方:
    service = IngestionService()
    unit = service.ingest(Path("report.pdf"), title="Monthly Report")
    # → data/knowledge/report.md が作成される

設計方針:
  - IngestionService は KnowledgeManager に依存する
  - KnowledgeManager はコンストラクタで注入可能（テスト用）
  - 変換処理は converters.py に委譲する
"""

from __future__ import annotations

from pathlib import Path

from .converters import convert_to_markdown, SUPPORTED_EXTENSIONS


class IngestionService:
    """
    ドキュメントファイルを Markdown に変換して KnowledgeManager に登録する。

    Parameters
    ----------
    knowledge_manager : KnowledgeManager | None
        保存先の KnowledgeManager。None の場合はデフォルトパスで初期化する。
    """

    def __init__(self, knowledge_manager=None) -> None:
        if knowledge_manager is None:
            # 循環インポートを避けるためレイジーインポート
            from services.knowledge.knowledge_manager.knowledge_manager import KnowledgeManager
            knowledge_manager = KnowledgeManager()
        self._km = knowledge_manager

    def ingest(
        self,
        file_path: Path | str,
        knowledge_id: str | None = None,
        title: str = "",
        source: str = "",
        overwrite: bool = False,
    ):
        """
        ファイルを変換して KnowledgeUnit として登録する。

        Parameters
        ----------
        file_path : Path | str
            変換対象のファイルパス。
        knowledge_id : str | None
            登録する knowledge_id。None の場合はファイル名のステムを使用。
        title : str
            ナレッジのタイトル（メタデータ）。
        source : str
            ナレッジの出所（メタデータ）。省略時はファイルパスを使用。
        overwrite : bool
            同名の knowledge_id が存在する場合に上書きするか。

        Returns
        -------
        KnowledgeUnit
            登録された KnowledgeUnit。
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

        markdown = convert_to_markdown(file_path)
        kid = knowledge_id or file_path.stem
        src = source or str(file_path)

        return self._km.add(
            knowledge_id=kid,
            text=markdown,
            title=title or file_path.name,
            source=src,
            overwrite=overwrite,
        )

    def ingest_directory(
        self,
        dir_path: Path | str,
        overwrite: bool = False,
    ) -> list:
        """
        ディレクトリ内の対応フォーマットファイルをすべて変換・登録する。

        Parameters
        ----------
        dir_path : Path | str
            対象ディレクトリ。
        overwrite : bool
            既存の knowledge_id を上書きするか。

        Returns
        -------
        list[KnowledgeUnit]
            登録された KnowledgeUnit のリスト。
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"ディレクトリが見つかりません: {dir_path}")

        units = []
        errors: list[str] = []

        for file_path in sorted(dir_path.iterdir()):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                unit = self.ingest(file_path, overwrite=overwrite)
                units.append(unit)
            except Exception as e:
                errors.append(f"{file_path.name}: {e}")

        if errors:
            print(f"[IngestionService] {len(errors)} ファイルでエラー:")
            for err in errors:
                print(f"  - {err}")

        return units

    @property
    def supported_extensions(self) -> list[str]:
        return SUPPORTED_EXTENSIONS
