from pathlib import Path


def promote_model_version(
    *,
    artifacts_dir: Path,
    artifact_version: str,
) -> None:
    """
    Делает сохранённую версию текущей.

    Args:
        artifacts_dir: Корневая директория artifacts.
        artifact_version: Версия model artifacts.

    Raises:
        FileNotFoundError: Если версия не существует.
    """
    version_dir = artifacts_dir / "versions" / artifact_version

    if not version_dir.is_dir():
        raise FileNotFoundError(f"Artifact version не существует: {artifact_version}.")

    current_path = artifacts_dir / "current"
    temp_path = artifacts_dir / ".current.tmp"

    temp_path.write_text(
        artifact_version,
        encoding="utf-8",
    )

    temp_path.replace(current_path)
