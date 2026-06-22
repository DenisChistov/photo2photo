from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"

MODEL_SOURCES: dict[str, tuple[Path, ...]] = {
    "cyclegan_export.pt": (
        PROJECT_DIR / "cyclegan_export.pt",
        PROJECT_DIR / "apple2orange.pt",
        PROJECT_DIR / "checkpoints" / "apple2orange.pt",
    ),
    "cyclegan_export_monet.pt": (
        PROJECT_DIR / "cyclegan_export_monet.pt",
        PROJECT_DIR / "monet2photo.pt",
        PROJECT_DIR / "checkpoints" / "monet2photo.pt",
    ),
}


def find_source(candidates: tuple[Path, ...]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Не найден подходящий checkpoint. Проверены пути:\n"
        f"{checked}"
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for target_name, candidates in MODEL_SOURCES.items():
        source = find_source(candidates)
        target = MODELS_DIR / target_name

        if source.resolve() == target.resolve():
            print(f"Уже на месте: {target}")
            continue

        shutil.copy2(source, target)
        size_mib = target.stat().st_size / 1024**2
        print(f"Скопирован: {source} -> {target} ({size_mib:.1f} МиБ)")

    print("\nГотово. В папке models должны быть:")
    for model_path in sorted(MODELS_DIR.glob("*.pt")):
        print(f"  - {model_path.name}")


if __name__ == "__main__":
    main()
