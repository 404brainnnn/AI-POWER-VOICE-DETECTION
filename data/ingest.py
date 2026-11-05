"""Discover labeled audio files and create reproducible dataset manifests."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
REAL_LABEL_NAMES = {"real", "human", "bonafide", "genuine"}
FAKE_LABEL_NAMES = {"fake", "deepfake", "spoof", "synthetic", "cloned"}


def discover_audio_files(dataset_dir: str | Path) -> list[Path]:
    """Return supported audio files below ``dataset_dir`` in stable order."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def infer_label(audio_path: str | Path) -> str:
    """Infer ``real`` or ``deepfake`` from a parent directory name."""
    path = Path(audio_path)
    for part in reversed(path.parts[:-1]):
        name = part.lower().replace("_", "-").replace(" ", "-")
        if name in REAL_LABEL_NAMES:
            return "real"
        if name in FAKE_LABEL_NAMES:
            return "deepfake"
    raise ValueError(
        f"Could not infer a label for {audio_path}; place it under a real or deepfake directory"
    )


def build_manifest(
    dataset_dir: str | Path,
    output_csv: str | Path | None = None,
) -> list[dict[str, str]]:
    """Build labeled records and optionally write them to a CSV manifest."""
    root = Path(dataset_dir).resolve()
    records: list[dict[str, str]] = []
    for audio_path in discover_audio_files(root):
        records.append(
            {
                "audio_path": audio_path.relative_to(root).as_posix(),
                "label": infer_label(audio_path),
            }
        )

    if output_csv is not None:
        destination = Path(output_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_path", "label"])
            writer.writeheader()
            writer.writerows(records)

    return records


def split_manifest(
    records: Iterable[dict[str, str]],
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """Split records into reproducible, label-stratified train/validation/test sets."""
    if not 0 <= validation_size < 1 or not 0 <= test_size < 1:
        raise ValueError("validation_size and test_size must be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1")

    import random
    from collections import defaultdict

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        if "label" not in record or "audio_path" not in record:
            raise ValueError("Every record must contain audio_path and label")
        grouped[record["label"]].append(dict(record))

    rng = random.Random(random_seed)
    splits = {"train": [], "validation": [], "test": []}
    for label_records in grouped.values():
        rng.shuffle(label_records)
        total = len(label_records)
        test_count = round(total * test_size)
        validation_count = round(total * validation_size)
        test_records = label_records[:test_count]
        validation_records = label_records[test_count : test_count + validation_count]
        train_records = label_records[test_count + validation_count :]
        splits["test"].extend(test_records)
        splits["validation"].extend(validation_records)
        splits["train"].extend(train_records)

    for split_records in splits.values():
        split_records.sort(key=lambda record: record["audio_path"])
    return splits
