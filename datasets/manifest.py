from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

FIELDS = ["dataset_id", "sample_id", "audio_path", "speaker_id", "source_recording_id", "label", "generator_id", "accent", "language", "noise_condition", "snr_db", "codec", "sample_rate", "recording_environment", "split"]


def validate_records(records: Iterable[dict]) -> list[dict]:
    result = [dict(row) for row in records]
    required = {"dataset_id", "sample_id", "audio_path", "label", "split"}
    for row in result:
        missing = required - row.keys()
        if missing: raise ValueError(f"manifest row missing {sorted(missing)}")
        if row["label"] not in {"real", "deepfake"}: raise ValueError("label must be real or deepfake")
        if row["split"] not in {"train", "validation", "test"}: raise ValueError("invalid split")
    return result


def write_manifest(records: Iterable[dict], path: str | Path) -> str:
    rows = validate_records(records)
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)
    return manifest_hash(destination)


def manifest_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
