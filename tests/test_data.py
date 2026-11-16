from pathlib import Path

import numpy as np
import pytest

from data.ingest import build_manifest, discover_audio_files, infer_label, split_manifest
from data.preprocess import normalize_audio, pad_or_trim


def test_discover_audio_files_filters_extensions_and_hidden_paths(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "sample.wav").touch()
    (tmp_path / "real" / "notes.txt").touch()
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "hidden.wav").touch()

    assert discover_audio_files(tmp_path) == [tmp_path / "real" / "sample.wav"]


def test_infer_label_supports_common_directory_names() -> None:
    assert infer_label("dataset/real/speaker.wav") == "real"
    assert infer_label("dataset/synthetic/speaker.wav") == "deepfake"
    with pytest.raises(ValueError):
        infer_label("dataset/unknown/speaker.wav")


def test_build_manifest_writes_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "deepfake").mkdir()
    (tmp_path / "real" / "human.wav").touch()
    (tmp_path / "deepfake" / "clone.flac").touch()

    output = tmp_path / "manifests" / "all.csv"
    records = build_manifest(tmp_path, output)

    assert records == [
        {"audio_path": "deepfake/clone.flac", "label": "deepfake"},
        {"audio_path": "real/human.wav", "label": "real"},
    ]
    assert output.read_text(encoding="utf-8").splitlines()[0] == "audio_path,label"


def test_split_manifest_is_reproducible_and_stratified() -> None:
    records = [
        {"audio_path": f"{label}/{index}.wav", "label": label}
        for label in ("real", "deepfake")
        for index in range(10)
    ]

    first = split_manifest(records, validation_size=0.2, test_size=0.2, random_seed=7)
    second = split_manifest(records, validation_size=0.2, test_size=0.2, random_seed=7)

    assert first == second
    assert [len(first[name]) for name in ("train", "validation", "test")] == [12, 4, 4]
    assert {record["label"] for record in first["validation"]} == {"real", "deepfake"}


def test_pad_or_trim_and_normalize_audio() -> None:
    assert pad_or_trim(np.array([1, 2]), 4).tolist() == [1, 2, 0, 0]
    assert pad_or_trim(np.array([1, 2, 3]), 2).tolist() == [1, 2]
    normalized = normalize_audio(np.array([-2.0, 0.0, 1.0]))
    assert np.isclose(np.max(np.abs(normalized)), 0.99)
