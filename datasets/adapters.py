from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .manifest import validate_records


class DatasetProvider(Protocol):
    def records(self) -> list[dict]: ...


class LocalDatasetProvider:
    def __init__(self, manifest: str | Path): self.manifest = Path(manifest)
    def records(self) -> list[dict]:
        import csv
        with self.manifest.open(newline="", encoding="utf-8") as handle: return validate_records(list(csv.DictReader(handle)))


class RemoteSFTPProvider:
    """Configuration-only placeholder; never attempts SSH automatically."""
    def __init__(self, host: str | None = None, username: str | None = None, remote_path: str | None = None): self.host, self.username, self.remote_path = host, username, remote_path
    def records(self) -> list[dict]: raise RuntimeError("remote SFTP ingestion is deferred and requires explicit deployment configuration")
