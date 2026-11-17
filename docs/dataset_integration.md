# Dataset integration

Use `datasets.manifest` for schema validation and `evaluation.splits.group_split` for speaker/source/generator-disjoint grouping. Remote SSH/SFTP access is intentionally absent. Later, implement an adapter that materializes a manifest from an authorized mount or SFTP configuration without credentials in source.

`datasets.adapters.LocalDatasetProvider` reads a validated local CSV. `RemoteSFTPProvider` is an explicit failure-safe interface and does not connect anywhere. Reproducible noise, resampling, bandwidth, and packet-loss transformations are in `datasets.augment`.
