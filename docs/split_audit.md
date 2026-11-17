# Split leakage audit

Before reporting perfect or near-perfect results, audit the supplied train/validation/test CSVs:

```bash
python scripts/audit_splits.py \
  --train /mnt/d/data_generate/tamil_speech_dataset/finetuning_dataset/train.csv \
  --validation /mnt/d/data_generate/tamil_speech_dataset/finetuning_dataset/validation.csv \
  --test /mnt/d/data_generate/tamil_speech_dataset/finetuning_dataset/test.csv
```

The report checks exact path overlap, duplicate SHA-256 audio content, missing files, label counts, and overlap in optional `speaker_id`, `source_recording_id`, and `generator_id` columns. Empty metadata-overlap results are inconclusive if those columns are absent.
