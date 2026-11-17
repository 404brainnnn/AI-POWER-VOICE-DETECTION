# Training workflow

Prepare a CSV with exactly these required columns:

```csv
location,label
/absolute/or/relative/path/human.wav,0
/absolute/or/relative/path/ai.wav,1
```

`0` means real human voice and `1` means AI voice. Optional metadata columns are retained for later per-condition evaluation. Audio is loaded as mono, resampled to 16 kHz, and padded/truncated to 3 seconds. The loader uses torchaudio when available and the existing standard-library WAV fallback otherwise.

Train the baseline or XLS-R model:

```bash
python scripts/train_neural.py --config configs/train.yaml --device cuda
```

Training prints live batch progress and an end-of-epoch summary containing train loss, validation EER, minDCF, false-alarm rates at 90/95/99% spoof recall, learning rate, best-checkpoint status, early-stopping patience, and elapsed time. Set `train.verbose: false` to suppress progress output or change `train.log_interval` to control batch logging frequency.

Set `train.model: xlsr_aasist` in `configs/train.yaml` to download `facebook/wav2vec2-xls-r-300m` from Hugging Face. Use a local Hugging Face cache or approved mirror for repeatable/offline runs. Checkpoints are written to `artifacts/checkpoints`; `best.pt` is selected by validation EER and `last.pt` supports resume.

Calibrate only on the held-out validation split, then evaluate once on the untouched test split:

```bash
python scripts/calibrate_neural.py --config configs/train.yaml --checkpoint artifacts/checkpoints/best.pt
python scripts/evaluate_neural.py --config configs/train.yaml --checkpoint artifacts/checkpoints/best.pt --calibration artifacts/calibration/calibration.json
```

The calibration command writes `thresholds.json`, `calibration.json`, temperature, ECE, and the three operating tiers. Metrics from toy or synthetic fixtures are pipeline checks only, not research results.

The ZenML wrapper is launchable with:

```bash
zenml init
python scripts/run_zenml_training.py --csv data/manifest.csv --config configs/train.yaml
```

The current wrapper exposes typed cacheable orchestration boundaries; the CLI contains the full local training implementation. MLflow logging and registry promotion must only be enabled for trained, validated artifacts.

With `--mlflow`, the training command logs configuration, epoch curves, and the best checkpoint. After calibration, register and explicitly promote a version:

```bash
python scripts/train_neural.py --config configs/train.yaml --device cuda --mlflow
python scripts/register_model.py --run-id RUN_ID --checkpoint artifacts/checkpoints/best.pt --thresholds artifacts/calibration/thresholds.json --model-name voice-deepfake-detector --stage Staging
```

Production promotion should be a reviewed operation and the deployed ECS/SageMaker configuration must pin the promoted artifact version.
