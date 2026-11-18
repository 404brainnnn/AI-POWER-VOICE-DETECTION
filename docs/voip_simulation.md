# Offline VoIP simulation

The local simulator packetizes 16-kHz PCM into RTP-like 10/20/30-ms frames, applies deterministic loss/reordering/noise/bandwidth/gain impairments, and feeds the reconstructed frames into the existing detector service. It is a lab simulator: it does not bind a UDP port, send packets to a real call, inject RTP into a provider, or alter another party's audio.

Run a synthetic fixture:

```bash
python scripts/simulate_voip.py --mode synthetic --seconds 8 --packet-loss 0.02 --reorder-rate 0.03 --snr-db 20 --bandwidth-hz 3400
```

Run a local WAV fixture in real-time pacing:

```bash
python scripts/simulate_voip.py --wav /path/to/authorized/test.wav --mode human --realtime --frame-ms 20
```

Use a trained neural checkpoint and calibrated thresholds:

```bash
python scripts/simulate_voip.py --checkpoint artifacts/checkpoints/best.pt --thresholds artifacts/calibration/thresholds.json --model-config configs/model.yaml --mode synthetic --packet-loss 0.02 --jitter-ms 8 --snr-db 20
```

The output is written to `artifacts/voip_simulation.json`. The current no-model state reports `insufficient_evidence`; configure the trained inference runtime separately before using the simulator for neural decisions.
