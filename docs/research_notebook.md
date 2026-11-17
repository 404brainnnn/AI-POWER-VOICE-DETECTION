# Research analysis notebook

Run `jupyter notebook data/research_analysis.ipynb` after `python -m pip install -r requirements-research.txt`. Set `VOICE_DATA_CSV` if the CSV is not `data/manifest.csv`. The notebook generates class-distribution, waveform, individual log-Mel, and optional class-mean log-Mel figures under `artifacts/research_figures/` at 300 DPI. It does not overwrite source audio or report model performance.
