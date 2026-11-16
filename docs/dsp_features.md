# DSP features

`dsp.features.extract_features` computes deterministic FFT/STFT-derived centroid, bandwidth, rolloff, flatness, flux, entropy, high-frequency ratio, energy, zero-crossing rate, Mel/DCT summaries, pitch autocorrelation proxy, voiced ratio proxy, and optional phase consistency/discontinuity. High-frequency evidence is clipped by the Nyquist limit and is not interpreted as definitive cloning evidence. Respiration is represented only by a pause/energy proxy; it is not a physiological measurement.
