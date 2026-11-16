# Streaming

The REST path accepts bounded JSON audio arrays. The WebSocket accepts JSON frames or raw float32 binary frames and emits decisions only for complete speech-positive 3-second windows. The first decision necessarily includes approximately 3 seconds of observation delay; processing latency is reported separately.
