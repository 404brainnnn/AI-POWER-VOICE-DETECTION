"""Locust load test for 3-second PCM16 /score calls."""
from locust import HttpUser, between, task
import numpy as np


class CallUser(HttpUser):
    wait_time = between(0.5, 1.5)
    waveform = (0.02 * np.sin(2 * np.pi * 220 * np.arange(48_000) / 16_000)).astype("float32")
    payload = (np.clip(waveform, -1, 1) * 32767).astype("<i2").tobytes()

    @task
    def score_chunk(self):
        self.client.post("/score", data=self.payload, headers={"Content-Type":"application/octet-stream","X-Sample-Rate":"16000"}, name="/score")
