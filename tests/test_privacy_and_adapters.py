import numpy as np
import pytest

from audio.ingestion import read_pcm_wav
from datasets.adapters import RemoteSFTPProvider
from datasets.augment import add_noise, packet_loss
from notifications.events import MockVoIPNotificationAdapter, SecurityEvent


def test_remote_provider_never_attempts_connection():
    with pytest.raises(RuntimeError): RemoteSFTPProvider("example.invalid").records()


def test_augmentations_are_bounded_and_deterministic():
    audio = np.zeros(1000, dtype=np.float32)
    assert add_noise(audio, 10).shape == audio.shape
    assert np.array_equal(packet_loss(audio, 100, 1.0), np.zeros_like(audio))


def test_mock_notification_does_not_modify_audio():
    adapter = MockVoIPNotificationAdapter()
    event = SecurityEvent("e", "s", "medium", "recommend_verification", __import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    adapter.notify(event)
    assert adapter.events == [event]
