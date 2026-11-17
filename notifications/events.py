from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


@dataclass(frozen=True)
class SecurityEvent:
    event_id: str
    session_id: str
    tier: str
    recommendation: str
    created_at: datetime


class VoIPNotificationAdapter(Protocol):
    def notify(self, event: SecurityEvent) -> None: ...


class MockVoIPNotificationAdapter:
    """Safe local adapter; it never changes RTP or terminates calls."""
    def __init__(self): self.events: list[SecurityEvent] = []
    def notify(self, event: SecurityEvent) -> None: self.events.append(event)


def recommendation_for_tier(tier: str) -> str:
    return {"low": "continue_monitoring", "medium": "recommend_verification", "high": "recommend_step_up_authentication"}.get(tier, "insufficient_evidence")
