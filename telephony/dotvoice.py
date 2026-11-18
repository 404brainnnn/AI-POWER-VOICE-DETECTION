from __future__ import annotations

import re
from dataclasses import dataclass


E164 = re.compile(r"^\+[1-9]\d{7,14}$")


@dataclass
class DotVoiceConfig:
    api_key: str | None
    from_number: str | None
    base_url: str = "https://api.dotvoice.ai/v1"
    public_base_url: str = ""


class DotVoiceClient:
    def __init__(self, config: DotVoiceConfig): self.config=config
    def configured(self) -> bool: return bool(self.config.api_key and self.config.from_number and self.config.public_base_url)
    async def create_call(self, to_number: str, mode: str, consent_confirmed: bool) -> dict:
        if not consent_confirmed: raise ValueError("explicit call consent confirmation is required")
        if not E164.fullmatch(to_number): raise ValueError("destination must be an E.164 phone number")
        if mode not in {"human", "synthetic"}: raise ValueError("mode must be human or synthetic")
        if not self.configured(): raise RuntimeError("DotVoice is not configured; set DOTVOICE_API_KEY, DOTVOICE_FROM_NUMBER, and PUBLIC_BASE_URL")
        try: import httpx
        except ImportError as exc: raise RuntimeError("install httpx to use the DotVoice adapter") from exc
        payload={"from":self.config.from_number,"to":to_number,"answer_url":f"{self.config.public_base_url.rstrip('/')}/api/v1/telephony/dotvoice/answer","stream_url":f"{self.config.public_base_url.rstrip('/')}/api/v1/telephony/dotvoice/media","record":False,"ai_mode":mode=="synthetic"}
        async with httpx.AsyncClient(timeout=20) as client:
            response=await client.post(f"{self.config.base_url.rstrip('/')}/calls",headers={"Authorization":f"Bearer {self.config.api_key}","Content-Type":"application/json"},json=payload); response.raise_for_status(); return response.json()
