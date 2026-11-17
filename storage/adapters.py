from __future__ import annotations

from typing import Any


class OptionalRedisState:
    def __init__(self, url: str): self.url, self.client = url, None
    def connect(self) -> bool:
        try:
            import redis
            self.client = redis.Redis.from_url(self.url, decode_responses=True); self.client.ping(); return True
        except Exception: self.client = None; return False
    def set(self, key: str, value: str, ttl_seconds: int = 300) -> bool:
        return bool(self.client and self.client.set(key, value, ex=ttl_seconds))


class OptionalMongoEvents:
    def __init__(self, url: str, database: str): self.url, self.database, self.collection = url, database, None
    def connect(self) -> bool:
        try:
            from pymongo import MongoClient
            client = MongoClient(self.url, serverSelectionTimeoutMS=1000); client.admin.command("ping")
            self.collection = client[self.database]["security_events"]; self.collection.create_index("created_at", expireAfterSeconds=2_592_000); return True
        except Exception: self.collection = None; return False
    def insert_metadata(self, event: dict[str, Any]) -> bool:
        if not self.collection: return False
        safe = {key: value for key, value in event.items() if key not in {"audio", "waveform", "raw_audio"}}
        self.collection.insert_one(safe); return True
