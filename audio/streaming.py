from __future__ import annotations

import asyncio
from collections import deque


class BoundedFrameQueue:
    def __init__(self, max_frames: int = 100): self.queue=asyncio.Queue(maxsize=max_frames); self.dropped=0
    async def put(self, frame: bytes) -> None:
        if self.queue.full():
            try: self.queue.get_nowait(); self.dropped += 1
            except asyncio.QueueEmpty: pass
        await self.queue.put(frame)
    async def get(self) -> bytes: return await self.queue.get()
