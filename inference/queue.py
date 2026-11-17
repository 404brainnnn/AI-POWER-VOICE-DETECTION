from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class InferenceRequest:
    waveform: object
    future: asyncio.Future


class BatchingInferenceQueue:
    """Bounded async queue that batches requests and executes model work off-loop."""
    def __init__(self, predict_batch, max_batch_size: int = 8, max_wait_ms: int = 20, max_queue_size: int = 128):
        if min(max_batch_size,max_wait_ms,max_queue_size)<=0: raise ValueError("queue settings must be positive")
        self.predict_batch= predict_batch; self.max_batch_size=max_batch_size; self.max_wait_ms=max_wait_ms; self.queue=asyncio.Queue(maxsize=max_queue_size); self.worker_task=None; self.dropped=0
    async def start(self):
        if self.worker_task is None: self.worker_task=asyncio.create_task(self._worker())
    async def stop(self):
        if self.worker_task: self.worker_task.cancel(); await asyncio.gather(self.worker_task,return_exceptions=True); self.worker_task=None
    async def submit(self,waveform):
        loop=asyncio.get_running_loop(); future=loop.create_future(); request=InferenceRequest(waveform,future)
        if self.queue.full(): self.dropped+=1; raise RuntimeError("inference queue is full")
        await self.queue.put(request); return await future
    async def _worker(self):
        while True:
            first=await self.queue.get(); batch=[first]; deadline=asyncio.get_running_loop().time()+self.max_wait_ms/1000
            while len(batch)<self.max_batch_size:
                timeout=deadline-asyncio.get_running_loop().time()
                if timeout<=0: break
                try: batch.append(await asyncio.wait_for(self.queue.get(),timeout))
                except asyncio.TimeoutError: break
            try:
                results=await asyncio.to_thread(self.predict_batch,[item.waveform for item in batch])
                for item,result in zip(batch,results):
                    if not item.future.done(): item.future.set_result(result)
            except Exception as exc:
                for item in batch:
                    if not item.future.done(): item.future.set_exception(exc)
