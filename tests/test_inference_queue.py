import asyncio

from inference.queue import BatchingInferenceQueue
from inference.thresholds import ThresholdPolicy


def test_threshold_policy():
    policy=ThresholdPolicy({"no_action":0.0,"silent_flag":0.4,"play_warning":0.8})
    assert policy.tier(.2)=="no_action"; assert policy.tier(.5)=="silent_flag"; assert policy.tier(.9)=="play_warning"


def test_batching_queue_batches_requests():
    seen=[]
    def predict(batch): seen.append(len(batch)); return [{"score":float(item),"tier":"no_action"} for item in batch]
    async def run():
        queue=BatchingInferenceQueue(predict,max_batch_size=2,max_wait_ms=50); await queue.start(); values=await asyncio.gather(queue.submit(1),queue.submit(2)); await queue.stop(); return values
    assert asyncio.run(run())==[{"score":1.0,"tier":"no_action"},{"score":2.0,"tier":"no_action"}]; assert seen==[2]
