# Load testing

Install `requirements-loadtest.txt`, start a model-loaded service, then run:

```bash
locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 20 -r 2 -t 60s
```

Locust reports p50/p95/p99 request latency. GPU utilization should be collected simultaneously from CloudWatch (`nvidia-smi` on EC2 during a controlled test). The load test does not generate or persist raw audio beyond the request body.
