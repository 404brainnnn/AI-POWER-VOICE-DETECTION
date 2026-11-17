from __future__ import annotations


def metrics_text(request_count: int, tier_counts: dict[str,int], latency_samples: list[float]) -> str:
    lines=["# HELP voice_detector_requests_total Total score requests", "# TYPE voice_detector_requests_total counter", f"voice_detector_requests_total {request_count}"]
    for tier,count in sorted(tier_counts.items()): lines.append(f'voice_detector_tier_total{{tier="{tier}"}} {count}')
    for value in latency_samples: lines.append(f"voice_detector_latency_ms {float(value)}")
    return "\n".join(lines)+"\n"
