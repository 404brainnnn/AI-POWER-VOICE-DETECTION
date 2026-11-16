"""Optional ZenML smoke pipeline; imports ZenML only when executed."""

def run_smoke_pipeline() -> dict:
    try:
        from zenml import pipeline, step
    except ImportError as exc: raise RuntimeError("install the optional mlops extra") from exc

    @step
    def source() -> int: return 3

    @step
    def double(value: int) -> int: return value * 2

    @pipeline
    def smoke(): return double(source())
    smoke()
    return {"status": "submitted", "pipeline": "smoke"}
