# Privacy

Audio processing is memory-only and buffers are bounded and explicitly cleared on session deletion or TTL expiration. Metadata alerts may be persisted by a future storage adapter, but raw audio must never be persisted. Memory-only processing does not guarantee physical erasure: crash dumps, swap, tracing, and host-level telemetry require separate operational controls.
