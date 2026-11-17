# Security

Set `API_KEY` to require the `X-API-Key` header on session creation and analysis. Health/readiness endpoints remain unauthenticated for load balancers. Add an identity-aware proxy, TLS, rate limiting, and secret manager integration before production. WebSocket authentication should use an authorized gateway or an API-key query/header policy; raw audio is never logged.
