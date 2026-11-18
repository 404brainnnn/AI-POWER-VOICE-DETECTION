# Frontend dashboard

The Next.js dashboard lives in `frontend/`. It is inspired by the referenced Framer/MCP visual direction through editorial typography, a dark signal card, warm accent colors, and spacious control panels; it does not copy proprietary assets.

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

The dashboard calls the FastAPI backend, never DotVoice directly. Configure `DOTVOICE_API_KEY`, `DOTVOICE_FROM_NUMBER`, and a public HTTPS `PUBLIC_BASE_URL` on the backend. The destination must be an authorized test number and the dashboard requires explicit confirmation. Synthetic mode is labeled as a disclosed test mode and uses the provider adapter's AI mode; it does not inject arbitrary RTP packets.
