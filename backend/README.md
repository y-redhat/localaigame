# Backend integration notes

The API must not claim that actions control Oasis until they are connected to the model's real sampling loop.

Next implementation step:
1. Pin an upstream Oasis commit.
2. Inspect exact model initialization and sampling code.
3. Extract a session-oriented runtime that preserves latent/history state across user actions.
4. Expose action input and frame updates over WebSocket.
5. Add per-session authentication before exposing the service through Cloudflare.

Do not launch `generate.py` once per keypress: that is batch generation, not a persistent interactive world.
