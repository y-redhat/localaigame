# LocalAI Game — Oasis World Model prototype

Browser-based control surface for experimenting with an action-conditioned video world model on a Google Colab GPU.

## Current status

This repository is at the integration-scaffold stage, not yet a playable Oasis client. The upstream `etched-ai/open-oasis` project provides a video generation/inference pipeline, but it does not provide a browser-control API. A correct interactive adapter must preserve model/session state across actions. This project will not disguise batch MP4 generation as real-time gameplay.

## Target workflow

1. Open Colab with a T4 GPU.
2. Clone this repository and the upstream Oasis repository.
3. Accept the Oasis 500M model terms on Hugging Face and authenticate in Colab.
4. Load the model and expose a session API.
5. Start a Cloudflare Quick Tunnel and open its HTTPS URL.

## Planned architecture

- `backend/`: FastAPI session API and Oasis runtime adapter.
- `frontend/`: browser controls and generated-frame display.
- `colab/`: one-cell launcher for dependencies, model setup, server and tunnel.

## Limitations and security

- T4 performance and VRAM use must be measured; parameter count alone does not guarantee real-time generation.
- Colab runtimes are temporary and can disconnect.
- A Cloudflare Quick Tunnel URL is public to anyone who has it. Add per-session authentication before sharing it.
- Never commit Hugging Face tokens or other secrets. Use Colab Secrets or interactive authentication.

## Upstream

- Oasis: https://github.com/etched-ai/open-oasis
- Oasis 500M weights: https://huggingface.co/Etched/oasis-500m

## Development rule

Keep existing repository files and upstream code intact. Add integration files separately and patch existing files only when a verified compatibility issue requires it.
