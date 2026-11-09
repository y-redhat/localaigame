# Google Colab quick start

1. Open a fresh Colab notebook.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Accept the access conditions for `Etched/oasis-500m` on Hugging Face.
4. Run this single cell (replace the URL only if the repository owner changes):

```python
!git clone https://github.com/y-redhat/localaigame.git && cd localaigame && python colab/launch.py
```

5. Enter your Hugging Face token when prompted if you have not configured `HF_TOKEN`.
6. Open the printed Cloudflare URL and enter the session token printed by Colab.
7. Upload an initial image and press **モデルを読み込んで開始**.

The first run downloads the model checkpoints and initializes the GPU model. This can take several minutes. The interface generates one frame per keypress; it is an interactive prototype, not a guaranteed real-time game.

## Security

The tunnel is publicly reachable. The app requires a random session token, but anyone with both the URL and token can access the session. Do not publish either. Stop the Colab cell to terminate the server and tunnel.

## Notes

- This requires a CUDA GPU. CPU mode is not implemented.
- If Hugging Face reports gated/restricted access, first accept the model terms while signed in to the account used for the token.
- Colab may disconnect or reclaim the GPU at any time.
- Use a small, non-sensitive prompt image. The upload is processed in the Colab runtime.
