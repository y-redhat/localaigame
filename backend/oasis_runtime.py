"""Persistent single-session adapter for the upstream Oasis 500M sampler.

This follows the sampling equations in etched-ai/open-oasis/generate.py, but
keeps latent history in memory and decodes only the newest frame per action.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
from torchvision.io import read_image
from torchvision.transforms.functional import resize, to_pil_image

ACTION_KEYS = [
    "inventory", "ESC", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4",
    "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
    "forward", "back", "left", "right", "cameraX", "cameraY", "jump",
    "sneak", "sprint", "swapHands", "attack", "use", "pickItem", "drop",
]


class OasisRuntime:
    def __init__(self, prompt_path: str, oasis_root: str | Path):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required. In Colab choose Runtime > Change runtime type > T4 GPU.")

        root = str(Path(oasis_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

        from dit import DiT_models
        from vae import VAE_models
        from utils import sigmoid_beta_schedule

        self.device = torch.device("cuda:0")
        self.sigmoid_beta_schedule = sigmoid_beta_schedule
        self.steps = int(os.getenv("OASIS_DDIM_STEPS", "10"))
        self.max_noise_level = 1000
        self.noise_abs_max = 20
        self.stabilization_level = 15
        self.scaling_factor = 0.07843137255

        dit_path = os.getenv("OASIS_DIT_CKPT", str(Path(root) / "oasis500m.safetensors"))
        vae_path = os.getenv("OASIS_VAE_CKPT", str(Path(root) / "vit-l-20.safetensors"))
        if not Path(dit_path).is_file() or not Path(vae_path).is_file():
            raise FileNotFoundError("Oasis checkpoints are missing. Run colab/launch.py first.")

        self.model = DiT_models["DiT-S/2"]()
        load_model(self.model, dit_path)
        self.model = self.model.to(self.device).eval()

        self.vae = VAE_models["vit-l-20-shallow-encoder"]()
        load_model(self.vae, vae_path)
        self.vae = self.vae.to(self.device).eval()

        image = read_image(prompt_path)
        image = resize(image, (360, 640)).float().div(255.0)
        image = rearrange(image, "c h w -> 1 1 c h w").to(self.device)
        with torch.inference_mode(), autocast("cuda", dtype=torch.half):
            encoded = self.vae.encode(image.reshape(1, 3, 360, 640) * 2 - 1).mean
            encoded = encoded * self.scaling_factor
        self.latents = rearrange(
            encoded, "(b t) (h w) c -> b t c h w",
            b=1, t=1, h=360 // self.vae.patch_size, w=640 // self.vae.patch_size
        )
        self.action_history = [torch.zeros(len(ACTION_KEYS), device=self.device)]
        betas = self.sigmoid_beta_schedule(self.max_noise_level).float().to(self.device)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
        self.noise_range = torch.linspace(-1, self.max_noise_level - 1, self.steps + 1, device=self.device)

    @staticmethod
    def _action_vector(state: dict[str, Any]) -> torch.Tensor:
        values = torch.zeros(len(ACTION_KEYS), dtype=torch.float32)
        for key in ("inventory", "ESC", "jump", "sneak", "sprint", "swapHands", "attack", "use", "pickItem", "drop",
                    "forward", "back", "left", "right"):
            values[ACTION_KEYS.index(key)] = 1.0 if state.get(key, False) else 0.0
        # Oasis encodes camera values into 0.5-pixel buckets; 40 is neutral.
        camera = state.get("camera", [0.0, 0.0])
        for idx, axis in enumerate(("cameraX", "cameraY")):
            raw_bucket = max(0.0, min(80.0, 40.0 + float(camera[idx]) * 2.0))
            values[ACTION_KEYS.index(axis)] = (raw_bucket - 40.0) / 40.0
        return values

    @torch.inference_mode()
    def next_frame(self, state: dict[str, Any]):
        action = self._action_vector(state).to(self.device)
        self.action_history.append(action)
        actions = torch.stack(self.action_history, dim=0).unsqueeze(0)
        b, i = self.latents.shape[:2]
        chunk = torch.randn((b, 1, *self.latents.shape[-3:]), device=self.device)
        chunk = torch.clamp(chunk, -self.noise_abs_max, self.noise_abs_max)
        self.latents = torch.cat([self.latents, chunk], dim=1)
        start_frame = max(0, i + 1 - self.model.max_frames)

        for noise_idx in reversed(range(1, self.steps + 1)):
            t_ctx = torch.full((b, i), self.stabilization_level - 1, dtype=torch.long, device=self.device)
            t = torch.full((b, 1), self.noise_range[noise_idx], dtype=torch.long, device=self.device)
            t_next = torch.full((b, 1), self.noise_range[noise_idx - 1], dtype=torch.long, device=self.device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)
            x_curr = self.latents[:, start_frame:].clone()
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]
            with autocast("cuda", dtype=torch.half):
                velocity = self.model(x_curr, t, actions[:, start_frame:i + 1])
            alpha = self.alphas_cumprod[t]
            x_start = alpha.sqrt() * x_curr - (1 - alpha).sqrt() * velocity
            x_noise = ((1 / alpha).sqrt() * x_curr - x_start) / (1 / alpha - 1).sqrt()
            alpha_next = self.alphas_cumprod[t_next].clone()
            alpha_next[:, :-1] = 1
            if noise_idx == 1:
                alpha_next[:, -1:] = 1
            predicted = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            self.latents[:, -1:] = predicted[:, -1:]

        latest = rearrange(self.latents[:, -1:], "b t c h w -> (b t) (h w) c")
        with autocast("cuda", dtype=torch.half):
            decoded = (self.vae.decode(latest / self.scaling_factor) + 1) / 2
        frame = decoded[0].float().clamp(0, 1).cpu()
        return to_pil_image(frame)
