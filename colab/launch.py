"""Colab launcher: install deps, download gated weights, run API + Quick Tunnel."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import secrets
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OASIS_ROOT = Path("/content/open-oasis")
DIT = OASIS_ROOT / "oasis500m.safetensors"
VAE = OASIS_ROOT / "vit-l-20.safetensors"


def run(args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def main():
    if not Path("/content").exists():
        raise SystemExit("このランチャーはGoogle Colab上で実行してください。")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("GPUが見つかりません。ColabのランタイムをT4 GPUに変更してください。")

    run([sys.executable, "-m", "pip", "install", "-q", "einops", "diffusers", "timm", "av",
         "safetensors", "huggingface_hub", "fastapi", "uvicorn[standard]", "python-multipart", "pillow"])

    if not OASIS_ROOT.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/etched-ai/open-oasis.git", str(OASIS_ROOT)])

    from huggingface_hub import hf_hub_download, login
    if not os.getenv("HF_TOKEN"):
        print("Hugging Faceのアクセストークンを入力してください（トークンはGitHubに保存されません）。")
        login()
    token = os.getenv("HF_TOKEN")
    print("Oasis 500Mの重みを確認・ダウンロードしています。モデル利用条件への同意が必要です。")
    hf_hub_download("Etched/oasis-500m", "oasis500m.safetensors", local_dir=str(OASIS_ROOT), token=token)
    hf_hub_download("Etched/oasis-500m", "vit-l-20.safetensors", local_dir=str(OASIS_ROOT), token=token)

    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        deb = "/tmp/cloudflared.deb"
        urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb", deb)
        run(["dpkg", "-i", deb])
        cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        raise SystemExit("cloudflaredをインストールできませんでした。")

    access = secrets.token_urlsafe(24)
    env = os.environ.copy()
    env.update({
        "OASIS_ROOT": str(OASIS_ROOT),
        "OASIS_DIT_CKPT": str(DIT),
        "OASIS_VAE_CKPT": str(VAE),
        "LOCALAI_ACCESS_TOKEN": access,
        "PYTHONPATH": str(ROOT / "backend") + os.pathsep + env.get("PYTHONPATH", ""),
    })
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"], cwd=ROOT / "backend", env=env)
    tunnel = subprocess.Popen([cloudflared, "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:7860"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    print("\n=== LocalAI Game ===")
    print("セッショントークン（ブラウザ画面に入力）:", access)
    print("このColabランタイムを動かしている間だけ利用できます。URLとトークンは他人に共有しないでください。")
    try:
        deadline = time.time() + 90
        for line in iter(tunnel.stdout.readline, ""):
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if match:
                print("\nブラウザで開くURL:", match.group(0))
                break
            if time.time() > deadline:
                print(line, end="")
                raise TimeoutError("Cloudflare URLの取得がタイムアウトしました。")
        while server.poll() is None and tunnel.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("停止しています…")
    finally:
        for proc in (tunnel, server):
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
