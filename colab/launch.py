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

    # GitHub may temporarily require authentication for anonymous HTTPS git clones.
    # Prefer a normal shallow clone, then fall back to the public source archive.
    if not (OASIS_ROOT / "generate.py").is_file():
        if OASIS_ROOT.exists():
            shutil.rmtree(OASIS_ROOT)
        clone_ok = False
        try:
            run(["git", "-c", "http.version=HTTP/1.1", "clone", "--depth", "1",
                 "https://github.com/etched-ai/open-oasis.git", str(OASIS_ROOT)])
            clone_ok = (OASIS_ROOT / "generate.py").is_file()
        except subprocess.CalledProcessError:
            print("通常のgit cloneに失敗。公開ソースアーカイブ方式に切り替えます。")
        if not clone_ok:
            import zipfile
            import tempfile
            archive_url = "https://codeload.github.com/etched-ai/open-oasis/zip/refs/heads/main"
            archive_path = Path(tempfile.gettempdir()) / "open-oasis-main.zip"
            try:
                urllib.request.urlretrieve(archive_url, archive_path)
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall("/content")
                extracted = Path("/content/open-oasis-main")
                if not (extracted / "generate.py").is_file():
                    # GitHub archive folders can include a commit suffix.
                    candidates = [p for p in Path("/content").glob("open-oasis-*") if (p / "generate.py").is_file()]
                    if not candidates:
                        raise RuntimeError("アーカイブ内にgenerate.pyが見つかりません。")
                    extracted = candidates[0]
                if OASIS_ROOT.exists():
                    shutil.rmtree(OASIS_ROOT)
                extracted.rename(OASIS_ROOT)
            except Exception as exc:
                raise RuntimeError(
                    "Oasisソースの取得に失敗しました。GitHubの一時的な認証要求、"
                    "ネットワーク制限、またはアーカイブURLを確認してください。"
                ) from exc
            finally:
                archive_path.unlink(missing_ok=True)

    if not (OASIS_ROOT / "generate.py").is_file():
        raise RuntimeError(f"Oasisソースが不完全です: {OASIS_ROOT}")

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
