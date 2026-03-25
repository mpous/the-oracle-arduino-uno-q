"""
╔══════════════════════════════════════════════════════════════╗
║              MODEL DOWNLOADER for The Oracle                 ║
║                                                              ║
║  Downloads small GGUF models suitable for running on         ║
║  CPU-only machines or edge devices.                          ║
╚══════════════════════════════════════════════════════════════╝

Usage:
  python download_model.py                    # Interactive menu
  python download_model.py --model smollm2    # Direct pick
  python download_model.py --list             # Show options
"""

import os
import sys
import argparse
import urllib.request
import hashlib
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

# ── Available models (small enough for edge / Arduino companion) ──
MODELS = {
    "smollm2-135m": {
        "name": "SmolLM2-135M-Instruct (Q4_K_M)  ★ DEFAULT",
        "url": "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "filename": "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "size_mb": 95,
        "params": "135M",
        "description": "Default for Arduino UNO Q. Ultra-fast on CPU. ~95 MB.",
    },
    "smollm2-360m": {
        "name": "SmolLM2-360M-Instruct (Q4_K_M)",
        "url": "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf",
        "filename": "smollm2-360m-instruct.Q4_K_M.gguf",
        "size_mb": 230,
        "params": "360M",
        "description": "Better response quality. Sweet spot for Oracle personality.",
    },
    "qwen2.5-0.5b": {
        "name": "Qwen2.5-0.5B-Instruct (Q4_K_M)",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-0.5b-instruct.Q4_K_M.gguf",
        "size_mb": 386,
        "params": "500M",
        "description": "Best quality in the small range. Slightly larger.",
    },
    "tinyllama-1.1b": {
        "name": "TinyLlama-1.1B-Chat (Q4_K_M)",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-1.1b-chat.Q4_K_M.gguf",
        "size_mb": 669,
        "params": "1.1B",
        "description": "Best overall quality. Needs ~1 GB RAM.",
    },
}


def show_models():
    """Display available models."""
    print("\n\033[32m╔══════════════════════════════════════════════════════════╗")
    print("║           Available Models for The Oracle                ║")
    print("╚══════════════════════════════════════════════════════════╝\033[0m\n")

    for i, (key, m) in enumerate(MODELS.items(), 1):
        print(f"  \033[32m[{i}]\033[0m {m['name']}")
        print(f"      Params: {m['params']} | Size: ~{m['size_mb']}MB")
        print(f"      {m['description']}")
        print()


def download_with_progress(url: str, dest: Path):
    """Download a file with a progress bar."""
    print(f"\n  Downloading: {dest.name}")
    print(f"  From: {url}\n")

    class ProgressReporter:
        def __init__(self):
            self.last_percent = -1

        def __call__(self, block_count, block_size, total_size):
            if total_size > 0:
                percent = int(block_count * block_size * 100 / total_size)
                percent = min(percent, 100)
                if percent != self.last_percent:
                    self.last_percent = percent
                    bar_len = 40
                    filled = int(bar_len * percent / 100)
                    bar = '█' * filled + '░' * (bar_len - filled)
                    mb_done = block_count * block_size / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(
                        f"\r  [{bar}] {percent}%  ({mb_done:.1f}/{mb_total:.1f} MB)"
                    )
                    sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest), ProgressReporter())
        print("\n\n  \033[32m✓ Download complete!\033[0m")
        return True
    except Exception as e:
        print(f"\n\n  \033[31m✗ Download failed: {e}\033[0m")
        if dest.exists():
            dest.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(description="Download models for The Oracle")
    parser.add_argument("--model", type=str, help="Model key to download")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    if args.list:
        show_models()
        return

    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(MODELS.keys())}")
            return
        choice = args.model
    else:
        show_models()
        print("  \033[32mDefault for Arduino UNO Q: [1] SmolLM2-135M\033[0m\n")

        try:
            selection = input("  Select a model [1-4] (Enter = default SmolLM2-135M): ").strip() or "1"
            idx = int(selection) - 1
            choice = list(MODELS.keys())[idx]
        except (ValueError, IndexError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return

    model = MODELS[choice]
    dest = MODELS_DIR / model["filename"]

    if dest.exists():
        print(f"\n  Model already exists: {dest}")
        overwrite = input("  Re-download? [y/N]: ").strip().lower()
        if overwrite != "y":
            print(f"\n  Use with: python main.py --model {dest}")
            return

    success = download_with_progress(model["url"], dest)

    if success:
        print(f"\n  Model saved to: {dest}")
        print(f"\n  \033[32mRun The Oracle:\033[0m")
        print(f"  python main.py --model {dest}")
        print()


if __name__ == "__main__":
    main()
