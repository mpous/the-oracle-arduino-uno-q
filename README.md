# The Arduino Oracle — Matrix-Themed Edge Impulse and LLM Chat with Arduino UNO Q

This project is an Arduino UNO Q self-hosted, Matrix-themed powered by Edge Impulse keyword spotting doing model cascading to a chat interface **llama.cpp** with a small local LLM.

The Arduino Oracle answers only **2 questions** per session after detecting the keyword `Hey Arduino` so choose wisely.

## Quick Start

SETUP (on Arduino UNO Q) via SSH:

  1. python3 -m venv .venv && source .venv/bin/activate
  2. pip install -r requirements.txt
  3. python download_model.py          # downloads SmolLM2-135M
  4. python main.py                    # starts on port 5001

ACCESS:

  http://<device-ip>:5001

## Model Options

| Model | Params | Size | Quality | Speed |
|-------|--------|------|---------|-------|
| SmolLM2-135M | 135M | ~144MB | ★★☆☆ | ★★★★ |
| SmolLM2-360M | 360M | ~230MB | ★★★☆ | ★★★★ |
| Qwen2.5-0.5B | 500M | ~386MB | ★★★★ | ★★★☆ |
| TinyLlama-1.1B | 1.1B | ~669MB | ★★★★ | ★★★☆ |
