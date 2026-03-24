# 🟢 The Oracle — Matrix-Themed LLM Chat

A self-hosted, Matrix-themed chat interface powered by **llama.cpp** with a small local LLM.
The Oracle answers only **2 questions** per session — choose wisely.

```
         _______________
        /               \
       /  _.----------._  \
      |  /    O    O    \  |
      | |    \______/    | |
       \  '-.________.-'  /
         \_/  ||  ||  \_/
          ___/|  ||\___
         /______|  ||______\
             _|  ||_
            |________|
```

## Architecture

```
┌──────────────┐     HTTP/SSE      ┌──────────────────┐
│  Browser UI  │ ◄──────────────►  │  Python Server    │
│  (Matrix)    │   /api/oracle     │  (FastAPI)        │
└──────────────┘                   └────────┬─────────┘
                                            │
                                   llama-cpp-python
                                            │
                                   ┌────────▼─────────┐
                                   │  GGUF Model File  │
                                   │  SmolLM2 / Qwen   │
                                   └──────────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download a model

```bash
python download_model.py
```

Recommended: **SmolLM2-360M-Instruct** (~230MB, fast on any CPU)

### 3. Run The Oracle

```bash
# With LLM model:
python oracle_server.py --model ./models/smollm2-360m-instruct.Q4_K_M.gguf

# Without a model (pre-baked responses for testing):
python oracle_server.py --fallback
```

### 4. Open in browser

Navigate to **http://localhost:8080**

## Command Reference

```bash
# Basic usage
python oracle_server.py --model ./models/your-model.gguf

# Custom port
python oracle_server.py --model ./models/model.gguf --port 3000

# GPU acceleration (if compiled with CUDA/Metal)
python oracle_server.py --model ./models/model.gguf --gpu-layers 99

# More CPU threads
python oracle_server.py --model ./models/model.gguf --threads 8

# Fallback mode (no model needed)
python oracle_server.py --fallback
```

## Model Options

| Model | Params | Size | Quality | Speed |
|-------|--------|------|---------|-------|
| SmolLM2-135M | 135M | ~144MB | ★★☆☆ | ★★★★ |
| SmolLM2-360M | 360M | ~230MB | ★★★☆ | ★★★★ |
| Qwen2.5-0.5B | 500M | ~386MB | ★★★★ | ★★★☆ |
| TinyLlama-1.1B | 1.1B | ~669MB | ★★★★ | ★★★☆ |

## Arduino UNO R4 WiFi Integration

To serve this from an Arduino UNO R4 WiFi as a companion setup:

1. Run the Python server on a Raspberry Pi or laptop on your LAN
2. The Arduino connects to WiFi and acts as a proxy/redirect
3. Or: embed a minified version of the UI in Arduino PROGMEM with
   the API calls pointing to the Python server's IP

## API Endpoint

```
POST /api/oracle
Content-Type: application/json

{ "question": "What is the nature of choice?" }

Response: Server-Sent Events stream
  data: {"token": "You"}
  data: {"token": " didn't"}
  data: {"token": " come"}
  ...
  data: [DONE]
```

## Files

```
oracle-project/
├── oracle_server.py      # Main server (FastAPI + LLM + embedded UI)
├── download_model.py      # Model download helper
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── models/                # GGUF model files (created by download_model.py)
```
