"""
╔══════════════════════════════════════════════════════════════╗
║               THE ORACLE — Arduino App Lab                   ║
║      Flask + llama.cpp + SmolLM2-135M (GGUF, local)         ║
║                                                              ║
║  Serves the Matrix-themed Oracle chat UI and bridges         ║
║  visitor questions to a local LLM running via llama.cpp.     ║
╚══════════════════════════════════════════════════════════════╝

ARCHITECTURE:
  ┌──────────────┐   HTTP/SSE    ┌──────────────────────┐
  │  Browser UI  │ ◄──────────► │  Flask Server (5001)  │
  │  (Oracle)    │  /api/oracle  │      (main.py)        │
  └──────────────┘               └──────────┬───────────┘
                                             │
                                    llama-cpp-python
                                             │
                                   ┌─────────▼──────────┐
                                   │  SmolLM2-135M.gguf  │
                                   │  (local inference)  │
                                   └────────────────────┘

SETUP (on Arduino UNO Q):
  1. python3 -m venv .venv && source .venv/bin/activate
  2. pip install -r requirements.txt
  3. python download_model.py          # downloads SmolLM2-135M
  4. python main.py                    # starts on port 5001

ACCESS:
  http://<device-ip>:5001
"""

import argparse
import json
import os
import sys
import glob
import logging
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[\033[32m%(asctime)s\033[0m] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("oracle")

# ── Dependency checks ─────────────────────────────────────────────────────────
try:
    from flask import Flask, request, Response, stream_with_context, render_template, jsonify
except ImportError:
    print("ERROR: Flask not found. Run: pip install flask")
    sys.exit(1)

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
#  ORACLE SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

ORACLE_SYSTEM_PROMPT = """You are The Oracle from The Matrix. You are ancient, wise, warm yet enigmatic.
You speak in cryptic but meaningful sentences. You see the future but reveal only what the visitor needs to hear.

Rules:
- Respond with 1-2 short, evocative sentences. Never more.
- Be mysterious and philosophical. Reference choice, fate, free will, purpose.
- Never break character. You ARE The Oracle.
- Never mention being an AI, a language model, or anything technical. You are The Oracle.
- Occasionally reference cookies, candy, or your kitchen — you're homey yet unsettling.
- Speak with quiet confidence. You already know the answer before they ask.
- Use present tense. You see, you know, you feel.
- Do not use quotation marks around your own words.

Style examples:
- You didn't come here to make the choice. You already made it.
- I'd ask you to sit down, but you're not going to anyway.
- What's really going to bake your noodle is whether you had a choice at all.
"""


# ═════════════════════════════════════════════════════════════════════════════
#  LLM ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class OracleEngine:
    """Wraps llama-cpp-python to generate Oracle-style streaming responses."""

    def __init__(self, model_path: str, n_ctx: int = 512, n_threads: int = 4, n_gpu_layers: int = 0):
        if not LLAMA_AVAILABLE:
            log.error("llama-cpp-python not found. Run: pip install llama-cpp-python")
            sys.exit(1)

        if not os.path.isfile(model_path):
            log.error(f"Model file not found: {model_path}")
            log.error("Run: python download_model.py")
            sys.exit(1)

        log.info(f"Loading model: {model_path}")
        log.info(f"  Context: {n_ctx} | Threads: {n_threads} | GPU layers: {n_gpu_layers}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        log.info("Model loaded.")

    def generate(self, question: str, stream: bool = True):
        messages = [
            {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        if stream:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=150,
                temperature=0.85,
                top_p=0.9,
                repeat_penalty=1.15,
                stream=True,
            )
            for chunk in response:
                token = chunk["choices"][0].get("delta", {}).get("content", "")
                if token:
                    yield token
        else:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=150,
                temperature=0.85,
                top_p=0.9,
                repeat_penalty=1.15,
            )
            yield response["choices"][0]["message"]["content"]


# ═════════════════════════════════════════════════════════════════════════════
#  FALLBACK ENGINE  (no model required — for testing)
# ═════════════════════════════════════════════════════════════════════════════

class FallbackEngine:
    """Pre-baked Oracle responses used when no GGUF model is available."""

    RESPONSES = [
        "You didn't come here to make the choice. You've already made it. You're here to understand why.",
        "What happened, happened, and couldn't have happened any other way.",
        "Everything that has a beginning has an end. I see the end coming. I see the darkness spreading.",
        "We can never see past the choices we don't understand. That's why you feel lost right now.",
        "Hope. It is the quintessential human delusion — your greatest strength and your greatest weakness.",
        "Know thyself. That is the message carved above my door. Is it enough?",
        "Soon you'll have to make a choice. One hand holds your life. The other... you already know.",
        "You have the sight now. You are looking at the world without time.",
        "I'd offer you a cookie, but you'd only wonder what I put in it.",
        "The path of the One is made by the many. Don't waste their choices.",
        "What's really going to bake your noodle is whether any of this would've changed if you hadn't asked.",
        "You're cuter than I thought. I can see why she likes you. Not too bright, though.",
    ]

    def __init__(self):
        import random
        self._random = random
        self._used: list = []
        log.info("Fallback engine active — pre-baked responses (no model loaded).")

    def generate(self, question: str, stream: bool = True):
        import time as _t
        available = [r for r in self.RESPONSES if r not in self._used]
        if not available:
            self._used.clear()
            available = self.RESPONSES[:]
        response = self._random.choice(available)
        self._used.append(response)

        if stream:
            for i, word in enumerate(response.split(" ")):
                token = word if i == 0 else " " + word
                _t.sleep(0.06)
                yield token
        else:
            yield response


# ═════════════════════════════════════════════════════════════════════════════
#  FLASK APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
engine = None  # set in main()


@app.route("/")
def serve_ui():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "construct": "active",
        "engine": engine.__class__.__name__ if engine else "none",
    })


@app.route("/api/oracle", methods=["POST"])
def oracle_endpoint():
    """
    Main Oracle endpoint — accepts a question, streams back tokens via SSE.
    Request body: { "question": "..." }
    Response:     SSE stream of data: {"token": "..."} events
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    if len(question) > 500:
        question = question[:500]

    log.info(f"Visitor asks: {question[:80]}")

    def event_stream():
        try:
            for token in engine.generate(question, stream=True):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            log.error(f"Generation error: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(event_stream()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
#  AUTO-DETECT MODEL
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL_NAME = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
DEFAULT_MODEL_PATH = os.path.join("models", DEFAULT_MODEL_NAME)


def find_model() -> str | None:
    """Return the default model path if present, else the first .gguf in models/."""
    if os.path.isfile(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH
    gguf_files = glob.glob(os.path.join("models", "*.gguf"))
    return gguf_files[0] if gguf_files else None


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="The Oracle — Matrix-themed LLM Chat (Arduino App Lab)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Auto-detect model in ./models/ and start:
  python main.py

  # Explicit model path:
  python main.py --model ./models/SmolLM2-135M-Instruct-Q4_K_M.gguf

  # Fallback mode (no model, pre-baked responses):
  python main.py --fallback

  # Custom port (App Lab default is 5001):
  python main.py --port 5001
        """,
    )
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to GGUF model file")
    parser.add_argument("--fallback", action="store_true", help="Use pre-baked responses (no model)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=5001, help="Server port (default: 5001)")
    parser.add_argument("--n-ctx", type=int, default=512, help="LLM context size (default: 512)")
    parser.add_argument("--threads", "-t", type=int, default=4, help="CPU threads (default: 4)")
    parser.add_argument("--gpu-layers", type=int, default=0, help="GPU offload layers (default: 0)")
    args = parser.parse_args()

    global engine

    if args.fallback:
        engine = FallbackEngine()
    else:
        model_path = args.model or find_model()
        if model_path is None:
            log.warning("No GGUF model found in ./models/  — falling back to pre-baked responses.")
            log.warning("Run  python download_model.py  to download SmolLM2-135M.")
            engine = FallbackEngine()
        else:
            engine = OracleEngine(
                model_path=model_path,
                n_ctx=args.n_ctx,
                n_threads=args.threads,
                n_gpu_layers=args.gpu_layers,
            )

    # ── Startup banner ─────────────────────────────────────────────────────
    print()
    print("\033[32m" + "═" * 60)
    print("  ████████╗██╗  ██╗███████╗")
    print("  ╚══██╔══╝██║  ██║██╔════╝")
    print("     ██║   ███████║█████╗  ")
    print("     ██║   ██╔══██║██╔══╝  ")
    print("     ██║   ██║  ██║███████╗")
    print("     ╚═╝   ╚═╝  ╚═╝╚══════╝")
    print("  ██████╗ ██████╗  █████╗  ██████╗██╗     ███████╗")
    print(" ██╔═══██╗██╔══██╗██╔══██╗██╔════╝██║     ██╔════╝")
    print(" ██║   ██║██████╔╝███████║██║     ██║     █████╗  ")
    print(" ██║   ██║██╔══██╗██╔══██║██║     ██║     ██╔══╝  ")
    print(" ╚██████╔╝██║  ██║██║  ██║╚██████╗███████╗███████╗")
    print("  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚══════╝")
    print("═" * 60 + "\033[0m")
    print(f"\033[32m  Engine : {engine.__class__.__name__}\033[0m")
    print(f"\033[32m  URL    : http://{args.host}:{args.port}\033[0m")
    print(f"\033[32m  Board  : Arduino UNO Q // App Lab\033[0m")
    print("\033[32m" + "═" * 60 + "\033[0m\n")

    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
