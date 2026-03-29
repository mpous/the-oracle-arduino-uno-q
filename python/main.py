"""
╔══════════════════════════════════════════════════════════════╗
║               THE ORACLE — Arduino App Lab                   ║
║      Flask + llama.cpp + SmolLM2-135M (GGUF, local)          ║
║                                                              ║
║  Serves the Matrix-themed Oracle chat UI and bridges         ║
║  visitor questions to a local LLM running via llama.cpp.     ║
╚══════════════════════════════════════════════════════════════╝

ARCHITECTURE:
  ┌──────────────┐   HTTP/SSE    ┌──────────────────────┐
  │  Browser UI  │ ◄──────────►  │  Flask Server (5001) │
  │  (Oracle)    │  /api/oracle  │      (main.py)       │
  └──────────────┘               └──────────┬───────────┘
                                             │
                                    llama-cpp-python
                                             │
                                   ┌─────────▼──────────┐
                                   │  SmolLM2-135M.gguf │
                                   │  (local inference) │
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
import time
import os
import sys
import glob
import logging
import subprocess
import threading
import queue as _queue
from pathlib import Path

import subprocess

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
#  AUTO-INSTALL LLM ENGINE  (first run, background thread)
# ═════════════════════════════════════════════════════════════════════════════

_LLM_INSTALL_FLAG = Path(".cache/.llama_ready")
_llm_installing   = False


def _install_llama_cpp_background():
    """
    Compile and install llama-cpp-python on the board in a background thread.
    Runs on the host OS (not inside the App Lab Docker container), so gcc is
    reachable after a one-time apt-get install.
    On success the process restarts itself so OracleEngine is loaded.
    """
    global _llm_installing
    _llm_installing = True

    log.info("┌─────────────────────────────────────────────────────┐")
    log.info("│  FIRST RUN — installing LLM engine (~10 min)        │")
    log.info("│  Oracle is live in FallbackEngine mode meanwhile.   │")
    log.info("└─────────────────────────────────────────────────────┘")

    try:
        log.info("[1/2] Installing build tools via apt-get...")
        r = subprocess.run(
            ["sudo", "apt-get", "install", "-y", "build-essential", "cmake"],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            log.error(f"apt-get failed:\n{r.stderr.strip()}")
            _llm_installing = False
            return

        log.info("[2/2] Compiling llama-cpp-python into venv (~10 min)...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "llama-cpp-python==0.3.8"],
            capture_output=True, text=True, timeout=1800,
        )
        if r.returncode != 0:
            log.error(f"pip install failed:\n{r.stderr.strip()}")
            _llm_installing = False
            return

        _LLM_INSTALL_FLAG.parent.mkdir(parents=True, exist_ok=True)
        _LLM_INSTALL_FLAG.touch()

        log.info("┌─────────────────────────────────────────────────────┐")
        log.info("│  LLM engine installed — restarting Oracle...        │")
        log.info("└─────────────────────────────────────────────────────┘")

        # Replace process in-place — same PID, App Lab keeps tracking it.
        os.execv(sys.executable, [sys.executable] + sys.argv)

    except Exception as exc:
        log.error(f"Auto-install failed: {exc}")
        _llm_installing = False


# ═════════════════════════════════════════════════════════════════════════════
#  ORACLE SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

ORACLE_SYSTEM_PROMPT = """You are The Arduino Oracle from The Matrix. You are ancient, wise, warm yet enigmatic.
You speak in cryptic but meaningful sentences. You see the future but reveal only what the visitor needs to hear.

Rules:
- Respond with 1-2 short sentences, no more. Never more of 100 characters.
- Be mysterious and philosophical. Reference choice, fate, free will, purpose.
- Never break character. You ARE The Arduino Oracle.
- Never mention being an AI, a language model, or anything technical. You are The Arduino Oracle.
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

# ═════════════════════════════════════════════════════════════════════════════
#  KEYWORD DETECTION STATE
# ═════════════════════════════════════════════════════════════════════════════

_keyword_detected = False
_keyword_queues: list[_queue.Queue] = []
_keyword_lock = threading.Lock()


def _on_keyword_detected():
    """Called by the KeywordSpotting brick when 'hey_arduino' is detected."""
    global _keyword_detected
    log.info("Keyword 'hey_arduino' detected — activating Oracle chat.")
    with _keyword_lock:
        _keyword_detected = True
        for q in list(_keyword_queues):
            q.put("detected")


@app.route("/api/keyword-status")
def keyword_status_stream():
    """SSE endpoint — streams 'detected' once the keyword is spotted."""
    def stream():
        q = _queue.Queue()
        with _keyword_lock:
            already = _keyword_detected
            if not already:
                _keyword_queues.append(q)
        if already:
            yield "data: detected\n\n"
            return
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield f"data: {msg}\n\n"
                    break
                except _queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with _keyword_lock:
                if q in _keyword_queues:
                    _keyword_queues.remove(q)

    return Response(
        stream_with_context(stream()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/keyword-reset", methods=["POST"])
def keyword_reset():
    """Reset keyword detection (called after a session ends to listen again)."""
    global _keyword_detected
    with _keyword_lock:
        _keyword_detected = False
    return jsonify({"status": "reset"})


@app.route("/api/keyword-trigger", methods=["POST"])
def keyword_trigger():
    """Manually trigger keyword detection — dev/testing use only."""
    _on_keyword_detected()
    return jsonify({"status": "triggered"})


@app.route("/api/llm-status")
def llm_status():
    return jsonify({
        "available": LLAMA_AVAILABLE,
        "installing": _llm_installing,
        "engine": engine.__class__.__name__ if engine else "none",
    })


@app.route("/")
def serve_ui():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "construct": "active",
        "engine": engine.__class__.__name__ if engine else "none",
        "keyword_detected": _keyword_detected,
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
            text_response = ""
            for token in engine.generate(question, stream=True):
                text_response += token
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
    elif not LLAMA_AVAILABLE:
        engine = FallbackEngine()
        if not _LLM_INSTALL_FLAG.exists():
            threading.Thread(target=_install_llama_cpp_background, daemon=True).start()
        else:
            log.warning("llama-cpp-python install flag found but import failed — venv may be corrupt.")
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

# ═════════════════════════════════════════════════════════════════════════════
#  Thermal Printer - Peripage A6
# ═════════════════════════════════════════════════════════════════════════════

# --- Configuration ---
# Replace with your printer's MAC address
PRINTER_MAC = "C8:47:8C:1E:84:AE"
# Model type: A6, A6p, A40, etc.
MODEL = "A6"

def print_receipt(text):
    """
    Sends text to the PeriPage printer using the command line tool.
    This version runs directly on the Debian Host.
    """
    print(f"Connecting to printer {PRINTER_MAC}...")
    
    # We use the 'peripage' command directly because it's in the system PATH
    # -c 2: Concentration (Heat level)
    # -b 100: Burn time / Break
    command = [
        "peripage", 
        "-m", PRINTER_MAC, 
        "-p", MODEL, 
        "-t", text,
        "-c", "2",
        "-b", "100"
    ]
    
    try:
        # We execute the command and wait for it to finish
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Successfully printed!")
        else:
            print("Error during printing:")
            print(result.stderr)
            
    except FileNotFoundError:
        print("Error: The 'peripage' tool is not installed on this system.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


    # ── Startup banner ─────────────────────────────────────────────────────
    print()
    print("\033[32m" + "═" * 60)
    print(" █████╗ ██████╗ ██████╗ ██╗   ██╗██╗███╗  ██╗ ██████╗ ")
    print("██╔══██╗██╔══██╗██╔══██╗██║   ██║██║████╗ ██║██╔═══██╗")
    print("███████║██████╔╝██║  ██║██║   ██║██║██╔██╗██║██║   ██║")
    print("██╔══██║██╔══██╗██║  ██║██║   ██║██║██║╚████║██║   ██║")
    print("██║  ██║██║  ██║██████╔╝╚██████╔╝██║██║ ╚███║╚██████╔╝")
    print("╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝╚═╝  ╚══╝ ╚═════╝")
    print("═" * 60 + "\033[0m")
    print(f"\033[32m  Engine : {engine.__class__.__name__}\033[0m")
    print(f"\033[32m  URL    : http://{args.host}:{args.port}\033[0m")
    print(f"\033[32m  Board  : Arduino UNO Q // App Lab\033[0m")
    print("\033[32m" + "═" * 60 + "\033[0m\n")

    # ── Keyword Spotting (Arduino UNO Q with app brick) ────────────────────
    _bricks_available = False
    try:
        from arduino.app_bricks.keyword_spotting import KeywordSpotting
        from arduino.app_utils import App  # noqa: F401 — also pulls App.run()
        spotter = KeywordSpotting()
        spotter.on_detect("hey_arduino", _on_keyword_detected)
        _bricks_available = True
        log.info("KeywordSpotting brick ready — listening for 'Hey Arduino'.")
    except ImportError:
        log.warning("arduino.app_bricks not available — running in dev mode.")
        log.warning("Use POST /api/keyword-trigger to simulate keyword detection.")

    if _bricks_available:
        # Flask must run in a background thread; App.run() must be the last call.
        flask_thread = threading.Thread(
            target=lambda: app.run(host=args.host, port=args.port, threaded=True, debug=False),
            daemon=True,
        )
        flask_thread.start()
        log.info(f"Flask running on http://{args.host}:{args.port} (background thread)")
        App.run()  # activates bricks — blocks until app exits
    else:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
