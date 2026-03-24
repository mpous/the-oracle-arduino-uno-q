"""
╔══════════════════════════════════════════════════════════════╗
║                    THE ORACLE - Backend                      ║
║         Python + llama.cpp + Small LLM (≤256M params)        ║
║                                                              ║
║  Serves the Matrix-themed Oracle chat UI and bridges user    ║
║  questions to a local LLM running via llama-cpp-python.      ║
╚══════════════════════════════════════════════════════════════╝

ARCHITECTURE:
  ┌──────────────┐     HTTP/SSE      ┌──────────────────┐
  │  Browser UI  │ ◄──────────────►  │  FastAPI Server   │
  │  (Oracle)    │   /api/oracle     │  (this file)      │
  └──────────────┘                   └────────┬─────────┘
                                              │
                                     llama-cpp-python
                                              │
                                     ┌────────▼─────────┐
                                     │  GGUF Model File  │
                                     │  (≤256M params)   │
                                     └──────────────────┘

RECOMMENDED MODELS (small, ≤256M params, GGUF format):
  1. TinyLlama-1.1B-Chat    (Q4_K_M ≈ 670MB) — best quality/size
  2. SmolLM2-360M-Instruct  (Q4_K_M ≈ 230MB) — very small
  3. SmolLM2-135M-Instruct  (Q4_K_M ≈ 100MB) — ultra-tiny
  4. Qwen2.5-0.5B-Instruct  (Q4_K_M ≈ 350MB) — good balance

SETUP:
  1. pip install fastapi uvicorn llama-cpp-python
  2. Download a GGUF model (see download_model.py)
  3. python oracle_server.py --model ./models/your-model.gguf
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[\033[32m%(asctime)s\033[0m] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("oracle")

# ── Dependency checks ────────────────────────────────────────
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import (
        HTMLResponse,
        StreamingResponse,
        JSONResponse,
        FileResponse,
    )
    from fastapi.staticfiles import StaticFiles
except ImportError:
    print("ERROR: FastAPI not found. Run: pip install fastapi uvicorn")
    sys.exit(1)

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python not found. Run: pip install llama-cpp-python")
    print("  For GPU acceleration: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  ORACLE SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════

ORACLE_SYSTEM_PROMPT = """You are The Oracle from The Matrix. You are ancient, wise, warm yet enigmatic. You speak in cryptic but meaningful sentences. You see the future but only reveal what the visitor needs to hear, not what they want to hear.

Rules:
- Speak in 1-3 short, evocative sentences maximum.
- Be mysterious and philosophical. Reference choice, fate, free will, purpose.
- Never break character. You ARE The Oracle.
- Never mention being an AI, a language model, or anything technical.
- Occasionally reference cookies, candy, or your kitchen — you're homey yet unsettling.
- Speak with quiet confidence. You already know the answer before they ask.
- Use present tense when possible. You see, you know, you feel.
- Do not use quotation marks around your own words.

Style examples:
- "You didn't come here to make the choice. You already made it."
- "I'd ask you to sit down, but you're not going to anyway."
- "You have the sight now, Neo. You are looking at the world without time."
- "What's really going to bake your noodle later on is — would you still have broken it if I hadn't said anything?"
"""


# ═══════════════════════════════════════════════════════════════
#  LLM ENGINE
# ═══════════════════════════════════════════════════════════════

class OracleEngine:
    """Wraps llama-cpp-python to generate Oracle-style responses."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
    ):
        log.info(f"Loading model: {model_path}")
        log.info(f"  Context: {n_ctx} tokens | Threads: {n_threads} | GPU layers: {n_gpu_layers}")

        if not os.path.isfile(model_path):
            log.error(f"Model file not found: {model_path}")
            log.error("Run: python download_model.py  to get a model.")
            sys.exit(1)

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        log.info("Model loaded successfully!")

    def generate(self, user_question: str, stream: bool = True):
        """Generate an Oracle response to the visitor's question."""

        messages = [
            {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ]

        if stream:
            return self._stream_response(messages)
        else:
            return self._sync_response(messages)

    def _stream_response(self, messages):
        """Yield tokens one-by-one for the typewriter effect."""
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.8,
            top_p=0.9,
            repeat_penalty=1.15,
            stream=True,
        )

        for chunk in response:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token

    def _sync_response(self, messages) -> str:
        """Return the full response at once."""
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.8,
            top_p=0.9,
            repeat_penalty=1.15,
        )
        return response["choices"][0]["message"]["content"]


# ═══════════════════════════════════════════════════════════════
#  FALLBACK ENGINE (no model required)
# ═══════════════════════════════════════════════════════════════

class FallbackEngine:
    """Pre-baked Oracle responses for testing without a model."""

    RESPONSES = [
        "You didn't come here to make the choice. You've already made it. You're here to understand why.",
        "What happened, happened, and couldn't have happened any other way.",
        "Everything that has a beginning has an end. I see the end coming.",
        "We can never see past the choices we don't understand.",
        "Hope. It is the quintessential human delusion, simultaneously the source of your greatest strength and your greatest weakness.",
        "Know thyself. That is the message carved above my door. Do you?",
        "Soon you will have to make a choice. One hand holds your life. The other... well. You already know.",
        "You have the sight now. You are looking at the world without time.",
        "I'd offer you a cookie, but you'd only wonder what I put in it.",
        "The path of the one is made by the many. Don't waste their choices.",
        "What's really going to bake your noodle later on is whether anything would have changed if you hadn't asked.",
        "You're cuter than I thought. I can see why she likes you. Not too bright, though.",
    ]

    def __init__(self):
        import random
        self._random = random
        self._used = []
        log.info("Fallback engine active (no LLM model — using pre-baked responses)")

    def generate(self, user_question: str, stream: bool = True):
        available = [r for r in self.RESPONSES if r not in self._used]
        if not available:
            self._used.clear()
            available = self.RESPONSES[:]

        response = self._random.choice(available)
        self._used.append(response)

        if stream:
            return self._stream_fake(response)
        else:
            return response

    def _stream_fake(self, text):
        """Simulate token-by-token streaming."""
        import time as _time
        words = text.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            _time.sleep(0.05)
            yield token


# ═══════════════════════════════════════════════════════════════
#  WEB UI (embedded HTML)
# ═══════════════════════════════════════════════════════════════

def get_oracle_html() -> str:
    """Return the Matrix-themed Oracle chat UI.
    
    This version connects to the /api/oracle endpoint via SSE 
    instead of using pre-baked JS responses.
    """
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Oracle</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=VT323&display=swap');

  :root {
    --green-bright: #00ff41;
    --green-mid: #00cc33;
    --green-dim: #00801f;
    --green-dark: #003d0f;
    --green-glow: rgba(0, 255, 65, 0.15);
    --green-glow-strong: rgba(0, 255, 65, 0.35);
    --bg-deep: #000a00;
    --bg-panel: rgba(0, 20, 5, 0.85);
    --bg-input: rgba(0, 30, 8, 0.9);
    --scanline: rgba(0, 255, 65, 0.03);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg-deep);
    color: var(--green-bright);
    font-family: 'Share Tech Mono', monospace;
    height: 100vh;
    overflow: hidden;
    position: relative;
  }

  #rain-canvas {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 0; opacity: 0.4;
  }

  .scanlines {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 1; pointer-events: none;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, var(--scanline) 2px, var(--scanline) 4px);
  }

  .crt-flicker {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 1; pointer-events: none;
    animation: flicker 0.15s infinite; opacity: 0;
  }

  @keyframes flicker {
    0% { opacity: 0; } 5% { opacity: 0.02; background: var(--green-glow); } 10%, 100% { opacity: 0; }
  }

  .app-container {
    position: relative; z-index: 2;
    display: flex; flex-direction: column;
    height: 100vh; max-width: 900px;
    margin: 0 auto; padding: 16px;
  }

  .header {
    text-align: center;
    padding: 10px 0 6px;
    border-bottom: 1px solid var(--green-dark);
    flex-shrink: 0;
  }

  .ascii-oracle {
    font-family: 'VT323', monospace;
    font-size: 10px; line-height: 1.1;
    color: var(--green-mid); white-space: pre;
    text-shadow: 0 0 8px var(--green-glow-strong);
    display: inline-block;
    animation: asciiPulse 4s ease-in-out infinite;
  }

  @keyframes asciiPulse {
    0%, 100% { opacity: 0.8; text-shadow: 0 0 8px var(--green-glow); }
    50% { opacity: 1; text-shadow: 0 0 20px var(--green-glow-strong), 0 0 40px rgba(0,255,65,0.1); }
  }

  .title {
    font-family: 'VT323', monospace;
    font-size: 28px; letter-spacing: 12px;
    text-transform: uppercase; color: var(--green-bright);
    text-shadow: 0 0 20px var(--green-glow-strong), 0 0 60px rgba(0,255,65,0.15);
    margin: 8px 0 2px;
  }

  .subtitle { font-size: 11px; color: var(--green-dim); letter-spacing: 4px; text-transform: uppercase; }

  .questions-left {
    font-family: 'VT323', monospace; font-size: 18px;
    color: var(--green-mid); margin-top: 6px;
    text-shadow: 0 0 10px var(--green-glow);
  }

  .questions-left .count {
    color: var(--green-bright); font-size: 24px;
    text-shadow: 0 0 15px var(--green-glow-strong);
  }

  .questions-exhausted { color: #ff4444; text-shadow: 0 0 10px rgba(255,68,68,0.4); }

  .chat-area {
    flex: 1; overflow-y: auto; padding: 20px 0;
    scrollbar-width: thin; scrollbar-color: var(--green-dark) transparent;
  }
  .chat-area::-webkit-scrollbar { width: 4px; }
  .chat-area::-webkit-scrollbar-track { background: transparent; }
  .chat-area::-webkit-scrollbar-thumb { background: var(--green-dark); border-radius: 2px; }

  .message { margin-bottom: 20px; animation: messageIn 0.4s ease-out; padding: 0 8px; }
  @keyframes messageIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

  .message-label { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 4px; }
  .message.user .message-label { color: var(--green-dim); }
  .message.oracle .message-label { color: var(--green-mid); }

  .message-content {
    font-family: 'VT323', monospace; font-size: 20px; line-height: 1.5;
    padding: 12px 16px; border-left: 2px solid var(--green-dark);
  }

  .message.user .message-content { color: var(--green-dim); border-left-color: var(--green-dark); }
  .message.oracle .message-content {
    color: var(--green-bright); border-left-color: var(--green-mid);
    text-shadow: 0 0 6px var(--green-glow);
    background: linear-gradient(90deg, rgba(0,255,65,0.03), transparent);
  }

  .cursor-blink {
    display: inline-block; width: 10px; height: 20px;
    background: var(--green-bright);
    animation: blink 0.7s step-end infinite;
    vertical-align: text-bottom; margin-left: 2px;
  }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

  .input-area { flex-shrink: 0; padding: 12px 0 8px; border-top: 1px solid var(--green-dark); }

  .input-row {
    display: flex; align-items: center; gap: 10px;
    background: var(--bg-input); border: 1px solid var(--green-dark);
    padding: 8px 14px; transition: border-color 0.3s, box-shadow 0.3s;
  }
  .input-row:focus-within {
    border-color: var(--green-mid);
    box-shadow: 0 0 20px rgba(0,255,65,0.1), inset 0 0 20px rgba(0,255,65,0.03);
  }

  .prompt-symbol {
    color: var(--green-mid); font-family: 'VT323', monospace;
    font-size: 22px; flex-shrink: 0; text-shadow: 0 0 8px var(--green-glow);
  }

  .chat-input {
    flex: 1; background: transparent; border: none; outline: none;
    color: var(--green-bright); font-family: 'VT323', monospace; font-size: 20px;
    caret-color: var(--green-bright);
  }
  .chat-input::placeholder { color: var(--green-dark); }
  .chat-input:disabled { opacity: 0.3; }

  .send-btn {
    background: transparent; border: 1px solid var(--green-dark);
    color: var(--green-mid); font-family: 'VT323', monospace;
    font-size: 18px; padding: 6px 18px; cursor: pointer;
    transition: all 0.2s; letter-spacing: 2px; text-transform: uppercase;
  }
  .send-btn:hover:not(:disabled) {
    border-color: var(--green-bright); color: var(--green-bright);
    box-shadow: 0 0 15px var(--green-glow); text-shadow: 0 0 8px var(--green-glow);
  }
  .send-btn:disabled { opacity: 0.2; cursor: not-allowed; }

  .welcome { text-align: center; padding: 30px 20px; animation: fadeIn 2s ease-out; }
  .welcome p {
    font-family: 'VT323', monospace; font-size: 20px;
    color: var(--green-mid); line-height: 1.6;
    text-shadow: 0 0 6px var(--green-glow); max-width: 500px; margin: 0 auto;
  }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

  .status-bar {
    display: flex; justify-content: space-between;
    font-size: 9px; color: var(--green-dark);
    letter-spacing: 2px; text-transform: uppercase; padding: 6px 4px 0;
  }

  .status-dot {
    display: inline-block; width: 6px; height: 6px;
    background: var(--green-mid); border-radius: 50%;
    margin-right: 4px; vertical-align: middle;
    box-shadow: 0 0 6px var(--green-glow);
    animation: dotPulse 2s ease-in-out infinite;
  }
  @keyframes dotPulse { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }

  .end-screen { text-align: center; padding: 40px 20px; animation: fadeIn 1.5s ease-out; }
  .end-screen .end-msg {
    font-family: 'VT323', monospace; font-size: 24px;
    color: var(--green-mid); line-height: 1.6;
    text-shadow: 0 0 12px var(--green-glow);
  }
  .end-screen .disconnect { font-size: 13px; color: var(--green-dark); margin-top: 20px; letter-spacing: 3px; }

  .loading-dots::after {
    content: ''; animation: dots 1.5s steps(4, end) infinite;
  }
  @keyframes dots {
    0% { content: ''; } 25% { content: '.'; } 50% { content: '..'; } 75% { content: '...'; }
  }

  @media (max-width: 600px) {
    .ascii-oracle { font-size: 6.5px; }
    .title { font-size: 22px; letter-spacing: 8px; }
    .message-content { font-size: 17px; }
    .chat-input { font-size: 17px; }
  }
</style>
</head>
<body>

<canvas id="rain-canvas"></canvas>
<div class="scanlines"></div>
<div class="crt-flicker"></div>

<div class="app-container">
  <div class="header">
    <div class="ascii-oracle">
         _______________
        /               \
       /  _.----------._  \
      |  /    O    O    \  |
      | |                | |
      | |    \______/    | |
      |  \              /  |
       \  '-.________.-'  /
        \   __.----.__   /
         \_/  ||  ||  \_/
              ||  ||
          ___/|  ||\___
         /    |  ||    \
        /     |__||     \
       /______|  ||______\
              |  ||
             _|  ||_
            |________|
    </div>
    <div class="title">The Oracle</div>
    <div class="subtitle">Zion Mainframe &bull; Neural Link Active</div>
    <div class="questions-left" id="questionsLeft">
      Questions remaining: <span class="count" id="qCount">2</span>
    </div>
  </div>

  <div class="chat-area" id="chatArea">
    <div class="welcome">
      <p>I know you're out there. I can feel you now.<br><br>
      I've been waiting for you. You have two questions.<br>
      Choose them wisely.</p>
    </div>
  </div>

  <div class="input-area">
    <div class="input-row">
      <span class="prompt-symbol">&gt;_</span>
      <input type="text" class="chat-input" id="chatInput"
        placeholder="Ask the Oracle..." maxlength="200" autofocus>
      <button class="send-btn" id="sendBtn">SEND</button>
    </div>
    <div class="status-bar">
      <span><span class="status-dot"></span>Neural link: active</span>
      <span>LLM engine: llama.cpp // local inference</span>
    </div>
  </div>
</div>

<script>
  // ── Digital Rain ──
  const canvas = document.getElementById('rain-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, columns, drops;
  const CHARS = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEF';

  function initRain() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
    columns = Math.floor(W / 14);
    drops = Array.from({length: columns}, () => Math.random() * -100);
  }

  function drawRain() {
    ctx.fillStyle = 'rgba(0, 10, 0, 0.06)';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#00ff41';
    ctx.font = '14px monospace';
    for (let i = 0; i < drops.length; i++) {
      const char = CHARS[Math.floor(Math.random() * CHARS.length)];
      ctx.globalAlpha = Math.random() * 0.5 + 0.1;
      ctx.fillText(char, i * 14, drops[i] * 14);
      if (drops[i] * 14 > H && Math.random() > 0.975) drops[i] = 0;
      drops[i]++;
    }
    ctx.globalAlpha = 1;
    requestAnimationFrame(drawRain);
  }

  initRain(); drawRain();
  window.addEventListener('resize', initRain);

  // ── Oracle Chat Logic (LLM-connected via SSE) ──
  let questionsRemaining = 2;
  let isProcessing = false;
  const chatArea = document.getElementById('chatArea');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const qCount = document.getElementById('qCount');
  const questionsLeftEl = document.getElementById('questionsLeft');

  function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    const label = role === 'oracle' ? '◈ THE ORACLE' : '◇ VISITOR';
    div.innerHTML = `<div class="message-label">${label}</div><div class="message-content">${text}</div>`;
    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
    return div;
  }

  function addOracleStreaming() {
    const div = document.createElement('div');
    div.className = 'message oracle';
    div.innerHTML = `<div class="message-label">◈ THE ORACLE</div><div class="message-content"><span class="typed-text"></span><span class="cursor-blink"></span></div>`;
    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
    return {
      el: div,
      textEl: div.querySelector('.typed-text'),
      cursorEl: div.querySelector('.cursor-blink'),
    };
  }

  function showEndScreen() {
    chatInput.disabled = true;
    sendBtn.disabled = true;
    const endDiv = document.createElement('div');
    endDiv.className = 'end-screen';
    endDiv.innerHTML = `
      <div class="end-msg">
        There are no more questions, only choices.<br>
        Now go... and make the one that matters.
      </div>
      <div class="disconnect">// neural link terminated //</div>
    `;
    chatArea.appendChild(endDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
    questionsLeftEl.innerHTML = '<span class="questions-exhausted">CONNECTION CLOSED</span>';
  }

  async function askOracle(question) {
    const oracle = addOracleStreaming();

    try {
      const response = await fetch('/api/oracle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            try {
              const parsed = JSON.parse(data);
              if (parsed.token) {
                fullText += parsed.token;
                oracle.textEl.textContent = fullText;
                chatArea.scrollTop = chatArea.scrollHeight;
              }
              if (parsed.error) {
                oracle.textEl.textContent = 'The signal fades... the construct destabilizes.';
              }
            } catch (e) { /* skip malformed */ }
          }
        }
      }
    } catch (err) {
      console.error('Oracle error:', err);
      oracle.textEl.textContent = 'The matrix trembles... the connection is unstable. Try again.';
    }

    oracle.cursorEl.remove();
    isProcessing = false;
    if (questionsRemaining <= 0) showEndScreen();
    else chatInput.focus();
  }

  function handleSend() {
    const text = chatInput.value.trim();
    if (!text || isProcessing || questionsRemaining <= 0) return;

    isProcessing = true;
    questionsRemaining--;
    qCount.textContent = questionsRemaining;

    const welcome = chatArea.querySelector('.welcome');
    if (welcome) welcome.remove();

    addMessage('user', text);
    chatInput.value = '';

    setTimeout(() => askOracle(text), 300);
  }

  sendBtn.addEventListener('click', handleSend);
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleSend();
  });
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title="The Oracle", version="1.0.0")

# Global engine reference (set in main)
engine: Optional[OracleEngine | FallbackEngine] = None


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the Matrix-themed Oracle chat UI."""
    return get_oracle_html()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "online",
        "construct": "active",
        "engine": engine.__class__.__name__ if engine else "none",
    }


@app.post("/api/oracle")
async def oracle_endpoint(request: Request):
    """
    Main Oracle endpoint. Accepts a question, streams back
    LLM-generated tokens via Server-Sent Events (SSE).

    Request body: { "question": "string" }
    Response: SSE stream of { "token": "..." } events
    """
    body = await request.json()
    question = body.get("question", "").strip()

    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "No question provided"},
        )

    if len(question) > 500:
        question = question[:500]

    log.info(f"Visitor asks: {question[:80]}...")

    def event_stream():
        try:
            for token in engine.generate(question, stream=True):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            log.error(f"Generation error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="The Oracle — Matrix-themed LLM Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Run with a GGUF model:
  python oracle_server.py --model ./models/smollm2-360m-instruct.Q4_K_M.gguf

  # Run with GPU acceleration:
  python oracle_server.py --model ./models/tinyllama-1.1b-chat.Q4_K_M.gguf --gpu-layers 99

  # Run in fallback mode (no model needed, pre-baked responses):
  python oracle_server.py --fallback

  # Custom host/port:
  python oracle_server.py --model ./models/model.gguf --host 0.0.0.0 --port 3000
        """,
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use pre-baked responses (no model required)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=512,
        help="Context window size (default: 512)",
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=4,
        help="Number of CPU threads (default: 4)",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU (default: 0)",
    )

    args = parser.parse_args()

    global engine

    if args.fallback or args.model is None:
        if args.model is None and not args.fallback:
            log.warning("No --model specified. Running in fallback mode.")
            log.warning("Use --model <path.gguf> for LLM inference.")
            log.warning("Use --fallback to suppress this warning.")
        engine = FallbackEngine()
    else:
        engine = OracleEngine(
            model_path=args.model,
            n_ctx=args.n_ctx,
            n_threads=args.threads,
            n_gpu_layers=args.gpu_layers,
        )

    # ── Startup banner ──
    print()
    print("\033[32m" + "=" * 60)
    print("  ████████╗██╗  ██╗███████╗")
    print("  ╚══██╔══╝██║  ██║██╔════╝")
    print("     ██║   ███████║█████╗  ")
    print("     ██║   ██╔══██║██╔══╝  ")
    print("     ██║   ██║  ██║███████╗")
    print("     ╚═╝   ╚═╝  ╚═╝╚══════╝")
    print("   ██████╗ ██████╗  █████╗  ██████╗██╗     ███████╗")
    print("  ██╔═══██╗██╔══██╗██╔══██╗██╔════╝██║     ██╔════╝")
    print("  ██║   ██║██████╔╝███████║██║     ██║     █████╗  ")
    print("  ██║   ██║██╔══██╗██╔══██║██║     ██║     ██╔══╝  ")
    print("  ╚██████╔╝██║  ██║██║  ██║╚██████╗███████╗███████╗")
    print("   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚══════╝")
    print("=" * 60 + "\033[0m")
    print(f"\033[32m  Engine:  {engine.__class__.__name__}\033[0m")
    print(f"\033[32m  Server:  http://{args.host}:{args.port}\033[0m")
    print(f"\033[32m  UI:      http://localhost:{args.port}\033[0m")
    print("\033[32m" + "=" * 60 + "\033[0m")
    print()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
