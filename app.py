# app.py
from flask import Flask, render_template_string, request, jsonify, session
import os, threading, re, ast, operator, time, traceback
import requests

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# Hosted LLM (Hugging Face Inference API)
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN = os.environ.get("HF_API_TOKEN")
HF_URL   = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# --- diagnostics (safe to expose; does NOT print your token) ---
@app.get("/diag")
def diag():
    return jsonify({
        "model_env": os.environ.get("HF_MODEL"),
        "model_in_app": HF_MODEL,
        "has_token": bool(HF_TOKEN),
        "url": f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    })

# --- extra diagnostics (temporary; safe: does NOT expose your token) ---
@app.get("/hf-whoami")
def hf_whoami():
    if not HF_TOKEN:
        return jsonify({"ok": False, "msg": "HF_API_TOKEN missing"}), 500
    r = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        timeout=20,
    )
    try:
        body = r.json()
    except Exception:
        body = r.text
    return jsonify({"status_code": r.status_code, "body": body})

@app.get("/hf-modelmeta")
def hf_modelmeta():
    model = os.environ.get("HF_MODEL", "gpt2")
    r = requests.get(f"https://huggingface.co/api/models/{model}", timeout=20)
    try:
        body = r.json()
    except Exception:
        body = r.text
    return jsonify({"status_code": r.status_code, "length": len(str(body))})


@app.get("/hf-test")
def hf_test():
    if not HF_TOKEN:
        return jsonify({"ok": False, "msg": "HF_API_TOKEN missing"}), 500
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type":"application/json"}
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    r = requests.post(url, headers=headers, json={"inputs":"ping"}, timeout=30)
    try:
        body = r.json()
    except Exception:
        body = r.text[:500]
    return jsonify({"status_code": r.status_code, "body": body})


model_lock = threading.Lock()
model_ready = False

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant for a beginner-friendly Python web app. "
    "Answer clearly, use short paragraphs, and avoid making things up. "
    "If you are unsure, say so briefly."
)

def ensure_model_loaded():
    """Using a hosted API—nothing to load locally; just mark ready."""
    global model_ready
    if model_ready:
        return
    with model_lock:
        if model_ready:
            return
        if not HF_TOKEN:
            print("WARNING: HF_API_TOKEN is not set. Set it in your Render environment.")
        model_ready = True

# -------------------------------
# Math guardrail (robust)
# -------------------------------
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,   # remove if you don't want exponentiation
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_MATH_CHARS = "0123456789+-*/^(). "
_MATH_RE = re.compile(r"^[0-9+\-*/^().\s]+$")

_WORD_TO_OP = [
    ("divided by", "/"),
    ("over", "/"),
    ("times", "*"),
    ("multiplied by", "*"),
    ("x", "*"),
    ("×", "*"),
    ("·", "*"),
    ("plus", "+"),
    ("minus", "-"),
    ("–", "-"),
    ("—", "-"),
    ("÷", "/"),
]

def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Invalid constant")
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_OPS[type(node.op)](left, right)
    raise ValueError("Unsupported expression")

def safe_eval_expr(expr: str):
    if not _MATH_RE.fullmatch(expr):
        raise ValueError("Unsafe characters")
    tree = ast.parse(expr, mode="eval")
    val = _eval_node(tree)
    return int(val) if isinstance(val, float) and val.is_integer() else val

def normalize_sentence_to_math(text: str) -> str:
    s = text.lower()
    if "=" in s:
        s = s.split("=", 1)[0]
    for word, op in _WORD_TO_OP:
        s = s.replace(word, op)
    s = "".join(ch for ch in s if ch in _MATH_CHARS)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def longest_math_substring(text: str) -> str:
    candidates = re.findall(r"[0-9+\-*/^(). ]+", text)
    cleaned = []
    for c in candidates:
        c2 = c.replace(" ", "")
        if not c2:
            continue
        if _MATH_RE.fullmatch(c2) and re.search(r"[+\-*/^]", c2):
            cleaned.append(c2)
    if not cleaned:
        return ""
    cleaned.sort(key=len)
    return cleaned[-1]

def extract_math_expression(user_text: str) -> str:
    normalized = normalize_sentence_to_math(user_text)
    return longest_math_substring(normalized)

# -------------------------------
# Chat helpers
# -------------------------------
def build_prompt(history, user_text):
    lines = [SYSTEM_PROMPT, ""]
    for m in history:
        role = "Assistant" if m["role"] == "bot" else "User"
        lines.append(f"{role}: {m['text']}")
    lines.append(f"User: {user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)

def call_hf_inference(prompt_text, retries=2):
    """Call HF API with basic handling for cold-start and rate limits."""
    if not HF_TOKEN:
        return False, "Server error: HF_API_TOKEN is not set."

    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    for attempt in range(retries + 1):
        try:
            r = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
            # Handle common transient statuses
            if r.status_code in (429, 503):
                # 503 can be "Model is loading" on first hit; wait then retry
                wait = 3 * (attempt + 1)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                text = data[0]["generated_text"].strip()
                for marker in ["\nUser:", "\nAssistant:"]:
                    if marker in text:
                        text = text.split(marker)[0].strip()
                return True, (text or "...")
            # Some models return dicts with error/info
            if isinstance(data, dict) and "error" in data:
                return False, f"Model error: {data['error']}"
            return True, str(data)
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            return False, f"Network error: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def index():
    session.setdefault("messages", [])
    # Inline UI to avoid template errors in production
    return render_template_string("""
    <!doctype html>
    <html><head><meta charset="utf-8" />
      <meta name="viewport" content="width=device-width,initial-scale=1" />
      <title>Chatbot</title>
      <style>
        body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
        #log { white-space: pre-wrap; border: 1px solid #ddd; padding: 1rem; min-height: 8rem; }
        .row { display:flex; gap:.5rem; margin-top:.75rem; }
        input { flex:1; padding:.6rem; }
        button { padding:.6rem .9rem; }
      </style>
    </head>
    <body>
      <h1>Chatbot</h1>
      <p>Try math like <code>(4*3)/(2*2)</code> or ask a question.</p>
      <div id="log"></div>
      <div class="row">
        <input id="msg" placeholder="Say something…" />
        <button id="send">Send</button>
        <button id="reset" title="Clear server history">Reset</button>
      </div>
      <script>
        const log = document.getElementById('log');
        const msg = document.getElementById('msg');
        const send = document.getElementById('send');
        const reset = document.getElementById('reset');
        function append(role, text){ log.textContent += role + ': ' + text + '\\n'; }
        async function post(url, data){
          const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
          return r.json();
        }
        send.onclick = async () => {
          const text = msg.value.trim(); if(!text) return;
          append('You', text); msg.value='';
          const res = await post('/chat', {message: text});
          append(res.ok ? 'Bot' : 'Error', res.ok ? res.reply : (res.error||'Unknown error'));
        };
        reset.onclick = async () => {
          const res = await post('/reset', {});
          if(res.ok){ log.textContent=''; append('System','Conversation reset.'); }
        };
      </script>
    </body></html>
    """)

@app.post("/chat")
def chat():
    try:
        user_text = (request.json or {}).get("message", "").strip()
        if not user_text:
            return jsonify({"ok": False, "error": "Empty message"}), 400

        # Math guardrail first
        expr = extract_math_expression(user_text)
        if expr:
            try:
                result = safe_eval_expr(expr)
                print(f"[math] extracted='{expr}' -> {result}")
                msgs = session.get("messages", [])
                msgs.append({"role": "user", "text": user_text})
                msgs.append({"role": "bot", "text": str(result)})
                session["messages"] = msgs[-10:]
                return jsonify({"ok": True, "reply": str(result)})
            except Exception as e:
                print(f"[math] failed to eval '{expr}': {e}  (falling back to model)")

        # Otherwise call hosted model
        msgs = session.get("messages", [])
        msgs.append({"role": "user", "text": user_text})
        if len(msgs) > 10:
            msgs = msgs[-10:]

        ensure_model_loaded()
        prompt_text = build_prompt(msgs, user_text)
        ok, reply_or_err = call_hf_inference(prompt_text)
        reply_text = reply_or_err if ok else f"(LLM) {reply_or_err}"

        msgs.append({"role": "bot", "text": reply_text})
        session["messages"] = msgs
        return jsonify({"ok": True, "reply": reply_text})
    except Exception as e:
        # Log the server-side traceback so you can see it in Render → Logs → Runtime
        print("SERVER ERROR in /chat:", e)
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Internal server error"}), 500

@app.post("/reset")
def reset():
    session.pop("messages", None)
    return jsonify({"ok": True})

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_ready": model_ready})

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", "5050"))
    print(f"Open your browser to:  http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)





