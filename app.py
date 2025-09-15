# app.py
from flask import Flask, render_template, request, jsonify, session
import os, threading, re, ast, operator
import requests

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# -------------------------------
# Hosted LLM (Hugging Face Inference API)
# -------------------------------
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN = os.environ.get("HF_API_TOKEN")  # set this in Render → Environment
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
        # Minimal sanity check: require token to be present
        if not HF_TOKEN:
            print("WARNING: HF_API_TOKEN is not set. Set it in your environment.")
        model_ready = True

# -------------------------------
# Math guardrail (robust)
# - Finds the longest arithmetic substring inside any sentence
# - Normalizes word operators and unicode symbols
# - Safely evaluates + - * / ^ and parentheses
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
    ("–", "-"),   # en-dash
    ("—", "-"),   # em-dash
    ("÷", "/"),
]

def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):  # Py3.8+
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Invalid constant")
    if isinstance(node, ast.Num):       # legacy (<3.8)
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
    """Convert any sentence to a condensed arithmetic expression candidate."""
    s = text.lower()

    # If there's an equals sign, keep only the left side (ignore asserted result)
    if "=" in s:
        s = s.split("=", 1)[0]

    # Replace common word operators and unicode symbols
    for word, op in _WORD_TO_OP:
        s = s.replace(word, op)

    # Remove quotes/letters/commas etc., keep mathy chars + spaces
    s = "".join(ch for ch in s if ch in _MATH_CHARS)

    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def longest_math_substring(text: str) -> str:
    """
    From a normalized string (only math-ish chars), pick the longest substring
    that contains at least one operator and parses our allowed pattern.
    """
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
    """Full pipeline: sentence -> normalized -> longest arithmetic substring."""
    normalized = normalize_sentence_to_math(user_text)
    expr = longest_math_substring(normalized)
    return expr

# -------------------------------
# Chat helpers
# -------------------------------
def to_chat_messages(history, user_text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        role = "assistant" if m["role"] == "bot" else "user"
        msgs.append({"role": role, "content": m["text"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs

def build_inputs(history, user_text):
    """
    Build a simple prompt for instruction-tuned models.
    (Keeps your history and system prompt.)
    """
    lines = [SYSTEM_PROMPT, ""]
    for m in history:
        role = "Assistant" if m["role"] == "bot" else "User"
        lines.append(f"{role}: {m['text']}")
    lines.append(f"User: {user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)

def generate_reply(prompt_text):
    """
    Call Hugging Face Inference API. Keep params modest for speed/cost.
    """
    if not HF_TOKEN:
        return "Server error: HF_API_TOKEN is not set."
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # Common HF response shape: list with {"generated_text": "..."}
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"].strip()
            # Trim at turn markers if present
            for marker in ["\nUser:", "\nAssistant:"]:
                if marker in text:
                    text = text.split(marker)[0].strip()
            return text or "..."
        # Other shapes: return a stringified version
        return str(data)
    except Exception as e:
        print("HF API error:", e)
        return "Sorry, I had trouble contacting the model API."

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def index():
    session.setdefault("messages", [])
    return render_template("index.html")

@app.post("/chat")
def chat():
    user_text = (request.json or {}).get("message", "").strip()
    if not user_text:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    # --- Math guardrail FIRST ---
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

    # --- Otherwise, call hosted model ---
    msgs = session.get("messages", [])
    msgs.append({"role": "user", "text": user_text})
    if len(msgs) > 10:
        msgs = msgs[-10:]

    ensure_model_loaded()
    prompt_text = build_inputs(msgs, user_text)
    reply_text = generate_reply(prompt_text)

    msgs.append({"role": "bot", "text": reply_text})
    session["messages"] = msgs
    return jsonify({"ok": True, "reply": reply_text})

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
