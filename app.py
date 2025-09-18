# app.py
from flask import Flask, render_template, request, jsonify, session
import os, re, ast, operator
from dotenv import load_dotenv
import cohere

# -------------------------------
# Setup
# -------------------------------
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=COHERE_API_KEY)

# NOTE: your templates folder is capitalized ("Templates"), so point Flask at it explicitly.
app = Flask(__name__, template_folder="Templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant for a beginner-friendly Python web app. "
    "Answer clearly, use short paragraphs, and avoid making things up. "
    "If you are unsure, say so briefly."
)

# -------------------------------
# Math guardrail
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
    expr = longest_math_substring(normalized)
    return expr

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def index():
    session.setdefault("messages", [])
    return render_template("index.html")

@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    user_text = (payload.get("message") or "").strip()
    if not user_text:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    # 1) Math guardrail first
    expr = extract_math_expression(user_text)
    if expr:
        try:
            result = safe_eval_expr(expr)
            msgs = session.get("messages", [])
            msgs.append({"role": "user", "text": user_text})
            msgs.append({"role": "bot",  "text": str(result)})
            session["messages"] = msgs[-10:]
            return jsonify({"ok": True, "reply": str(result)})
        except Exception as e:
            # fall through to model
            print(f"[math] failed to eval '{expr}': {e}")

    # 2) Cohere API fallback (no local torch/models)
    if not COHERE_API_KEY:
        return jsonify({"ok": False, "error": "Missing COHERE_API_KEY"}), 500

    history = session.get("messages", [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        role = "assistant" if m["role"] == "bot" else "user"
        messages.append({"role": role, "content": m["text"]})
    messages.append({"role": "user", "content": user_text})

    try:
        res = co.chat(model="command-a-03-2025", messages=messages)
        reply = res.message.content[0].text if res and res.message and res.message.content else "(no reply)"
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    history.append({"role": "user", "text": user_text})
    history.append({"role": "bot",  "text": reply})
    session["messages"] = history[-10:]

    return jsonify({"ok": True, "reply": reply})

@app.post("/reset")
def reset():
    session.pop("messages", None)
    return jsonify({"ok": True})

@app.get("/health")
def health():
    return jsonify({"status": "ok", "backend": "cohere", "session_len": len(session.get("messages", []))})

# Local dev runner (Render uses Gunicorn with app:app)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    print(f"Open your browser to: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
