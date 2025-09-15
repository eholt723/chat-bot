from flask import Flask, render_template, request, jsonify, session
import os, threading, re, ast, operator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# -------------------------------
# Model (lazy-loaded for fast startup)
# -------------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = None
model = None
model_lock = threading.Lock()
model_ready = False

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant for a beginner-friendly Python web app. "
    "Answer clearly, use short paragraphs, and avoid making things up. "
    "If you are unsure, say so briefly."
)

def ensure_model_loaded():
    """Load the tokenizer + model once, on demand."""
    global tokenizer, model, model_ready
    if model_ready:
        return
    with model_lock:
        if model_ready:
            return
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,   # CPU-friendly
            device_map="auto"      # CPU unless you have a GPU
        )
        mdl.eval()
        tokenizer = tok
        model = mdl
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
    # Split on non-math chars just in case (shouldn't exist after normalize)
    candidates = re.findall(r"[0-9+\-*/^(). ]+", text)
    # Clean and score by length
    cleaned = []
    for c in candidates:
        c2 = c.replace(" ", "")
        if not c2:
            continue
        if _MATH_RE.fullmatch(c2) and re.search(r"[+\-*/^]", c2):
            cleaned.append(c2)
    if not cleaned:
        return ""
    # Pick the longest; if tie, last occurrence (usually the main expression)
    cleaned.sort(key=len)
    return cleaned[-1]

def extract_math_expression(user_text: str) -> str:
    """
    Full pipeline: sentence -> normalized -> longest arithmetic substring.
    Returns "" if nothing reasonable is found.
    """
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
    msgs = to_chat_messages(history, user_text)
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    else:
        text = SYSTEM_PROMPT + "\n\n"
        for m in msgs:
            if m["role"] == "system":
                continue
            prefix = "User:" if m["role"] == "user" else "Assistant:"
            text += f"{prefix} {m['content']}\n"
        text += "Assistant:"
    return tokenizer(text, return_tensors="pt")

def generate_reply(inputs):
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.3,     # lower randomness
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    reply_ids = output[0][prompt_len:]
    reply_text = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
    for marker in ["<|user|>", "<|assistant|>", "User:", "Assistant:"]:
        if marker in reply_text:
            reply_text = reply_text.split(marker)[0].strip()
    return reply_text

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
            # Debug print so you can confirm what's being evaluated
            print(f"[math] extracted='{expr}' -> {result}")
            msgs = session.get("messages", [])
            msgs.append({"role": "user", "text": user_text})
            msgs.append({"role": "bot", "text": str(result)})
            session["messages"] = msgs[-10:]
            return jsonify({"ok": True, "reply": str(result)})
        except Exception as e:
            print(f"[math] failed to eval '{expr}': {e}  (falling back to model)")

    # --- Otherwise, fall back to model ---
    msgs = session.get("messages", [])
    msgs.append({"role": "user", "text": user_text})
    if len(msgs) > 10:
        msgs = msgs[-10:]

    ensure_model_loaded()
    inputs = build_inputs(msgs, user_text)
    reply_text = generate_reply(inputs)

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
    app.run(host="127.0.0.1", port=PORT, debug=True, use_reloader=False)
