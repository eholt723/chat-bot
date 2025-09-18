# app.py — minimal, faster, Cohere-only
from flask import Flask, render_template, request, jsonify, session
import os
from dotenv import load_dotenv
import cohere

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=COHERE_API_KEY)

# NOTE: your folder is capitalized in your repo, so point Flask to it explicitly.
app = Flask(__name__, template_folder="Templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

SYSTEM_PROMPT = "Be concise, friendly, and helpful. If unsure, say so briefly."
HISTORY_MAX = 6  # keep context small for speed
FAST_TRIGGER = {"hi", "hello", "hey", "yo", "sup", "howdy"}

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
    if not COHERE_API_KEY:
        return jsonify({"ok": False, "error": "Missing COHERE_API_KEY"}), 500

    # Build minimal message list (fast path for greetings)
    history = session.get("messages", [])
    if len(history) > HISTORY_MAX:
        history = history[-HISTORY_MAX:]

    if user_text.lower() in FAST_TRIGGER:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
    else:
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

    # Save short history
    history.append({"role": "user", "text": user_text})
    history.append({"role": "bot",  "text": reply})
    session["messages"] = history[-HISTORY_MAX:]

    return jsonify({"ok": True, "reply": reply})

@app.post("/reset")
def reset():
    session.pop("messages", None)
    return jsonify({"ok": True})

@app.get("/health")
def health():
    return jsonify({"status": "ok", "backend": "cohere", "session_len": len(session.get("messages", []))})

# Local dev (Render runs gunicorn app:app)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    print(f"Open your browser to: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
