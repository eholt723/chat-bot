from flask import Flask, render_template, request, jsonify, session
import os
import cohere

app = Flask(__name__, template_folder="Templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# Cohere setup
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
MODEL_NAME = os.environ.get("COHERE_MODEL", "command-r")
co = cohere.ClientV2(api_key=COHERE_API_KEY)

SYSTEM_PROMPT = (
    "You are a concise, friendly assistant for a beginner-friendly Python web app. "
    "Answer clearly and avoid making things up. "
    "If unsure, say so briefly."
)

@app.get("/")
def index():
    session.setdefault("messages", [])
    return render_template("index.html")

@app.post("/chat")
def chat():
    user_text = (request.json or {}).get("message", "").strip()
    if not user_text:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    # Build history
    msgs = session.get("messages", [])
    msgs.append({"role": "user", "text": user_text})
    if len(msgs) > 6:
        msgs = msgs[-6:]

    try:
        resp = co.chat(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] +
                     [{"role": "user" if m["role"]=="user" else "assistant", "content": m["text"]}
                      for m in msgs],
            temperature=0.5,
        )
        reply_text = resp.text
    except Exception as e:
        reply_text = f"(Cohere error: {e})"

    msgs.append({"role": "bot", "text": reply_text})
    session["messages"] = msgs
    return jsonify({"ok": True, "reply": reply_text})

@app.post("/reset")
def reset():
    session.pop("messages", None)
    return jsonify({"ok": True})

@app.get("/health")
def health():
    return jsonify({"status": "ok"})
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
