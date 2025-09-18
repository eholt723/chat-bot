# app.py — stable Flask + Cohere backend (with optional streaming)
import os
import sys
import json
import traceback
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from dotenv import load_dotenv
import cohere

# -------------------------------
# Setup
# -------------------------------
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=COHERE_API_KEY)

# NOTE: your templates folder is capitalized ("Templates")
app = Flask(__name__, template_folder="Templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

SYSTEM_PROMPT = "Be concise, friendly, and helpful. If unsure, say so briefly."
HISTORY_MAX = 6
FAST_TRIGGER = {"hi", "hello", "hey", "yo", "sup", "howdy"}

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def index():
    session.setdefault("messages", [])
    return render_template("index.html")

@app.post("/chat")
def chat():
    """Non-streaming endpoint used by your current UI."""
    try:
        payload = request.get_json(silent=True) or {}
        user_text = (payload.get("message") or "").strip()
        if not user_text:
            return jsonify({"ok": False, "error": "Empty message"}), 400
        if not COHERE_API_KEY:
            return jsonify({"ok": False, "error": "Missing COHERE_API_KEY"}), 500

        # Build short context (fast)
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

        # Cohere call
        res = co.chat(model="command-a-03-2025", messages=messages)
        reply = res.message.content[0].text if res and res.message and res.message.content else "(no reply)"

        # Save short history
        history.append({"role": "user", "text": user_text})
        history.append({"role": "bot",  "text": reply})
        session["messages"] = history[-HISTORY_MAX:]

        return jsonify({"ok": True, "reply": reply})

    except Exception as e:
        print("ERROR in /chat:", repr(e), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return jsonify({"ok": False, "error": f"Server exception: {e.__class__.__name__}"}), 500

@app.post("/chat_stream")
def chat_stream():
    """Streaming endpoint (SSE). Frontend must read the stream."""
    payload = request.get_json(silent=True) or {}
    user_text = (payload.get("message") or "").strip()
    if not user_text:
        return jsonify({"ok": False, "error": "Empty message"}), 400
    if not COHERE_API_KEY:
        return jsonify({"ok": False, "error": "Missing COHERE_API_KEY"}), 500

    # Build message list (same policy as /chat)
    history = session.get("messages", [])
    if len(history) > HISTORY_MAX:
        history = history[-HISTORY_MAX:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        role = "assistant" if m["role"] == "bot" else "user"
        messages.append({"role": role, "content": m["text"]})
    messages.append({"role": "user", "content": user_text})

    def generate():
        try:
            # Stream tokens as they arrive
            for event in co.chat_stream(model="command-a-03-2025", messages=messages):
                if getattr(event, "event_type", "") == "text-generation":
                    yield f"data: {json.dumps({'token': event.text})}\n\n"
            yield "data: [DONE]\n\n"

            # Update session after stream completes (non-critical)
            hist = session.get("messages", [])
            hist.append({"role": "user", "text": user_text})
            # NOTE: the full reply is built on the client; if you want to save it server-side,
            # you can accumulate tokens here as well.
            session["messages"] = hist[-HISTORY_MAX:]

        except Exception as e:
            print("ERROR in /chat_stream:", repr(e), file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.post("/reset")
def reset():
    session.pop("messages", None)
    return jsonify({"ok": True})

@app.get("/health")
def health():
    return jsonify({"status": "ok", "backend": "cohere", "session_len": len(session.get("messages", []))})

# Optional: JSON-only errors for /chat to avoid HTML error pages breaking the UI
@app.errorhandler(404)
def _404(e):
    if request.path == "/chat":
        return jsonify(ok=False, error="Endpoint not found"), 404
    return e

@app.errorhandler(405)
def _405(e):
    if request.path == "/chat":
        return jsonify(ok=False, error="Method not allowed"), 405
    return e

@app.errorhandler(500)
def _500(e):
    if request.path == "/chat":
        return jsonify(ok=False, error="Server error"), 500
    return e

# Local dev (Render runs gunicorn app:app)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    print(f"Open your browser to: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
