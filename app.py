# app.py — robust SSE streaming + fallback
import os, sys, json, traceback
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from dotenv import load_dotenv
import cohere

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=COHERE_API_KEY)

app = Flask(__name__, template_folder="Templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

SYSTEM_PROMPT = "Be concise, friendly, and helpful. If unsure, say so briefly."
HISTORY_MAX    = 6
FAST_TRIGGER   = {"hi","hello","hey","yo","sup","howdy"}

@app.get("/")
def index():
    session.setdefault("messages", [])
    return render_template("index.html")

def _build_messages(user_text, history):
    if len(history) > HISTORY_MAX:
        history = history[-HISTORY_MAX:]
    if user_text.lower() in FAST_TRIGGER:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        role = "assistant" if m["role"] == "bot" else "user"
        msgs.append({"role": role, "content": m["text"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs

@app.post("/chat")
def chat():
    try:
        payload   = request.get_json(silent=True) or {}
        user_text = (payload.get("message") or "").strip()
        if not user_text:
            return jsonify({"ok": False, "error": "Empty message"}), 400
        if not COHERE_API_KEY:
            return jsonify({"ok": False, "error": "Missing COHERE_API_KEY"}), 500

        history  = session.get("messages", [])
        messages = _build_messages(user_text, history)

        res = co.chat(model="command-a-03-2025", messages=messages)
        reply = res.message.content[0].text if res and res.message and res.message.content else "(no reply)"

        history.append({"role": "user", "text": user_text})
        history.append({"role": "bot",  "text": reply})
        session["messages"] = history[-HISTORY_MAX:]
        return jsonify({"ok": True, "reply": reply})
    except Exception as e:
        print("ERROR /chat:", e, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return jsonify({"ok": False, "error": f"{e.__class__.__name__}"}), 500

@app.post("/chat_stream")
def chat_stream():
    payload   = request.get_json(silent=True) or {}
    user_text = (payload.get("message") or "").strip()
    if not user_text:
        return jsonify({"ok": False, "error": "Empty message"}), 400
    if not COHERE_API_KEY:
        return jsonify({"ok": False, "error": "Missing COHERE_API_KEY"}), 500

    history  = session.get("messages", [])
    messages = _build_messages(user_text, history)

    def generate():
        full = []
        try:
            # send an initial ping so the client shows typing immediately
            yield "data: {\"token\":\"\"}\n\n"

            for event in co.chat_stream(model="command-a-03-2025", messages=messages):
                # Be permissive: emit any chunk that includes text
                text = getattr(event, "text", None)
                if text:
                    full.append(text)
                    yield f"data: {json.dumps({'token': text})}\n\n"

            # If no tokens arrived, fall back to non-streaming call
            if not full:
                try:
                    res = co.chat(model="command-a-03-2025", messages=messages)
                    fallback = res.message.content[0].text if res and res.message and res.message.content else ""
                except Exception as e:
                    fallback = ""
                if fallback:
                    yield f"data: {json.dumps({'token': fallback})}\n\n"
                else:
                    yield f"data: {json.dumps({'error':'No streamed tokens and fallback empty'})}\n\n"

            yield "data: [DONE]\n\n"

            # Save history (assemble full reply)
            reply = "".join(full) if full else (fallback if 'fallback' in locals() else "")
            if reply:
                hist = session.get("messages", [])
                hist.append({"role": "user", "text": user_text})
                hist.append({"role": "bot",  "text": reply})
                session["messages"] = hist[-HISTORY_MAX:]

        except Exception as e:
            print("ERROR /chat_stream:", e, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # reduce proxy buffering
        "Connection": "keep-alive",
    }
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers=headers)

@app.post("/reset")
def reset():
    session.pop("messages", None)
    return jsonify({"ok": True})

@app.get("/health")
def health():
    return jsonify({"status": "ok", "backend": "cohere", "session_len": len(session.get("messages", []))})
