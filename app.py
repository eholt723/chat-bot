from flask import Flask, render_template, request, jsonify
import os


# Optional: Cohere
# from dotenv import load_dotenv
# import cohere
# load_dotenv()
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# co = cohere.ClientV2(api_key=COHERE_API_KEY)
# MODEL = os.getenv("COHERE_MODEL", "command-r")


SYSTEM_PROMPT = "Be concise, friendly, and helpful. If unsure, say so briefly."


app = Flask(__name__, template_folder="Templates", static_folder="static")


@app.route("/chat", methods=["POST"])
def chat():
data = request.get_json(silent=True) or {}
user_msg = (data.get("message") or "").strip()
if not user_msg:
return jsonify({"reply": "Say that again?"})


# === Cohere (uncomment to use) ===
# try:
# resp = co.chat(
# model=MODEL,
# messages=[{"role":"system","content":SYSTEM_PROMPT},
# {"role":"user","content":user_msg}],
# temperature=0.5,
# )
# reply_text = resp.text
# except Exception as e:
# reply_text = f"Oops—model error: {e}"


# Temporary echo fallback
reply_text = f"You said: {user_msg}\n(Your model hook is ready—just uncomment it.)"
return jsonify({"reply": reply_text})


@app.route("/")
def index():
return render_template("index.html")


if __name__ == "__main__":
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port, debug=True)