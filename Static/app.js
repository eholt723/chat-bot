const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const input = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const typing = document.getElementById("typing");
const resetBtn = document.getElementById("resetBtn");

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chat.appendChild(div);
  // Auto-scroll so new messages push prior ones upward
  chat.scrollTop = chat.scrollHeight;
}

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let msg = "Request failed";
    try { const data = await res.json(); msg = data.error || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  addMessage("user", text);
  input.value = "";
  input.focus();

  typing.classList.remove("hidden");
  sendBtn.disabled = true;

  try {
    const data = await postJSON("/chat", { message: text });
    typing.classList.add("hidden");
    sendBtn.disabled = false;

    if (!data.ok) {
      addMessage("bot", `Error: ${data.error || "Unknown error"}`);
      return;
    }
    addMessage("bot", data.reply);
  } catch (err) {
    typing.classList.add("hidden");
    sendBtn.disabled = false;
    addMessage("bot", `Network error: ${err.message}`);
  }
});

resetBtn.addEventListener("click", async () => {
  try {
    await fetch("/reset", { method: "POST" });
  } finally {
    chat.innerHTML = "";
  }
});
