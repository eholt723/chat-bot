const chatEl = document.getElementById("chat");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("message");

const tplBot = document.getElementById("msg-template");
const tplMe = document.getElementById("msg-me-template");
const tplTyping = document.getElementById("typing-template");

function scrollToBottom() {
  chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" });
}

function addMessage(text, me=false) {
  const node = (me ? tplMe : tplBot).content.cloneNode(true);
  const bubble = node.querySelector(".bubble");
  bubble.textContent = text;
  chatEl.appendChild(node);
  scrollToBottom();
}

function addTyping() {
  const node = tplTyping.content.cloneNode(true);
  const el = node.querySelector(".msg");
  chatEl.appendChild(node);
  scrollToBottom();
  return el; // return the typing node root for removal later
}

async function sendToServer(message) {
  const res = await fetch("/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ message })
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

/** typewriter animation (prevents empty bubble flashes) */
function typeInto(bubbleEl, text, speed = 16) {
  return new Promise(resolve => {
    let i = 0;
    const tick = () => {
      // append a few chars per frame for snappier feel
      bubbleEl.textContent += text.slice(i, i+3);
      i += 3;
      scrollToBottom();
      if (i < text.length) {
        setTimeout(tick, speed);
      } else {
        resolve();
      }
    };
    tick();
  });
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = (inputEl.value || "").trim();
  if (!msg) return;

  // show my message
  addMessage(msg, true);
  inputEl.value = "";

  // show typing bubble
  const typingNode = addTyping();
  const typingBubble = typingNode.querySelector(".bubble");

  try {
    const { reply } = await sendToServer(msg);

    // replace typing bubble with an empty bot bubble then type into it
    typingBubble.classList.remove("typing");
    typingBubble.innerHTML = ""; // clear dots
    await typeInto(typingBubble, reply ?? "");
  } catch (err) {
    typingBubble.classList.remove("typing");
    typingBubble.textContent = `Error: ${err.message}`;
  }
});
