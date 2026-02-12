const chatEl = document.getElementById("chat");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("message");
const themeToggle = document.getElementById("themeToggle");
const clearBtn = document.getElementById("clearBtn");
const resetBtn = document.getElementById('resetBtn');


const tplBot = document.getElementById("msg-template");
const tplMe = document.getElementById("msg-me-template");
const tplTyping = document.getElementById("typing-template");


function scrollToBottom(){ chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" }); }
function nowTime(){ const d=new Date(); return d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}); }


function addMessage(text, me=false){
const node = (me ? tplMe : tplBot).content.cloneNode(true);
const content = node.querySelector('.content');
const time = node.querySelector('.time');
content.textContent = text;
time.textContent = nowTime();
const el = node.querySelector('.msg');
chatEl.appendChild(node);
scrollToBottom();
return el;
}


function addTyping(){
const node = tplTyping.content.cloneNode(true);
const el = node.querySelector('.msg');
chatEl.appendChild(node);
scrollToBottom();
return el;
}


async function sendToServer(message){
const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ message }) });
if(!res.ok) throw new Error(`HTTP ${res.status}`);
return await res.json();
}


function typeInto(bubbleEl, text, speed=16){
return new Promise(resolve => {
bubbleEl.textContent = '';
let i=0; const tick=()=>{ bubbleEl.textContent += text.slice(i, i+3); i+=3; scrollToBottom(); if(i<text.length){ setTimeout(tick, speed); } else { resolve(); }}; tick();
});
}


function setTheme(theme){
if(theme==='light'){ document.documentElement.classList.add('light'); }
else{ document.documentElement.classList.remove('light'); }
localStorage.setItem('theme', theme);
}


// init theme
setTheme(localStorage.getItem('theme') || 'dark');


themeToggle.addEventListener('click', ()=>{
const next = document.documentElement.classList.contains('light') ? 'dark' : 'light';
setTheme(next);
});


clearBtn.addEventListener('click', ()=>{ chatEl.innerHTML=''; addMessage('New chat started. How can I help?', false); });


// copy buttons 
chatEl.addEventListener('click', (e)=>{
const btn = e.target.closest('.copy');
if(!btn) return;
const bubble = btn.closest('.bubble');
const txt = bubble.querySelector('.content')?.textContent || '';
navigator.clipboard.writeText(txt).then(()=>{ btn.textContent='Copied!'; setTimeout(()=>btn.textContent='Copy', 1200); });
});


formEl.addEventListener('submit', async (e)=>{
e.preventDefault();
const msg = (inputEl.value||'').trim();
if(!msg) return;
addMessage(msg, true);
inputEl.value=''; inputEl.focus();

resetBtn?.addEventListener('click', async () => {
  try {
    await fetch('/reset', { method: 'POST' });
  } catch(e) {
    // ignore network errors
  }
  chatEl.innerHTML = '';
  addMessage('Session reset. How can I help?');
  inputEl.focus();
});

const typingNode = addTyping();

addMessage('Hey Eric! I\'m ready. Ask me anything.');
