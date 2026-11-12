const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};

const state = { messages: [], docs: [], streaming: false, controller: null, keyOk: false };

const thread = $("thread"), scroll = $("scroll"), input = $("input"), sendBtn = $("send");
let welcome = $("welcome");
const welcomeHtml = welcome.outerHTML;

/* ------------------------------------------------------------- helpers -- */
async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let message = `Request failed (${res.status}).`;
    try { message = (await res.json()).detail || message; } catch {}
    throw new Error(message);
  }
  return res.json();
}

let toastTimer;
function toast(message) {
  const t = $("toast");
  t.textContent = message;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 6000);
}

function nearBottom() {
  return scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 120;
}
function toBottom() { scroll.scrollTop = scroll.scrollHeight; }

function renderMarkdown(text) {
  if (window.marked && window.DOMPurify) {
    return DOMPurify.sanitize(marked.parse(text, { breaks: true, gfm: true }));
  }
  return el("p", null, text).outerHTML; // CDN unavailable: plain text
}

/* -------------------------------------------------------------- status -- */
async function loadStatus() {
  try {
    const s = await api("/api/status");
    state.keyOk = s.key_configured;
    $("banner").hidden = s.key_configured;
    $("key-dot").className = "dot " + (s.key_configured ? "ok" : "bad");
    $("key-text").textContent = s.key_configured ? "V-max is ready" : "API key missing";
  } catch {
    $("key-dot").className = "dot bad";
    $("key-text").textContent = "Server not reachable";
  }
}

$("recheck").addEventListener("click", loadStatus);

/* ----------------------------------------------------------- documents -- */
function docMeta(d) {
  const parts = [];
  if (d.pages) parts.push(`${d.pages} page${d.pages === 1 ? "" : "s"}`);
  parts.push(d.size > 1048576 ? `${(d.size / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(d.size / 1024))} KB`);
  return parts.join(", ");
}

function docRow(name, meta, type) {
  const li = el("li", "doc");
  const badge = el("div", "doc-type", type);
  const body = el("div");
  body.style.minWidth = "0";
  const title = el("div", "doc-name", name);
  title.title = name;
  body.append(title, el("div", "doc-meta", meta));
  li.append(badge, body);
  return li;
}

function renderDocs() {
  const list = $("doc-list");
  list.querySelectorAll(".doc:not(.busy):not(.failed)").forEach((n) => n.remove());
  for (const d of state.docs) {
    const ext = (d.name.split(".").pop() || "").toUpperCase().slice(0, 4);
    const li = docRow(d.name, docMeta(d), ext);
    const remove = el("button", "icon-btn");
    remove.type = "button";
    remove.setAttribute("aria-label", `Remove ${d.name}`);
    remove.innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>';
    remove.addEventListener("click", async () => {
      try {
        await api(`/api/documents/${d.id}`, { method: "DELETE" });
        state.docs = state.docs.filter((x) => x.id !== d.id);
        renderDocs();
      } catch (e) { toast(e.message); }
    });
    li.append(remove);
    list.prepend(li);
  }
  const n = state.docs.length;
  $("docs-empty").hidden = n > 0 || list.children.length > 0;
  $("doc-toggle").hidden = n === 0;
  $("doc-toggle-text").textContent = n === 1 ? "Answer from 1 document" : `Answer from ${n} documents`;
  input.placeholder = n && $("use-docs").checked ? "Ask about your documents" : "Ask anything";
}
$("use-docs").addEventListener("change", renderDocs);

async function loadDocs() {
  try {
    state.docs = (await api("/api/documents")).documents;
    renderDocs();
  } catch {}
}

async function uploadFiles(files) {
  for (const file of files) {
    const row = docRow(file.name, "Reading and indexing", "");
    row.classList.add("busy");
    row.firstChild.append(el("div", "spinner"));
    $("doc-list").prepend(row);
    $("docs-empty").hidden = true;
    try {
      const body = new FormData();
      body.append("file", file);
      const doc = await api("/api/documents", { method: "POST", body });
      row.remove();
      state.docs.push(doc);
      $("use-docs").checked = true;
      renderDocs();
    } catch (e) {
      row.classList.replace("busy", "failed");
      row.firstChild.replaceChildren();
      row.querySelector(".doc-meta").textContent = e.message;
      const dismiss = el("button", "icon-btn");
      dismiss.type = "button";
      dismiss.setAttribute("aria-label", "Dismiss");
      dismiss.innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>';
      dismiss.addEventListener("click", () => { row.remove(); renderDocs(); });
      row.append(dismiss);
      loadStatus();
    }
  }
}

$("file-input").addEventListener("change", (e) => {
  uploadFiles([...e.target.files]);
  e.target.value = "";
});
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => document.addEventListener(t, (e) => {
  if (e.dataTransfer?.types.includes("Files")) { e.preventDefault(); drop.classList.add("over"); }
}));
["dragleave", "drop"].forEach((t) => document.addEventListener(t, (e) => {
  if (t === "drop") {
    e.preventDefault();
    if (e.dataTransfer?.files.length) uploadFiles([...e.dataTransfer.files]);
  }
  if (t === "drop" || !e.relatedTarget) drop.classList.remove("over");
}));

/* ---------------------------------------------------------------- chat -- */
function addUser(text) {
  welcome?.remove();
  welcome = null;
  thread.append(el("div", "msg-user", text));
}

function addBot() {
  const wrap = el("div", "msg-bot");
  const who = el("div", "who");
  who.append(el("span", "dot"), el("span", null, "V-max"));
  const body = el("div", "prose");
  const pending = el("div", "pending");
  pending.append(el("div", "spinner"), el("span", null, "Working"));
  body.append(pending);
  wrap.append(who, body);
  thread.append(wrap);
  return { wrap, body, label: pending.lastChild };
}

/* Turn [1] style markers into buttons that open the matching excerpt. */
function linkCitations(root, sources, open) {
  const known = new Set(sources.map((s) => s.n));
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => (n.parentElement.closest("code, pre, a") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  const nodes = [];
  while (walker.nextNode()) if (/\[\d{1,2}(?:\s*,\s*\d{1,2})*\]/.test(walker.currentNode.nodeValue)) nodes.push(walker.currentNode);
  const cited = new Set();
  for (const node of nodes) {
    const frag = document.createDocumentFragment();
    let last = 0;
    const text = node.nodeValue;
    for (const m of text.matchAll(/\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g)) {
      const nums = m[1].split(",").map((x) => +x.trim());
      if (!nums.every((n) => known.has(n))) continue;
      frag.append(text.slice(last, m.index));
      for (const n of nums) {
        cited.add(n);
        const b = el("button", "cite", String(n));
        b.type = "button";
        b.dataset.n = n;
        b.setAttribute("aria-label", `Show source ${n}`);
        b.addEventListener("click", () => open(n, true));
        frag.append(b);
      }
      last = m.index + m[0].length;
    }
    frag.append(text.slice(last));
    node.replaceWith(frag);
  }
  return cited;
}

function renderAnswer(bot, text, sources, final) {
  bot.body.innerHTML = renderMarkdown(text);
  bot.wrap.querySelector(".sources")?.remove();
  if (!sources.length) return;

  let excerpt, chips;
  const open = (n, scrollTo) => {
    const s = sources.find((x) => x.n === n);
    const same = excerpt.dataset.n === String(n) && !excerpt.hidden;
    excerpt.hidden = same && !scrollTo;
    excerpt.dataset.n = n;
    bot.wrap.querySelectorAll(".cite, .source").forEach((b) => b.classList.toggle("on", !excerpt.hidden && +b.dataset.n === n));
    if (excerpt.hidden) return;
    const head = el("div", "excerpt-head");
    if (s.url && /^https?:\/\//.test(s.url)) {
      const link = el("a", null, s.doc);
      link.href = s.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      head.append(link);
    } else {
      head.textContent = s.doc + (s.page ? `, page ${s.page}` : "");
    }
    excerpt.replaceChildren(head, el("div", "excerpt-text", s.text));
    if (scrollTo) excerpt.scrollIntoView({ block: "nearest" });
  };

  const cited = linkCitations(bot.body, sources, (n, s) => open(n, s));
  if (!final) return;
  const shown = cited.size ? sources.filter((s) => cited.has(s.n)) : sources;

  const box = el("div", "sources");
  box.append(el("div", "sources-title", cited.size ? "Sources cited" : "Sources checked"));
  chips = el("div", "source-chips");
  for (const s of shown) {
    const b = el("button", "source");
    b.type = "button";
    b.dataset.n = s.n;
    b.append(el("b", null, String(s.n)), el("span", null, s.doc + (s.page ? `, p. ${s.page}` : "")));
    b.addEventListener("click", () => open(s.n, false));
    chips.append(b);
  }
  excerpt = el("div", "excerpt");
  excerpt.hidden = true;
  box.append(chips, excerpt);
  bot.wrap.append(box);
}

function setStreaming(on) {
  state.streaming = on;
  sendBtn.classList.toggle("stop", on);
  sendBtn.setAttribute("aria-label", on ? "Stop answering" : "Send message");
  refreshSend();
}
function refreshSend() {
  sendBtn.disabled = !state.streaming && !input.value.trim();
}

async function send(text) {
  text = text.trim();
  if (!text || state.streaming) return;
  input.value = "";
  autosize();
  addUser(text);
  state.messages.push({ role: "user", content: text });
  const bot = addBot();
  toBottom();
  setStreaming(true);

  let answer = "", sources = [], failed = null, frame = 0;
  const paint = () => {
    frame = 0;
    const stick = nearBottom();
    renderAnswer(bot, answer, sources, false);
    if (stick) toBottom();
  };
  state.controller = new AbortController();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: state.controller.signal,
      body: JSON.stringify({
        messages: state.messages,
        use_documents: state.docs.length > 0 && $("use-docs").checked,
      }),
    });
    if (!res.ok) {
      let message = `Request failed (${res.status}).`;
      try { message = (await res.json()).detail || message; } catch {}
      throw new Error(message);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const raw of events) {
        if (!raw.startsWith("data:")) continue;
        const ev = JSON.parse(raw.slice(5));
        if (ev.type === "sources") { sources = ev.sources; bot.label.textContent = "Reading your documents"; }
        else if (ev.type === "status") { if (!answer) bot.label.textContent = ev.text; }
        else if (ev.type === "thinking") { if (!answer) bot.label.textContent = "Thinking"; }
        else if (ev.type === "token") { answer += ev.text; if (!frame) frame = requestAnimationFrame(paint); }
        else if (ev.type === "error") failed = ev.message;
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") failed = e.message || "Something went wrong. Send again.";
  }

  cancelAnimationFrame(frame);
  const stick = nearBottom();
  if (answer) {
    renderAnswer(bot, answer, sources, true);
    state.messages.push({ role: "assistant", content: answer });
  } else {
    bot.body.replaceChildren();
    state.messages.pop(); // the question got no answer; keep history consistent
    if (!failed) bot.body.append(el("div", "pending", "Stopped before an answer arrived."));
  }
  if (failed) {
    bot.wrap.append(el("div", "msg-error", failed));
    loadStatus();
  }
  if (stick) toBottom();
  state.controller = null;
  setStreaming(false);
  input.focus();
}

/* ------------------------------------------------------------ composer -- */
function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
  refreshSend();
}
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send(input.value);
  }
});
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.streaming) state.controller?.abort();
  else send(input.value);
});

$("new-chat").addEventListener("click", () => {
  state.controller?.abort();
  state.messages = [];
  thread.innerHTML = welcomeHtml;
  welcome = $("welcome");
  closeSidebar();
  input.focus();
});

/* ------------------------------------------------------ mobile sidebar -- */
function closeSidebar() {
  $("sidebar").classList.remove("open");
  $("scrim").classList.remove("open");
}
$("menu-btn").addEventListener("click", () => {
  $("sidebar").classList.add("open");
  $("scrim").classList.add("open");
});
$("scrim").addEventListener("click", closeSidebar);

refreshSend();
loadStatus();
loadDocs();
input.focus();
