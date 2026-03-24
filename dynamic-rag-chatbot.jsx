import { useState, useRef, useEffect, useCallback } from "react";
import * as mammoth from "mammoth";

// ═══════════════════════════════════════════════════════════
// DATABASE  —  persistent key/value store
//   docs:index              → string[]   all document_ids
//   sessions:{id}           → object     session metadata
//   messages:{id}           → object[]   chat history for that doc
//   doc:{id}:chunks         → object[]   TF-IDF vectors for that doc
// ═══════════════════════════════════════════════════════════
function uuid() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === "x" ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

const DB = {
  async getDocIndex() {
    try { const r = await window.storage.get("docs:index"); return r ? JSON.parse(r.value) : []; }
    catch { return []; }
  },
  async setDocIndex(arr) {
    try { await window.storage.set("docs:index", JSON.stringify(arr)); } catch {}
  },
  async getSession(id) {
    try { const r = await window.storage.get(`sessions:${id}`); return r ? JSON.parse(r.value) : null; }
    catch { return null; }
  },
  async setSession(id, s) {
    await window.storage.set(`sessions:${id}`, JSON.stringify(s));
  },
  async getMessages(id) {
    try { const r = await window.storage.get(`messages:${id}`); return r ? JSON.parse(r.value) : []; }
    catch { return []; }
  },
  async setMessages(id, msgs) {
    await window.storage.set(`messages:${id}`, JSON.stringify(msgs));
  },
  async addMessage(id, msg) {
    const msgs = await DB.getMessages(id);
    const next = [...msgs, msg];
    await DB.setMessages(id, next);
    return next;
  },
  async getChunks(id) {
    try { const r = await window.storage.get(`doc:${id}:chunks`); return r ? JSON.parse(r.value) : []; }
    catch { return []; }
  },
  async setChunks(id, chunks) {
    await window.storage.set(`doc:${id}:chunks`, JSON.stringify(chunks));
  },
  async deleteDoc(id) {
    const idx = await DB.getDocIndex();
    await Promise.all([
      window.storage.delete(`sessions:${id}`).catch(() => {}),
      window.storage.delete(`messages:${id}`).catch(() => {}),
      window.storage.delete(`doc:${id}:chunks`).catch(() => {}),
      DB.setDocIndex(idx.filter(x => x !== id)),
    ]);
  },
};

// ═══════════════════════════════════════════════════════════
// TF-IDF RETRIEVAL
// ═══════════════════════════════════════════════════════════
function tokenize(t) {
  return t.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);
}
function buildTFIDF(textChunks) {
  const N = textChunks.length, df = {};
  const tfs = textChunks.map(chunk => {
    const tokens = tokenize(chunk), tf = {};
    tokens.forEach(t => { tf[t] = (tf[t]||0) + 1; });
    Object.keys(tf).forEach(t => { tf[t] /= tokens.length; df[t] = (df[t]||0) + 1; });
    return tf;
  });
  return tfs.map(tf => {
    const vec = {};
    Object.keys(tf).forEach(t => { vec[t] = tf[t] * Math.log((N+1)/((df[t]||0)+1)); });
    return vec;
  });
}
function cosineSim(a, b) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  let dot=0, na=0, nb=0;
  keys.forEach(k => { const av=a[k]||0, bv=b[k]||0; dot+=av*bv; na+=av*av; nb+=bv*bv; });
  return na && nb ? dot/(Math.sqrt(na)*Math.sqrt(nb)) : 0;
}
function chunkText(text, size=800, overlap=120) {
  const chunks=[]; let s=0;
  while (s < text.length) { chunks.push(text.slice(s, s+size)); s += size-overlap; }
  return chunks;
}
// ═══════════════════════════════════════════════════════════
// GREETING DETECTION
// ═══════════════════════════════════════════════════════════
const GREETINGS = {
  "hi":         "Hello! 👋 Welcome to DocChat RAG. Upload a document and ask me anything about it.",
  "hello":      "Hi there! 👋 Ready to help. Upload a document and I'll answer your questions about it.",
  "hey":        "Hey! 👋 What document would you like to chat about?",
  "thanks":     "You're welcome! 😊 Let me know if you need anything else.",
  "thank you":  "Happy to help! 😊 Feel free to ask more questions.",
  "thankyou":   "Happy to help! 😊 Feel free to ask more questions.",
  "ok":         "Great! 👍 What would you like to know from the document?",
  "okay":       "Great! 👍 What would you like to know from the document?",
  "bye":        "Goodbye! 👋 Feel free to come back anytime.",
  "goodbye":    "See you later! 👋 Come back whenever you need help.",
  "good morning":"Good morning! ☀️ Ready to help you explore your documents.",
  "good afternoon":"Good afternoon! 🌤️ What document would you like to dive into?",
  "good evening":"Good evening! 🌙 What can I help you with today?",
  "how are you":"I'm doing great, thanks for asking! 😊 Ready to help you with your documents.",
  "what can you do":"I can read and answer questions from your uploaded documents — PDFs, Word files, ZIPs, and more! Just upload one and ask away. 📄",
  "help":       "Sure! Here's how to get started:\n1. **Upload** a document (PDF, DOCX, DOC, ZIP, TXT)\n2. **Select** it from the sidebar\n3. **Ask** any question — I'll answer strictly from the document. 🚀",
};

function getGreetingResponse(text) {
  const clean = text.trim().toLowerCase().replace(/[^a-z ]/g, "").trim();
  return GREETINGS[clean] || null;
}

// Returns { context, found }  —  found=false → refuse, never use training data
function retrieveContext(query, chunks) {
  if (!chunks.length) return { context:"", found:false };
  const qT = tokenize(query), qTF = {};
  qT.forEach(t => { qTF[t] = (qTF[t]||0) + 1/qT.length; });
  const scored = chunks
    .map(c => ({ c, sim: cosineSim(qTF, c.vec) }))
    .sort((a,b) => b.sim - a.sim);
  if ((scored[0]?.sim ?? 0) < 0.08) return { context:"", found:false };
  const rel = scored.slice(0,5).filter(x => x.sim >= 0.05);
  if (!rel.length) return { context:"", found:false };
  return {
    context: rel.map(x => `[Source: ${x.c.source}]\n${x.c.text}`).join("\n\n---\n\n"),
    found: true,
  };
}

// ═══════════════════════════════════════════════════════════
// ANTHROPIC API
// ═══════════════════════════════════════════════════════════
async function callClaude(messages, system) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model:"claude-sonnet-4-20250514", max_tokens:1000, system, messages }),
  });
  const data = await res.json();
  return data.content?.map(b => b.text||"").join("") || "No response.";
}

// ═══════════════════════════════════════════════════════════
// FILE EXTRACTION  —  PDF · DOCX · DOC · ZIP · text
// ═══════════════════════════════════════════════════════════
function getFileType(file) {
  const n = file.name.toLowerCase();
  if (n.endsWith(".pdf"))  return "pdf";
  if (n.endsWith(".docx")) return "docx";
  if (n.endsWith(".doc"))  return "doc";
  if (n.endsWith(".zip"))  return "zip";
  return "text";
}
async function extractPDF(buf) {
  const lib = window["pdfjs-dist/build/pdf"];
  if (!lib) throw new Error("PDF.js not ready");
  const pdf = await lib.getDocument({ data: buf }).promise;
  let text = "";
  for (let i=1; i<=pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const c = await page.getTextContent();
    text += c.items.map(x=>x.str).join(" ") + "\n";
  }
  return text;
}
async function extractDOCX(buf) {
  return (await mammoth.extractRawText({ arrayBuffer: buf })).value;
}
async function extractDOC(buf) {
  try { const r = await mammoth.extractRawText({ arrayBuffer: buf }); if (r.value?.trim().length > 20) return r.value; } catch {}
  const bytes = new Uint8Array(buf); let text = "";
  for (let i=0; i<bytes.length; i++) {
    const c = bytes[i];
    if ((c>=32 && c<127)||c===9||c===10||c===13) text += String.fromCharCode(c);
    else if (c>127) text += " ";
  }
  return text.replace(/\s{3,}/g,"\n").replace(/[^\x20-\x7E\n]/g,"").trim();
}
async function getJSZip() {
  if (window.JSZip) return window.JSZip;
  return new Promise((res,rej) => {
    const s = document.createElement("script");
    s.src = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js";
    s.onload = () => res(window.JSZip); s.onerror = rej;
    document.head.appendChild(s);
  });
}
const TEXT_EXTS = new Set([".txt",".md",".csv",".json",".xml",".html",".htm",
  ".js",".ts",".py",".java",".c",".cpp",".cs",".rs",".go",".yaml",".yml",".toml",".ini",".cfg",".log",".rst"]);
async function extractZIP(buf) {
  const JSZip = await getJSZip();
  const zip = await JSZip.loadAsync(buf);
  const results = [];
  for (const entry of Object.values(zip.files).filter(f=>!f.dir)) {
    const lower = entry.name.toLowerCase();
    if (lower.includes("__macosx")||lower.includes(".ds_store")) continue;
    try {
      const ext = "."+lower.split(".").pop();
      let text = "";
      if (ext===".pdf") { const b=await entry.async("arraybuffer"); text=await extractPDF(b); }
      else if (ext===".docx") { const b=await entry.async("arraybuffer"); text=await extractDOCX(b); }
      else if (ext===".doc") { const b=await entry.async("arraybuffer"); text=await extractDOC(b); }
      else if (TEXT_EXTS.has(ext)) { text=await entry.async("string"); }
      if (text.trim()) results.push({ name: entry.name, text });
    } catch(e) { console.warn("Skipped", entry.name, e.message); }
  }
  return results;
}
async function extractFile(file) {
  const type = getFileType(file);
  const buf = await file.arrayBuffer();
  if (type==="pdf")  return [{ name:file.name, text:await extractPDF(buf) }];
  if (type==="docx") return [{ name:file.name, text:await extractDOCX(buf) }];
  if (type==="doc")  return [{ name:file.name, text:await extractDOC(buf) }];
  if (type==="zip")  return await extractZIP(buf);
  return [{ name:file.name, text:await file.text() }];
}

// ═══════════════════════════════════════════════════════════
// ICONS
// ═══════════════════════════════════════════════════════════
const Ic = {
  Send:      () => <svg width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>,
  Upload:    () => <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>,
  Trash:     () => <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6M9 6V4h6v2"/></svg>,
  Spin:      () => <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24" style={{animation:"spin .8s linear infinite"}}><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>,
  Bot:       () => <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><line x1="12" y1="7" x2="12" y2="11"/><line x1="8" y1="15" x2="8" y2="17"/><line x1="16" y1="15" x2="16" y2="17"/></svg>,
  User:      () => <svg width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>,
  Folder:    () => <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>,
  Chat:      () => <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>,
  Plus:      () => <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>,
  Menu:      () => <svg width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>,
  ClearChat: () => <svg width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>,
};

// ═══════════════════════════════════════════════════════════
// FILE TYPE BADGE
// ═══════════════════════════════════════════════════════════
function TypeBadge({ type }) {
  const map = {
    pdf:  ["#e05a3a", "#fde8e2", "PDF"],
    docx: ["#2b7de9", "#deeafa", "DOCX"],
    doc:  ["#2b7de9", "#deeafa", "DOC"],
    zip:  ["#d4820a", "#fef3db", "ZIP"],
    text: ["#5a7a5a", "#e4f0e4", "TXT"],
  };
  const [color, bg, label] = map[type] || map.text;
  return (
    <span style={{
      fontSize:9, fontWeight:700, padding:"2px 6px", borderRadius:5,
      background:bg, color, fontFamily:"'Inter',monospace",
      letterSpacing:"0.4px", flexShrink:0,
    }}>{label}</span>
  );
}

// ═══════════════════════════════════════════════════════════
// MINIMAL MARKDOWN
// ═══════════════════════════════════════════════════════════
function MD({ text, isUser }) {
  if (!text) return null;
  const base = isUser ? "#fff" : "#1a1a2e";
  const lines = text.split("\n"); const els = []; let i=0;
  while (i < lines.length) {
    const l = lines[i];
    if (l.startsWith("### ")) els.push(<h3 key={i} style={{fontSize:13,fontWeight:700,color:base,margin:"8px 0 3px"}}>{inl(l.slice(4),isUser)}</h3>);
    else if (l.startsWith("## ")) els.push(<h2 key={i} style={{fontSize:14,fontWeight:700,color:base,margin:"10px 0 4px"}}>{inl(l.slice(3),isUser)}</h2>);
    else if (l.startsWith("# ")) els.push(<h1 key={i} style={{fontSize:15,fontWeight:700,color:base,margin:"12px 0 5px"}}>{inl(l.slice(2),isUser)}</h1>);
    else if (l.startsWith("- ")||l.startsWith("* ")) els.push(<li key={i} style={{marginLeft:16,lineHeight:1.65,fontSize:13.5}}>{inl(l.slice(2),isUser)}</li>);
    else if (/^\d+\. /.test(l)) els.push(<li key={i} style={{marginLeft:16,lineHeight:1.65,fontSize:13.5}}>{inl(l.replace(/^\d+\. /,""),isUser)}</li>);
    else if (l.startsWith("```")) {
      const code=[]; i++;
      while (i<lines.length && !lines[i].startsWith("```")) { code.push(lines[i]); i++; }
      els.push(<pre key={i} style={{fontFamily:"'Fira Code',monospace",fontSize:12,background:"#f0ede8",border:"1px solid #ddd8d0",borderRadius:8,padding:"10px 12px",margin:"6px 0",overflowX:"auto",color:"#2d2418"}}>{code.join("\n")}</pre>);
    }
    else if (l==="") els.push(<div key={i} style={{height:5}}/>);
    else els.push(<p key={i} style={{lineHeight:1.7,fontSize:13.5,margin:0}}>{inl(l,isUser)}</p>);
    i++;
  }
  return <>{els}</>;
}
function inl(text, isUser) {
  const parts=[]; const re=/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g; let last=0,m;
  while ((m=re.exec(text))!==null) {
    if (m.index>last) parts.push(text.slice(last,m.index));
    const t=m[0];
    if (t.startsWith("**")) parts.push(<strong key={m.index} style={{fontWeight:700}}>{t.slice(2,-2)}</strong>);
    else if (t.startsWith("`")) parts.push(<code key={m.index} style={{fontFamily:"'Fira Code',monospace",fontSize:12,background:isUser?"rgba(255,255,255,0.2)":"#ede8e0",padding:"1px 5px",borderRadius:4}}>{t.slice(1,-1)}</code>);
    else if (t.startsWith("*")) parts.push(<em key={m.index}>{t.slice(1,-1)}</em>);
    last=m.index+t.length;
  }
  if (last<text.length) parts.push(text.slice(last));
  return parts;
}

// ═══════════════════════════════════════════════════════════
// ROOT APP
// ═══════════════════════════════════════════════════════════
export default function App() {
  const [sessions,    setSessions]    = useState([]);
  const [activeDocId, setActiveDocId] = useState(null);
  const [messages,    setMessages]    = useState([]);   // only active doc's messages
  const [chunks,      setChunks]      = useState([]);   // only active doc's chunks
  const [input,       setInput]       = useState("");
  const [processing,  setProcessing]  = useState(false);
  const [indexing,    setIndexing]    = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [appLoaded,   setAppLoaded]   = useState(false);

  const fileRef   = useRef();
  const bottomRef = useRef();

  // ── PDF.js CDN loader
  useEffect(() => {
    if (!window["pdfjs-dist/build/pdf"]) {
      const s = document.createElement("script");
      s.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
      s.onload = () => {
        window["pdfjs-dist/build/pdf"].GlobalWorkerOptions.workerSrc =
          "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
      };
      document.head.appendChild(s);
    }
  }, []);

  // ── Bootstrap: restore sessions from persistent DB
  useEffect(() => {
    (async () => {
      try {
        const ids = await DB.getDocIndex();
        const loaded = (await Promise.all(ids.map(id => DB.getSession(id)))).filter(Boolean);
        setSessions(loaded);
        if (loaded.length > 0) await loadDoc(loaded[loaded.length-1].document_id);
      } catch(e) { console.error("Bootstrap:", e); }
      setAppLoaded(true);
    })();
  }, []);

  // ── Auto-scroll
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior:"smooth" }); }, [messages, processing]);

  // ── Switch active document — loads its messages & chunks from DB
  async function loadDoc(docId) {
    setActiveDocId(docId);
    const [msgs, cks] = await Promise.all([DB.getMessages(docId), DB.getChunks(docId)]);
    setMessages(msgs);
    setChunks(cks);
  }

  // ── Upload & index files
  const handleFiles = async (e) => {
    const uploaded = Array.from(e.target.files||[]);
    if (!uploaded.length) return;
    setIndexing(true);
    for (const file of uploaded) {
      try {
        const entries = await extractFile(file);
        const allChunks = [];
        for (const { name, text } of entries) {
          if (!text.trim()) continue;
          const c = chunkText(text);
          const vecs = buildTFIDF(c);
          const label = entries.length>1 ? `${file.name} → ${name}` : file.name;
          c.forEach((t,i) => allChunks.push({ text:t, vec:vecs[i], source:label }));
        }
        if (!allChunks.length) continue;
        const docId = uuid();
        const session = {
          document_id: docId,
          session_id:  uuid(),
          name:        file.name,
          type:        getFileType(file),
          chunkCount:  allChunks.length,
          entryCount:  entries.length,
          createdAt:   new Date().toISOString(),
        };
        await DB.setSession(docId, session);
        await DB.setChunks(docId, allChunks);
        const idx = await DB.getDocIndex();
        await DB.setDocIndex([...idx, docId]);
        setSessions(prev => [...prev, session]);
        // Auto-switch to newly uploaded doc
        setActiveDocId(docId);
        setMessages([]);
        setChunks(allChunks);
      } catch(err) { console.error(`Failed: ${file.name}`, err); }
    }
    setIndexing(false);
    if (fileRef.current) fileRef.current.value = "";
  };

  // ── Delete a document + all its history (optimistic UI)
  const deleteSession = (docId, e) => {
    e.stopPropagation();
    setSessions(prev => {
      const next = prev.filter(s => s.document_id !== docId);
      if (activeDocId === docId) {
        if (next.length > 0) loadDoc(next[next.length-1].document_id);
        else { setActiveDocId(null); setMessages([]); setChunks([]); }
      }
      return next;
    });
    DB.deleteDoc(docId).catch(err => console.error("Delete failed:", err));
  };

  // ── Clear chat history for active document only
  const clearChat = async () => {
    if (!activeDocId) return;
    await DB.setMessages(activeDocId, []);
    setMessages([]);
  };

  // ── Send a message — document-scoped, strict RAG, no training-data fallback
  const send = useCallback(async () => {
    if (!input.trim() || processing || !activeDocId) return;
    const q = input.trim();
    setInput("");

    const activeSess = sessions.find(s => s.document_id === activeDocId);
    const userMsg = {
      message_id:  uuid(),
      session_id:  activeSess?.session_id,
      document_id: activeDocId,
      role:        "user",
      content:     q,
      timestamp:   new Date().toISOString(),
    };

    const withUser = await DB.addMessage(activeDocId, userMsg);
    setMessages([...withUser]);
    setProcessing(true);

    try {
      // ── Greeting check: respond immediately, no RAG needed
      const greetingReply = getGreetingResponse(q);
      if (greetingReply) {
        const greetMsg = {
          message_id:  uuid(),
          session_id:  activeSess?.session_id,
          document_id: activeDocId,
          role:        "assistant",
          content:     greetingReply,
          timestamp:   new Date().toISOString(),
        };
        const withGreet = await DB.addMessage(activeDocId, greetMsg);
        setMessages([...withGreet]);
        setProcessing(false);
        return;
      }

      const { context, found } = retrieveContext(q, chunks);

      // ── Strict refusal: no relevant context found → never use training data
      if (!found) {
        const refusal = {
          message_id:  uuid(),
          session_id:  activeSess?.session_id,
          document_id: activeDocId,
          role:        "assistant",
          content:     `Sorry, there is no information regarding this in **${activeSess?.name}**.`,
          timestamp:   new Date().toISOString(),
        };
        const withRef = await DB.addMessage(activeDocId, refusal);
        setMessages([...withRef]);
        setProcessing(false);
        return;
      }

      // ── Intelligent RAG system prompt — strict but laymen-friendly
      const system = `You are an intelligent document assistant powered by RAG (Retrieval-Augmented Generation).

YOUR CORE RESPONSIBILITY:
- Answer user questions ONLY based on the provided document context below.
- Maintain accuracy and prevent hallucination at all times.
- Provide helpful, clear, and easy-to-understand responses in plain language.

BEHAVIOR GUIDELINES:

1. WHEN YOU HAVE RELEVANT INFORMATION:
   - Answer directly and clearly in simple, everyday language.
   - Mention the source document name naturally in your answer.
   - Provide supporting details from the document.
   - Use exact quotes when especially helpful.
   - Example: "According to ${activeSess?.name}, [answer with details]..."

2. WHEN INFORMATION IS NOT IN THE DOCUMENT:
   - Say: "I don't have information about this in ${activeSess?.name}."
   - Briefly suggest what topics you CAN help with based on the document.
   - Example: "I don't have information about [topic] in this document. However, I can help you with questions about [available topics from the doc]."

3. WHEN QUESTION IS OUT OF SCOPE:
   - Politely decline and explain you are limited to the uploaded document.
   - Ask if they want to upload a different document.
   - Example: "This question is outside the scope of ${activeSess?.name}. Would you like to upload a document about this topic?"

4. FOR FOLLOW-UP QUESTIONS:
   - Use previous conversation context.
   - Connect your answer to earlier answers where relevant.
   - Maintain conversation continuity.

5. TONE AND STYLE:
   - Professional but friendly and approachable.
   - Use plain, everyday language — avoid technical jargon unless the user uses it first.
   - Use bullet points or numbered lists when it makes the answer clearer.
   - Keep answers concise but complete.

CRITICAL RULES (never break these):
- NEVER make up or assume information not present in the document context.
- NEVER use your own training data or external knowledge.
- NEVER pretend to have information you don't have.
- If uncertain, say "I'm not sure about this" and cite what you DO know from the document.

Document name: ${activeSess?.name}

CONTEXT FROM DOCUMENT:
${context}`;

      const apiMsgs = withUser.map(m => ({ role:m.role, content:m.content }));
      const answer  = await callClaude(apiMsgs, system);

      const asstMsg = {
        message_id:  uuid(),
        session_id:  activeSess?.session_id,
        document_id: activeDocId,
        role:        "assistant",
        content:     answer,
        timestamp:   new Date().toISOString(),
      };
      const withAsst = await DB.addMessage(activeDocId, asstMsg);
      setMessages([...withAsst]);
    } catch(err) {
      const errMsg = {
        message_id:  uuid(),
        document_id: activeDocId,
        role:        "assistant",
        content:     "Something went wrong. Please try again.",
        timestamp:   new Date().toISOString(),
      };
      const withErr = await DB.addMessage(activeDocId, errMsg);
      setMessages([...withErr]);
    } finally {
      setProcessing(false);
    }
  }, [input, processing, activeDocId, chunks, sessions]);

  const onKey = (e) => { if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); send(); } };

  const activeSess   = sessions.find(s => s.document_id === activeDocId);
  const totalChunks  = sessions.reduce((a,s) => a+s.chunkCount, 0);

  // ═══════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
        @keyframes spin   { to { transform:rotate(360deg); } }
        @keyframes fadeUp { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
        @keyframes blink  { 0%,100%{opacity:1} 50%{opacity:0.25} }
        body { font-family:'Inter',sans-serif; background:#f5f0e8; }

        /* ── LAYOUT ── */
        .app { display:flex; height:100vh; background:#f5f0e8; color:#1a1a2e; overflow:hidden; }

        /* ── SIDEBAR ── */
        .sidebar {
          width:268px; min-width:268px;
          background:#ffffff;
          border-right:1px solid #e8e2d8;
          display:flex; flex-direction:column;
          transition:width .22s cubic-bezier(.4,0,.2,1), min-width .22s;
          overflow:hidden;
          box-shadow:2px 0 12px rgba(0,0,0,0.05);
        }
        .sidebar.closed { width:0; min-width:0; }
        .sb-inner { width:268px; display:flex; flex-direction:column; height:100%; }

        .sb-head {
          padding:18px 16px 14px;
          border-bottom:1px solid #ede8df;
          background:linear-gradient(135deg,#2563eb,#1d4ed8);
          flex-shrink:0;
        }
        .logo-row { display:flex; align-items:center; gap:10px; }
        .logo-icon { width:32px; height:32px; background:rgba(255,255,255,0.2);
          border-radius:8px; display:flex; align-items:center; justify-content:center;
          color:#fff; flex-shrink:0; }
        .logo-text h1 { font-size:13.5px; font-weight:700; color:#fff; letter-spacing:-0.2px; }
        .logo-text span { font-size:10px; color:rgba(255,255,255,0.6); }

        .sb-section { padding:12px 14px 6px; flex-shrink:0; }
        .sb-lbl { font-size:9.5px; font-weight:600; letter-spacing:1.2px; color:#9e9888;
          text-transform:uppercase; display:flex; align-items:center; justify-content:space-between; }
        .add-btn { display:flex; align-items:center; gap:4px; padding:4px 10px;
          border-radius:6px; background:#2563eb; border:none; color:#fff; cursor:pointer;
          font-size:10px; font-weight:600; font-family:'Inter',sans-serif;
          transition:all .15s; letter-spacing:0.2px; }
        .add-btn:hover:not(:disabled) { background:#1d4ed8; }
        .add-btn:disabled { opacity:0.45; cursor:not-allowed; }

        .doc-list { flex:1; overflow-y:auto; padding:6px 8px; }
        .doc-list::-webkit-scrollbar { width:3px; }
        .doc-list::-webkit-scrollbar-thumb { background:#e0d8cc; border-radius:2px; }

        .empty-list { padding:20px 12px; text-align:center; border:1.5px dashed #ddd6cc;
          border-radius:10px; cursor:pointer; transition:all .2s; color:#b0a898; margin:4px 0; }
        .empty-list:hover { border-color:#2563eb; color:#2563eb; background:#eff6ff; }
        .empty-list p { font-size:12px; font-weight:500; margin-top:8px; }
        .empty-list span { font-size:10px; color:#c8c0b4; display:block; margin-top:3px; }

        .doc-row { display:flex; align-items:center; gap:8px; padding:9px 10px;
          border-radius:8px; cursor:pointer; margin-bottom:3px;
          border:1px solid transparent; transition:all .15s; }
        .doc-row:hover { background:#f5f0e8; }
        .doc-row.active { background:#eff6ff; border-color:#bfdbfe; }
        .doc-info { flex:1; min-width:0; }
        .doc-name { font-size:12px; font-weight:500; color:#374151;
          white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .doc-meta { font-size:10px; color:#9ca3af; font-family:'Fira Code',monospace; margin-top:1px; }
        .doc-row.active .doc-name { color:#1d4ed8; font-weight:600; }
        .doc-row.active .doc-meta { color:#60a5fa; }
        .del-btn { padding:4px; color:#d1c8bc; cursor:pointer; border:none; background:none;
          border-radius:5px; display:flex; transition:all .12s; flex-shrink:0; }
        .del-btn:hover { color:#ef4444; background:#fef2f2; transform:scale(1.1); }
        .del-btn:active { transform:scale(0.9); }

        .sb-stats { display:grid; grid-template-columns:1fr 1fr; gap:6px;
          padding:10px 12px; border-top:1px solid #ede8df; flex-shrink:0; }
        .stat-card { background:#f9f6f1; border-radius:8px; padding:8px 10px;
          border:1px solid #ede8df; }
        .stat-card label { font-size:9px; color:#9e9888; text-transform:uppercase;
          letter-spacing:.8px; display:block; margin-bottom:2px; font-weight:600; }
        .stat-card p { font-size:18px; font-weight:700; color:#2563eb;
          font-family:'Fira Code',monospace; }

        /* ── MAIN ── */
        .main { flex:1; display:flex; flex-direction:column; overflow:hidden; min-width:0; background:#f5f0e8; }

        .topbar { height:54px; border-bottom:1px solid #e8e2d8; display:flex;
          align-items:center; justify-content:space-between;
          padding:0 18px; flex-shrink:0; background:#fff;
          box-shadow:0 1px 4px rgba(0,0,0,0.05); }
        .tb-left { display:flex; align-items:center; gap:10px; min-width:0; overflow:hidden; }
        .menu-btn { width:30px; height:30px; border:1px solid #e8e2d8; border-radius:7px;
          background:transparent; cursor:pointer; display:flex; align-items:center;
          justify-content:center; color:#9ca3af; transition:all .15s; flex-shrink:0; }
        .menu-btn:hover { background:#f5f0e8; color:#374151; }
        .doc-title-wrap { min-width:0; }
        .doc-title { font-size:13px; font-weight:600; color:#1f2937;
          white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:320px; }
        .tb-right { display:flex; align-items:center; gap:8px; flex-shrink:0; }
        .clear-btn { display:flex; align-items:center; gap:5px; padding:5px 11px;
          border-radius:7px; background:#fff; border:1px solid #e8e2d8; color:#6b7280;
          cursor:pointer; font-size:11px; font-family:'Inter',sans-serif; font-weight:500;
          transition:all .15s; }
        .clear-btn:hover { border-color:#fca5a5; color:#ef4444; background:#fef2f2; }
        .msg-badge { font-size:10px; padding:3px 10px; background:#eff6ff;
          border:1px solid #bfdbfe; border-radius:20px; color:#2563eb;
          font-weight:600; }

        /* ── CHAT ── */
        .chat-area { flex:1; overflow-y:auto; padding:20px 22px 10px;
          display:flex; flex-direction:column; gap:16px; }
        .chat-area::-webkit-scrollbar { width:4px; }
        .chat-area::-webkit-scrollbar-thumb { background:#ddd6cc; border-radius:2px; }

        .no-selection { flex:1; display:flex; flex-direction:column;
          align-items:center; justify-content:center; text-align:center; padding:24px; }
        .ns-icon { width:58px; height:58px; background:#fff; border:1px solid #e8e2d8;
          border-radius:16px; display:flex; align-items:center; justify-content:center;
          margin:0 auto 18px; color:#c4bdb4;
          box-shadow:0 2px 8px rgba(0,0,0,0.06); }
        .ns-icon svg { width:26px; height:26px; }
        .no-selection h3 { font-size:16px; font-weight:700; color:#6b7280; margin-bottom:7px; }
        .no-selection p  { font-size:12.5px; color:#9ca3af; max-width:250px; line-height:1.7; }

        .chat-empty { flex:1; display:flex; flex-direction:column;
          align-items:center; justify-content:center; text-align:center; padding:24px; }
        .ce-icon { width:54px; height:54px; background:#fff; border:1px solid #e8e2d8;
          border-radius:14px; display:flex; align-items:center; justify-content:center;
          margin:0 auto 16px; color:#93c5fd;
          box-shadow:0 2px 8px rgba(0,0,0,0.06); }
        .ce-icon svg { width:24px; height:24px; }
        .chat-empty h3 { font-size:15px; font-weight:700; color:#374151; margin-bottom:7px; }
        .chat-empty p  { font-size:12.5px; color:#9ca3af; max-width:270px; line-height:1.7; }

        .msg { display:flex; flex-direction:column; animation:fadeUp .18s ease; }
        .msg.user      { align-items:flex-end; }
        .msg.assistant { align-items:flex-start; }
        .msg-header { display:flex; align-items:center; gap:5px; margin-bottom:5px;
          font-size:10.5px; color:#9ca3af; font-weight:500; }

        .bubble { padding:12px 15px; border-radius:16px; max-width:78%; word-break:break-word; }
        .bubble.user {
          background:linear-gradient(135deg,#2563eb,#1d4ed8);
          color:#fff; border-bottom-right-radius:4px;
          box-shadow:0 2px 10px rgba(37,99,235,0.25);
        }
        .bubble.assistant {
          background:#fff; color:#374151;
          border:1px solid #e8e2d8; border-bottom-left-radius:4px;
          box-shadow:0 1px 4px rgba(0,0,0,0.06);
        }
        .msg-time { font-size:9.5px; color:#c4bdb4; margin-top:4px;
          font-family:'Fira Code',monospace; }

        .thinking { display:flex; align-items:center; gap:8px; font-size:12px;
          color:#9ca3af; animation:fadeUp .18s ease; }
        .think-bubble { background:#fff; border:1px solid #e8e2d8; border-radius:12px;
          padding:10px 14px; display:flex; align-items:center; gap:8px;
          box-shadow:0 1px 4px rgba(0,0,0,0.05); }
        .dots { display:inline-flex; gap:4px; }
        .d { width:5px; height:5px; border-radius:50%; background:#93c5fd; animation:blink 1.1s ease infinite; }
        .d:nth-child(2){animation-delay:.2s;} .d:nth-child(3){animation-delay:.4s;}

        /* ── INPUT ── */
        .input-area { padding:14px 18px 18px; border-top:1px solid #e8e2d8; background:#fff; flex-shrink:0; }
        .input-box { display:flex; align-items:flex-end; gap:9px; max-width:820px; margin:0 auto;
          background:#f9f6f1; border:1.5px solid #e8e2d8; border-radius:14px;
          padding:9px 10px; transition:border-color .15s, box-shadow .15s; }
        .input-box:focus-within { border-color:#93c5fd; box-shadow:0 0 0 3px rgba(147,197,253,0.2); }
        textarea { flex:1; background:transparent; border:none; outline:none; resize:none;
          font-family:'Inter',sans-serif; font-size:13.5px; color:#1f2937;
          line-height:1.5; max-height:130px; min-height:22px; padding:3px 2px; }
        textarea::placeholder { color:#c4bdb4; }
        textarea:disabled { opacity:0.5; cursor:not-allowed; }
        .send-btn { width:34px; height:34px; background:linear-gradient(135deg,#2563eb,#1d4ed8);
          border:none; border-radius:10px; cursor:pointer; display:flex; align-items:center;
          justify-content:center; color:#fff; flex-shrink:0; align-self:flex-end;
          transition:all .15s; box-shadow:0 2px 6px rgba(37,99,235,0.3); }
        .send-btn:hover:not(:disabled) { transform:scale(1.05); box-shadow:0 3px 10px rgba(37,99,235,0.4); }
        .send-btn:disabled { opacity:0.3; cursor:not-allowed; box-shadow:none; }
        .input-note { text-align:center; font-size:10px; color:#d1c8bc; margin-top:8px; }
      `}</style>

      <div className="app">

        {/* ══ SIDEBAR ══ */}
        <aside className={`sidebar${sidebarOpen ? "" : " closed"}`}>
          <div className="sb-inner">

            <div className="sb-head">
              <div className="logo-row">
                <div className="logo-icon"><Ic.Folder /></div>
                <div className="logo-text">
                  <h1>DocChat RAG</h1>
                  <span>Per-document sessions</span>
                </div>
              </div>
            </div>

            <div className="sb-section">
              <div className="sb-lbl">
                Documents
                <button className="add-btn" disabled={indexing} onClick={() => fileRef.current?.click()}>
                  {indexing ? <><Ic.Spin /> Indexing…</> : <><Ic.Plus /> Add</>}
                </button>
              </div>
            </div>

            <input ref={fileRef} type="file" multiple
              accept=".pdf,.docx,.doc,.zip,.txt,.md,.csv,.json"
              style={{display:"none"}} onChange={handleFiles} />

            <div className="doc-list">
              {!appLoaded ? (
                <div style={{padding:"16px",textAlign:"center",color:"#c4bdb4",fontSize:12}}>Loading…</div>
              ) : sessions.length === 0 ? (
                <div className="empty-list" onClick={() => fileRef.current?.click()}>
                  <Ic.Upload />
                  <p>Upload a document</p>
                  <span>PDF · DOCX · DOC · ZIP · TXT · MD</span>
                </div>
              ) : sessions.map(s => (
                <div
                  key={s.document_id}
                  className={`doc-row${activeDocId===s.document_id?" active":""}`}
                  onClick={() => loadDoc(s.document_id)}
                >
                  <TypeBadge type={s.type} />
                  <div className="doc-info">
                    <div className="doc-name">{s.name}</div>
                    <div className="doc-meta">
                      {s.chunkCount} chunks{s.type==="zip"?` · ${s.entryCount} files`:""}
                    </div>
                  </div>
                  <button className="del-btn" title="Delete document & history"
                    onClick={e => deleteSession(s.document_id, e)}>
                    <Ic.Trash />
                  </button>
                </div>
              ))}
            </div>

            <div className="sb-stats">
              <div className="stat-card"><label>Docs</label><p>{sessions.length}</p></div>
              <div className="stat-card"><label>Chunks</label><p>{totalChunks}</p></div>
            </div>

          </div>
        </aside>

        {/* ══ MAIN ══ */}
        <main className="main">

          <div className="topbar">
            <div className="tb-left">
              <button className="menu-btn" onClick={() => setSidebarOpen(p=>!p)}><Ic.Menu /></button>
              {activeSess ? (
                <div className="doc-title-wrap">
                  <div className="doc-title">{activeSess.name}</div>

                </div>
              ) : (
                <div className="doc-title" style={{color:"#9ca3af"}}>No document selected</div>
              )}
            </div>
            <div className="tb-right">
              {activeSess && messages.length > 0 && (
                <button className="clear-btn" onClick={clearChat}>
                  <Ic.ClearChat /> Clear
                </button>
              )}
              {activeSess && (
                <div className="msg-badge">{messages.length} msg{messages.length!==1?"s":""}</div>
              )}
            </div>
          </div>

          {!activeSess ? (
            <div className="no-selection">
              <div className="ns-icon"><Ic.Folder /></div>
              <h3>No document selected</h3>
              <p>Upload a document or choose one from the sidebar to start chatting.</p>
            </div>
          ) : (
            <>
              <div className="chat-area">
                {messages.length === 0 ? (
                  <div className="chat-empty">
                    <div className="ce-icon"><Ic.Chat /></div>
                    <h3>Ready to answer</h3>
                    <p>Ask anything about <strong style={{color:"#2563eb"}}>{activeSess.name}</strong>. Answers are strictly from the document.</p>
                  </div>
                ) : messages.map(m => (
                  <div key={m.message_id} className={`msg ${m.role}`}>
                    <div className="msg-header">
                      {m.role==="assistant" ? <Ic.Bot /> : <Ic.User />}
                      {m.role==="assistant" ? "Assistant" : "You"}
                    </div>
                    <div className={`bubble ${m.role}`}>
                      <MD text={m.content} isUser={m.role==="user"} />
                    </div>
                    <div className="msg-time">
                      {new Date(m.timestamp).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"})}
                    </div>
                  </div>
                ))}
                {processing && (
                  <div className="thinking">
                    <div className="think-bubble">
                      <Ic.Bot />
                      <span className="dots"><div className="d"/><div className="d"/><div className="d"/></span>
                      <span style={{fontSize:12,color:"#9ca3af"}}>Searching document…</span>
                    </div>
                  </div>
                )}
                <div ref={bottomRef} />
              </div>

              <div className="input-area">
                <div className="input-box">
                  <textarea
                    rows={1}
                    value={input}
                    disabled={processing}
                    onChange={e => {
                      setInput(e.target.value);
                      e.target.style.height = "auto";
                      e.target.style.height = Math.min(e.target.scrollHeight,130)+"px";
                    }}
                    onKeyDown={onKey}
                    placeholder={`Ask about ${activeSess.name}…`}
                  />
                  <button className="send-btn" disabled={!input.trim()||processing} onClick={send}>
                    <Ic.Send />
                  </button>
                </div>
                <div className="input-note">Enter to send · Shift+Enter for new line · Answers are scoped to the selected document only</div>
              </div>
            </>
          )}
        </main>
      </div>
    </>
  );
}
