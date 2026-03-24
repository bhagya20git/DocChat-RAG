📄 DocChat RAG — Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload documents and ask questions about them. Answers are strictly sourced from your uploaded documents — no hallucination, no made-up information.


✨ Features :

📁 Multi-format support — Upload PDF, DOCX, DOC, ZIP, TXT, MD, CSV, JSON

🔒 Strict document-only answers — Never answers from general knowledge

💬 Per-document chat sessions — Each document has its own isolated chat history

💾 Persistent storage — Chat history and documents survive page refreshes

⚡ Fast deletion — Optimistic UI with parallel DB cleanup

🤝 Greeting detection — Responds naturally to greetings and small talk

🗜️ ZIP support — Extracts and indexes all files inside a ZIP automatically

🧠 TF-IDF retrieval — Finds the most relevant chunks from your document

🎨 Clean light UI — Warm, minimal design built with React


🚀 Getting Started : 

Prerequisites


Node.js v18 or higher
npm (comes with Node.js)


Installation

1. Clone the repository
git clone https://github.com/bhagya20git/DocChat-RAG.git
cd doc-chat-rag

# 2. Create a new Vite + React project
npm create vite@latest . -- --template react

# 3. Install dependencies
npm install mammoth

Setup
Replace src/App.jsx with the dynamic-rag-chatbot.jsx file from this repo.

Run locally :

npm run dev

Open http://localhost:5173 in your browser.


🏗️ Project Structure :

doc-chat-rag/ 

├── src/

│   └── App.jsx  
# Main chatbot component (dynamic-rag-chatbot.jsx)
├── public/

├── index.html

├── package.json

├── vite.config.js

└── README.md


🧠 How It Works :

1. User uploads document

2. Text extracted (PDF.js / mammoth / JSZip)

3. Text split into chunks (800 chars, 120 overlap)

4. TF-IDF vectors built for each chunk

5. User asks a question

6. Question matched against chunks via cosine similarity

7. Top relevant chunks passed as context to Claude API
                   
8. Answers strictly from the context


🔑 API

This project uses the Anthropic Claude API (claude-sonnet-4-20250514).

The API key is handled by the Claude.ai artifact environment. If you are running this outside of Claude.ai, you will need to add your own API key:
js// In the callClaude function, add your key:
headers: {
  "Content-Type": "application/json",
  "x-api-key": "YOUR_ANTHROPIC_API_KEY",
  "anthropic-version": "2023-06-01"
}


🛠️ Built With :


React — UI framework

Vite — Build tool

mammoth — DOCX/DOC extraction

PDF.js — PDF text extraction

JSZip — ZIP file handling

Anthropic Claude API — LLM for answers
