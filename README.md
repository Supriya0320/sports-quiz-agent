# 🏆 AI-Powered Sports Quiz Generation Agent

An AI agent that generates factually grounded, multiple-choice sports quizzes
for social media, using **Retrieval-Augmented Generation (RAG)**: a local
**ChromaDB** vector database of historic facts, combined with **live web
search** for fresh news, feeding into an **LLM** that writes the quiz.

Built for: *AI Product/Engineer Intern Assignment — Statupbox*

---

## ✨ What it does

1. You pick a **sport** (Cricket, Football, Tennis, Badminton, Basketball) and
   a **difficulty** (Easy / Medium / Hard) in the sidebar.
2. The agent retrieves:
   - **Historic facts** from ChromaDB (offline, curated dataset)
   - **Live news/results** from a DuckDuckGo web search
3. Both sources are merged into a single grounding context and sent to an LLM
   with strict instructions: *only use facts found in the context, never
   hallucinate.*
4. The LLM returns structured JSON, which the app renders as an interactive
   quiz — click an option to instantly see if you were right, plus an
   explanation citing the grounding fact.
5. Click **Generate Fresh Quiz** any time to regenerate a brand-new, unique
   set of questions.

---

## 🧠 How RAG works here (in plain English)

Think of a plain LLM as a student taking a closed-book exam — it can only use
what it memorized during training, and might guess (hallucinate) if it
doesn't actually know something.

RAG turns this into an **open-book exam**: before answering, the agent first
looks up real facts (from ChromaDB and the live web) and hands them to the
LLM as reference material. The LLM is instructed to answer *only* using that
material.

```
[User Input: Sport + Difficulty]
        │
        ▼
 ┌───────────┐     1. Search local facts   ──> [ChromaDB Vector Store]
 │ AI Agent  │ ──> 2. Search live web news  ──> [DuckDuckGo Search]
 └───────────┘
        │
        ▼  (retrieved text snippets)
 Combined Context + Structured Prompt
        │
        ▼
    LLM (OpenAI)
        │
        ▼
 Interactive quiz rendered in Streamlit
```

---

## 📁 Project structure

```
sports-quiz-agent/
│
├── .env.example          # Template for your API key — copy to .env
├── .gitignore             # Keeps .env and chroma_db/ out of git
├── requirements.txt       # Python dependencies
├── README.md              # You are here
│
├── data/
│   └── sports_facts.json  # Offline curated facts (25 facts, 5 sports)
│
├── chroma_db/              # Auto-created by ChromaDB to store vectors
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Loads env vars & app-wide settings
│   ├── database.py         # All ChromaDB logic (insert & query)
│   ├── search.py            # All DuckDuckGo web search logic
│   └── generator.py         # RAG orchestration: context + prompt + LLM
│
└── app.py                   # Streamlit UI — the entry point
```

---

## ⚙️ Setup instructions

### 1. Prerequisites
- Python **3.9, 3.10, or 3.11** (avoid 3.12+ for ChromaDB compatibility)
- An **OpenAI API key** ([platform.openai.com](https://platform.openai.com))

### 2. Clone / unzip and enter the project
```bash
cd sports-quiz-agent
```

### 3. Create and activate a virtual environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Add your API key
```bash
cp .env.example .env
```
Then open `.env` and paste your real key:
```
OPENAI_API_KEY=sk-proj-...
```

### 6. Run the app
```bash
streamlit run app.py
```
Streamlit will open the dashboard in your browser (usually `http://localhost:8501`).

---

## 🕹️ Using the dashboard

1. Choose a **sport** and **difficulty** in the sidebar.
2. Click **🎲 Generate Fresh Quiz**.
3. Click any answer option to get instant feedback (✅/❌) and an explanation.
4. Expand **🔍 Inspect Ground Truth** to see exactly which facts (ChromaDB +
   web search) grounded the quiz — useful for auditing accuracy.
5. Expand **📋 Copy as Plain Text** to grab a ready-to-post version for social
   media.
6. Click **Generate Fresh Quiz** again any time for a new, unique quiz.

---

## 🔧 Design notes / how hallucinations are avoided

- The system prompt explicitly instructs the LLM to use **only** the supplied
  context and to avoid inventing facts, scores, or names.
- The LLM is required to respond in **strict JSON** (`response_format:
  json_object`), which is validated in `generator.py` before being shown to
  the user — malformed output is retried automatically.
- Every question ships with an **explanation** field that references the
  grounding fact, so users (and reviewers) can verify accuracy directly
  against the "Ground Truth" panel.
- A random "variety seed" is included in the prompt on every request so
  regenerated quizzes are genuinely different, not repeats.

---

## 🛠 Troubleshooting

| Issue | Fix |
|---|---|
| `sqlite3` version error from ChromaDB | `pip install pysqlite3-binary`, then uncomment the shim at the top of `src/database.py` |
| "OPENAI_API_KEY is missing" warning | Make sure you copied `.env.example` to `.env` and added a real key |
| Quiz generation fails / bad JSON | The app auto-retries once; if it still fails, try again — this can happen with very obscure sports/difficulty combos with thin context |
| Web search returns nothing | The app gracefully falls back to ChromaDB-only context so quiz generation still works |
| Slow first run | The first run vectorizes the offline dataset once; subsequent runs are instant (cached via `st.cache_resource`) |

---

## 📌 Notes on extending

- **Add more sports/facts**: edit `data/sports_facts.json` and delete the
  `chroma_db/` folder so it re-populates on next run.
- **Swap LLM provider**: `src/generator.py` isolates all LLM calls — swap the
  `openai` client for `google-genai` or another SDK without touching the UI.
- **Swap vector DB**: `src/database.py` isolates all ChromaDB calls behind
  `query_historic_facts()` / `setup_and_populate_db()`.
