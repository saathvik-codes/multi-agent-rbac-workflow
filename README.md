# Secure Multi-Agent Enterprise Workflow Engine with RBAC & Guardrails

A production-grade, highly secure **Multi-Agent Coordination Engine** (FastAPI/Python) that automates business workflows while enforcing strict Role-Based Access Control (RBAC) at the data layer and screening traffic using security guardrails.

## 🚀 Key Features

1. **Custom State Machine Orchestrator**:
   * Chaining sequential execution of specialized agents: `GatewayRouter` → `SecureRetrieval` → `Analyst` → `Guardrail`.
   * Passes state context between steps, tracking timing and performance logs for each agent.
2. **Metadata-Driven RBAC RAG**:
   * Enforces security clearance boundaries (`Executive`, `Manager`, `Employee`, `Public`).
   * Documents are indexed with required roles. RBAC queries construct strict SQL filtering clauses, ensuring that users with lower clearance levels never retrieve higher-clearance contexts, even if semantic similarity is 100%.
3. **Input Prompt Injection Guardrail**:
   * Combines fast regex heuristics with semantic classification to catch jailbreak attempts (e.g., "ignore previous instructions", "override system prompts") before passing queries to subsequent agents.
4. **Output Leakage & PII Protection**:
   * Scans draft replies using high-performance regex pattern matching to securely redact Emails, Phone numbers, and API tokens/passwords with `[REDACTED_...]` placeholders.
5. **Modern Developer Admin Console**:
   * Tailwind dark-mode UI with simulated role selector, live pipeline trace timelines, grounding file chunk uploader, and active console telemetry logs.

---

## 📁 Repository Structure

```text
multi_agent_rbac_workflow/
├── app/
│   ├── agents/
│   │   ├── base.py          # Shared Agent interface & metrics tracker
│   │   ├── router.py        # Gateway router (injection blocking)
│   │   ├── retrieval.py     # Secure retrieval (clearance checker)
│   │   ├── analyst.py       # Reasoning agent (context analyzer)
│   │   ├── guardrail.py     # Output guardrail (PII redactions)
│   │   └── workflow.py      # State machine orchestrator engine
│   ├── database/
│   │   └── db.py            # SQLite RBAC SQL filters & audit logger
│   ├── services/
│   │   ├── embedding.py     # Embedding generator (local Ollama + mock fallback)
│   │   └── llm.py           # LLM API generator (Gemini + local Ollama fallback)
│   ├── templates/
│   │   └── index.html       # Sleek admin dashboard
│   ├── config.py            # Pydantic Settings configuration
│   ├── main.py              # FastAPI endpoints & lifespan
│   └── models.py            # Pydantic schemas
├── tests/
│   ├── test_rbac.py         # DB-level RBAC context boundary tests
│   ├── test_guardrails.py   # Prompt injection & output redaction checks
│   └── test_workflow.py     # End-to-end FastAPI endpoint tests
├── .env.example
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Requirements
Ensure **Ollama** is running locally for local embeddings:
```bash
ollama pull nomic-embed-text
ollama pull llama3
```

### 2. Install Python Dependencies
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Credentials
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```

---

## 🧪 Testing

Execute the test suite validating secure RAG retrieval, PII blocks, and prompt injection gates:
```bash
python -m pytest
```

---

## 🖥️ Running the Server

Start the FastAPI application:
```bash
uvicorn app.main:app --reload --port 8000
```
Open **`http://127.0.0.1:8000/`** to access the dashboard:
* **Playground**: Select a Simulated Role (e.g. `Public` vs `Executive`), submit your prompt, and watch the execution graph trace agent actions step-by-step.
* **Grounding Chunk Uploader**: Upload chunks specifying required security clearance roles to verify data containment.
* **Live Console Output**: Watch real-time logs printed in a developer-friendly JetBrains Mono terminal.
