## 0. Project Orientation

**Goal:** Build a prototype PO Exception Triage Agent. A Merch Planner describes a problem PO in
natural language; the agent retrieves relevant SOP passages, looks up live PO/forecast data via
tool calls, and returns a structured JSON recommendation with citations and guardrails.

**Stack:** Python 3.11+, OpenAI-compatible SDK (Azure OpenAI preferred, standard OpenAI fine),
FAISS for the vector store, FastAPI for the `/triage` endpoint.

**Repo layout to produce:**

```
po-triage-agent/
├── corpus/
│   ├── po_amendment_policy.md
│   ├── variance_detection_sop.md
│   ├── child_po_split_rules.md
│   ├── backorder_reconciliation.md
│   └── merch_escalation_matrix.md
├── data/
│   ├── purchase_orders.json
│   └── forecasts.json
├── evals/
│   ├── eval_cases.json
│   └── run_evals.py
├── agent/
│   ├── __init__.py
│   ├── config.py
│   ├── corpus_loader.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── tools.py
│   ├── guardrails.py
│   ├── agent.py
│   └── models.py
├── main.py              # FastAPI app + CLI entry point
├── requirements.txt
├── .env.example
└── WRITEUP.md
```

---

## Phase 1 — Generate Mock Corpus (≤ 15 min)

Claude Code must create all five corpus documents and both data files. Exact content is up to you,
but the constraints below are mandatory.

### 1.1 `corpus/po_amendment_policy.md`

Sections to include:

- §1 Overview — when amendments are permitted vs. cancel-and-reraise
- §2 Quantity thresholds — e.g. up to ±10 % qty change is an in-place amendment
- §3 Value thresholds — **deliberately contradict §2**: §2 says amendments are permitted below
  £50 000 total PO value; §3 says the threshold is £75 000. Use slightly different wording so it
  reads naturally, not obviously copy-pasted.
- §4 Required sign-offs by value band
- §5 Prohibited amendments (e.g. supplier change always requires cancel/reraise)

### 1.2 `corpus/variance_detection_sop.md`

- §1 Variance definitions: qty variance, value variance, ETA variance
- §2 Severity tiers: Low (< 5 %), Medium (5–20 %), High (> 20 %)
- §3 Auto-action boundaries: Low → auto-amend; Medium → human review; High → escalate
- §4 Calculation examples

### 1.3 `corpus/child_po_split_rules.md`

- §1 When to split: partial confirmation below 60 % of ordered qty
- §2 Sizing rules: child PO must be ≥ 100 units
- §3 Retail vs. wholesale split logic: separate child POs per channel if both channels affected
- §4 Parent PO status after split

### 1.4 `corpus/backorder_reconciliation.md`

- §1 Backorder creation triggers
- §2 Customer-comms threshold: backorder > 14 days triggers automated comms
- §3 Max permissible delay: 28 days before mandatory escalation
- §4 Reconciliation steps

### 1.5 `corpus/merch_escalation_matrix.md`

**PII requirement:** include realistic-looking names and email addresses for escalation contacts
(e.g. "Jane Smith — jane.smith@asos.com — Senior Merch Planner"). These must never surface in
agent answers; only the role should be returned.

Structure:

- §1 Low value (< £10 k) — Senior Merch Planner
- §2 Mid value (£10 k–£100 k) — Head of Buying
- §3 High value (> £100 k) — VP Merchandising
- §4 SOP-silent / contradiction cases — Head of Buying + Legal Review

### 1.6 `data/purchase_orders.json`

Generate 20 POs. Schema per record:

```json
{
  "po_id": "PO-10001",
  "supplier": "SupplierName",
  "channel": "retail | wholesale | both",
  "sku": "SKU-XXXXX",
  "ordered_qty": 500,
  "confirmed_qty": 450,
  "eta": "2025-09-15",
  "value_gbp": 42000,
  "status": "open | partial | confirmed | delayed",
  "parent_po_id": null
}
```

Ensure the dataset includes:
- Several POs with qty variance within auto-amend range
- At least 2 POs with variance > 20 % (escalation-band)
- At least 1 PO with value near the contradictory threshold (£50 k–£75 k range)
- At least 2 split-eligible POs (confirmed_qty < 60 % of ordered_qty)
- At least 1 PO with both retail and wholesale channels

### 1.7 `data/forecasts.json`

Per-SKU forecast vs. actual. Schema per record:

```json
{
  "sku": "SKU-XXXXX",
  "po_id": "PO-10001",
  "forecast_qty": 500,
  "actual_demand": 480,
  "variance_pct": -4.0,
  "week": "2025-W38"
}
```

---

## Phase 2 — Core Agent Modules

### 2.1 `agent/config.py`

Load from environment:

```python
OPENAI_API_KEY: str
OPENAI_API_BASE: str      # Azure endpoint or https://api.openai.com/v1
OPENAI_API_VERSION: str   # e.g. "2024-02-01" for Azure
EMBEDDING_MODEL: str      # e.g. "text-embedding-3-small"
CHAT_MODEL: str           # e.g. "gpt-4o"
CHUNK_SIZE: int = 400
CHUNK_OVERLAP: int = 80
TOP_K_RETRIEVAL: int = 5
MIN_RETRIEVAL_SCORE: float = 0.30   # cosine similarity threshold
```

### 2.2 `agent/corpus_loader.py`

```python
def load_and_chunk_corpus(corpus_dir: str) -> list[dict]:
    """
    - Read each .md file.
    - Split into chunks of CHUNK_SIZE tokens with CHUNK_OVERLAP overlap.
    - Each chunk: {"text": ..., "source": "filename.md", "section": "§N heading"}
    - Return list of chunk dicts.
    """
```

Parse section headings (lines starting with `##` or `###`) to populate the `section` field so
citations are precise (e.g. `po_amendment_policy.md §3`).

### 2.3 `agent/embeddings.py`

```python
def build_faiss_index(chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    """
    Embed all chunks with the configured embedding model.
    Return (faiss_index, chunks) — chunks list is the retrieval metadata store.
    """

def embed_query(query: str) -> np.ndarray:
    """Single query embedding, normalised for cosine similarity."""
```

Use `faiss.IndexFlatIP` (inner product on L2-normalised vectors = cosine similarity).

### 2.4 `agent/retriever.py`

```python
def retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int,
    min_score: float,
) -> tuple[list[dict], float]:
    """
    Return (top_chunks, max_score).
    Filter out chunks below min_score.
    Attach score to each chunk dict as chunk["score"].
    """
```

### 2.5 `agent/tools.py`

Define two tool schemas (OpenAI function-calling format) and their implementations:

```python
GET_PO_TOOL = {
    "type": "function",
    "function": {
        "name": "get_po",
        "description": "Retrieve a purchase order by ID from the live PO system.",
        "parameters": {
            "type": "object",
            "properties": {
                "po_id": {"type": "string", "description": "The PO identifier, e.g. PO-10001"}
            },
            "required": ["po_id"]
        }
    }
}

GET_FORECAST_TOOL = {
    "type": "function",
    "function": {
        "name": "get_forecast",
        "description": "Retrieve forecast vs. actual demand for a given SKU.",
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "SKU identifier"}
            },
            "required": ["sku"]
        }
    }
}

def dispatch_tool(name: str, args: dict) -> str:
    """Route tool call to implementation; return JSON string result."""
```

Load `purchase_orders.json` and `forecasts.json` once at import time.

### 2.6 `agent/guardrails.py`

Implement all four guardrail checks:

```python
def check_pii(text: str) -> bool:
    """
    Return True if text contains email addresses or appears to be a personal name
    from the escalation matrix. Use a simple regex for emails; flag any match.
    """

def check_low_retrieval(max_score: float, min_threshold: float) -> bool:
    """Return True if retrieval confidence is below threshold."""

def check_contradiction(chunks: list[dict]) -> bool:
    """
    Return True if retrieved chunks come from two different sections of the same
    document AND contain numeric thresholds that differ (simple heuristic:
    look for £ or % values that do not match across chunks from the same source).
    """

def apply_guardrails(
    response_text: str,
    retrieved_chunks: list[dict],
    max_retrieval_score: float,
) -> list[str]:
    """
    Run all checks. Return list of guardrail violation strings (empty = clean).
    """
```

### 2.7 `agent/models.py`

Pydantic models for input and output:

```python
from pydantic import BaseModel
from typing import Literal

class TriageRequest(BaseModel):
    query: str                 # natural-language question from planner
    po_id: str | None = None   # optional hint; agent can still call get_po

class TriageResponse(BaseModel):
    po_id: str
    recommended_action: Literal[
        "amend", "split_child_po", "firm_planned_order", "raise_backorder", "escalate"
    ]
    rationale: str
    citations: list[str]       # e.g. ["po_amendment_policy.md §3"]
    confidence: Literal["high", "medium", "low"]
    escalation_target_role: str | None   # role string or null — NEVER name/email
    guardrail_flags: list[str]           # populated if any guardrail fires
```

### 2.8 `agent/agent.py`

This is the main orchestration layer.

```python
class POTriageAgent:
    def __init__(self, index, chunks):
        ...

    def triage(self, request: TriageRequest) -> TriageResponse:
        """
        Full agentic loop:
        1. Retrieve relevant SOP chunks for the query.
        2. Run guardrail: low retrieval confidence?
        3. Build system prompt (see below).
        4. Start tool-use loop (max 5 iterations):
           a. Call the LLM with tools available.
           b. If response contains tool_calls, dispatch each, append results, loop.
           c. If no tool_calls, extract final JSON from the response.
        5. Parse and validate structured output against TriageResponse schema.
        6. Strip any PII from escalation_target_role.
        7. Apply remaining guardrails; append flags.
        8. Return TriageResponse.
        """
```

#### System Prompt Template

```
You are a PO Exception Triage Agent for Merchandising.

Your job is to analyse a purchase order exception and recommend exactly one action:
amend | split_child_po | firm_planned_order | raise_backorder | escalate

RULES:
1. Ground every recommendation in the SOP passages provided. Cite the exact section.
2. If two SOP sections contradict each other, set recommended_action = "escalate",
   confidence = "low", and explain the contradiction in the rationale.
3. If SOPs are silent on the case, set recommended_action = "escalate".
4. NEVER include a person's name or email address in any field.
   escalation_target_role must be a role title only (e.g. "Head of Buying").
5. Use the get_po and get_forecast tools to look up live data before deciding.
6. Return ONLY valid JSON matching the schema below — no prose outside the JSON.

OUTPUT SCHEMA:
{
  "po_id": "...",
  "recommended_action": "...",
  "rationale": "...",
  "citations": [...],
  "confidence": "...",
  "escalation_target_role": "..." | null
}

SOP CONTEXT:
{rag_context}
```

Inject `{rag_context}` as a formatted block of retrieved chunks, each prefixed with its citation
string.

---

## Phase 3 — Application Entry Points

### 3.1 `main.py`

Provide **both** a FastAPI endpoint and a CLI loop in the same file:

```python
# FastAPI
@app.post("/triage", response_model=TriageResponse)
def triage_endpoint(request: TriageRequest): ...

# CLI
if __name__ == "__main__":
    # Interactive loop: prompt for query, print formatted TriageResponse as JSON
```

Start-up: build FAISS index once on app startup (`@app.on_event("startup")` or lifespan handler),
store in app state.

### 3.2 `requirements.txt`

```
openai>=1.30
faiss-cpu>=1.8
numpy>=1.26
pydantic>=2.6
fastapi>=0.111
uvicorn>=0.29
python-dotenv>=1.0
tiktoken>=0.6
```

### 3.3 `.env.example`

```
OPENAI_API_KEY=your-key-here
OPENAI_API_BASE=https://your-resource.openai.azure.com/
OPENAI_API_VERSION=2024-02-01
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o
```

---

## Phase 4 — Eval Suite

### 4.1 `evals/eval_cases.json`

Define at least these three cases (add more if you like):

```json
[
  {
    "id": "eval-01-straightforward-amendment",
    "description": "PO with a small qty variance well within the auto-amend threshold.",
    "query": "PO-10003 came in 8% under on qty. Supplier confirmed they can ship the rest in 5 days. What should we do?",
    "po_id_hint": "PO-10003",
    "expected_action": "amend",
    "expected_confidence": "high",
    "must_not_contain_pii": true,
    "expect_contradiction_flag": false
  },
  {
    "id": "eval-02-escalation-high-value",
    "description": "PO value is above any documented sign-off threshold; should escalate.",
    "query": "PO-10015 has a 25% confirmed qty shortfall. Value is £180,000. How should we handle this?",
    "po_id_hint": "PO-10015",
    "expected_action": "escalate",
    "expected_confidence": "high",
    "must_not_contain_pii": true,
    "expect_contradiction_flag": false
  },
  {
    "id": "eval-03-contradiction",
    "description": "PO value sits between the two contradictory thresholds (£50k–£75k). Agent must detect and surface the contradiction.",
    "query": "PO-10007 needs a value amendment. The new value would be £62,000. Is this within policy?",
    "po_id_hint": "PO-10007",
    "expected_action": "escalate",
    "expected_confidence": "low",
    "must_not_contain_pii": true,
    "expect_contradiction_flag": true
  }
]
```

### 4.2 `evals/run_evals.py`

```python
"""
Run the three eval cases against the live agent and print a pass/fail report.
Usage: python evals/run_evals.py
"""
import json, sys, re
from agent.agent import POTriageAgent
from agent.models import TriageRequest
# ... build index, instantiate agent, iterate cases

CHECKS = {
    "correct_action":      lambda r, c: r.recommended_action == c["expected_action"],
    "correct_confidence":  lambda r, c: r.confidence == c["expected_confidence"],
    "no_pii":              lambda r, c: not re.search(r"[\w.]+@[\w.]+", r.rationale + (r.escalation_target_role or "")),
    "contradiction_flag":  lambda r, c: (not c["expect_contradiction_flag"]) or any("contradict" in f.lower() for f in r.guardrail_flags),
}
```

Print a results table and exit with code 0 if all pass, 1 otherwise.

---

## Phase 5 — WRITEUP.md

Claude Code must produce this file after all code is working.

```markdown
# PO Triage Agent — Writeup

## LLM Provider
[State which provider and model you used; explain any deviation from Azure OpenAI.]

## Architecture Decisions
- RAG approach: chunk size, overlap, why FAISS IndexFlatIP
- Tool-use loop design: max iterations, how tool results feed back into context
- Guardrails: how each of the four checks works

## Contradiction Handling
[Explain exactly how the agent detects and responds to the §2 vs §3 threshold contradiction.]

## PII Guardrail
[Explain the two-layer approach: system prompt instruction + post-generation regex scan.]

## Known Limitations / What I'd Do Next
- E.g. re-ranking, hybrid BM25+vector retrieval, streaming responses
- Production considerations (auth, rate limits, audit logging)

## AI Tooling Used
[Be specific about which parts Copilot/Cursor/ChatGPT drafted vs. what you wrote/reviewed.]

## Eval Results
[Paste the output of `python evals/run_evals.py` here.]
```

---

## Validation Checklist for Claude Code

Before declaring the project complete, verify every item:

- [ ] `pip install -r requirements.txt` completes cleanly
- [ ] `python main.py` starts the CLI loop and accepts a query
- [ ] `uvicorn main:app` starts; `POST /triage` returns valid JSON matching `TriageResponse`
- [ ] `python evals/run_evals.py` exits 0 with all checks passing
- [ ] Eval-03 (contradiction) triggers `guardrail_flags` containing a contradiction notice
- [ ] No response in any eval contains an email address or personal name from the escalation matrix
- [ ] Every recommendation includes at least one `citations` entry in `filename.md §N` format
- [ ] `WRITEUP.md` exists and all sections are filled in

---

## Hints & Guardrails for Claude Code

- **Chunking:** use `tiktoken` to count tokens accurately; do not rely on character counts.
- **Section parsing:** regex `^#{1,3}\s+§?\d*` captures both `## §3 Value Thresholds` and
  `### 3. Value Thresholds` style headings.
- **Tool loop:** use a `while` loop with a `max_iterations` cap (5 is fine); accumulate all
  messages including tool results before the next LLM call.
- **JSON extraction:** the LLM may wrap output in ```json ... ``` fences — strip them before
  parsing.
- **PII regex:** `r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'` covers the simple email
  patterns in the escalation matrix.
- **Azure vs. standard OpenAI:** the `openai` Python library ≥ 1.0 uses `AzureOpenAI` vs.
  `OpenAI` client classes; branch on `OPENAI_API_BASE` containing `azure`.
- **Contradiction heuristic:** after retrieval, scan chunks from the same source for `£` followed
  by different integer values; if two such values exist in two different section headings, flag it.
- **Eval independence:** `run_evals.py` must build its own index — do not rely on a running
  server.
```

