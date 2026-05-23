# Email Triage Agent

A LangGraph-based agent that reads your Gmail inbox, classifies messages, drafts replies, and acts with **tiered autonomy** — auto-sending low-risk replies while queueing high-stakes ones for human review.

Built to demonstrate the production patterns that distinguish real AI engineering from tutorial-grade demos: structured outputs, version-controlled prompts with an eval harness, hybrid rule/LLM safety logic, full audit logs, cost tracking, and least-privilege OAuth.

---

## Demo

```
$ streamlit run src/app.py
```

The UI gives you a tabbed view of pending drafts, processed inbox, full audit trail, cost stats, and a settings panel where you can refresh your writing-style profile from sent emails.

**Top bar:** Gmail query (e.g. `is:unread newer_than:7d`), batch size, "Process new" button.

**Auto-send toggle:** off by default — every draft queues for review. Flip it on after you trust the agent on low-stakes replies; high-stakes drafts continue to require approval.

---

## Architecture

```
                       ┌──────────────┐
                       │ fetch_emails │
                       │ (Gmail API)  │
                       └──────┬───────┘
                              ▼
                       ┌──────────────┐
                       │   classify   │  Haiku 4.5
                       │ (5 categories)│ structured output
                       └──────┬───────┘
                              ▼
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
       ┌────────┐       ┌──────────┐       ┌────────┐
       │archive │       │ label_   │       │ draft  │  Sonnet 4.6
       │ (spam) │       │   fyi    │       │        │ + style profile
       └───┬────┘       └────┬─────┘       └───┬────┘
           │                 │                 ▼
           │                 │          ┌──────────────┐
           │                 │          │ risk_assess  │  rules + LLM
           │                 │          │  (money,     │  tiebreaker
           │                 │          │   commits,   │
           │                 │          │   legal,...) │
           │                 │          └──────┬───────┘
           │                 │                 │
           │                 │       ┌─────────┼─────────┐
           │                 │       ▼         ▼         ▼
           │                 │   ┌────────┐ ┌────────┐ ┌────────┐
           │                 │   │ auto_  │ │approval│ │urgent  │
           │                 │   │ send   │ │ queue  │ │ notify │
           │                 │   └───┬────┘ └────┬───┘ └────┬───┘
           ▼                 ▼       ▼          ▼          ▼
                       ┌────────────────────────────────┐
                       │      audit_log (SQLite)        │
                       │  classification, reasoning,    │
                       │  draft, risk signals, tokens,  │
                       │  USD cost, latency             │
                       └────────────┬───────────────────┘
                                    ▼
                          ┌──────────────────┐
                          │  Streamlit UI    │
                          │  4 tabs + bar    │
                          └──────────────────┘
```

---

## Design decisions worth talking about

### 1. Structured output via tool_use, never JSON-from-prose
Every LLM call (classify, draft, risk tiebreaker, style extraction) uses Anthropic's `tool_use` with a JSON schema. Anthropic enforces the schema on the model side, so we never parse markdown fences or repair malformed JSON.

### 2. Hybrid risk assessment, not pure LLM judgment
`tools/risk.py` first runs deterministic signals — money terms, commitment language, legal/HR keywords, credentials, draft length. Any signal firing → high-risk, no LLM call needed. Only when rules are silent AND classification confidence is medium do we invoke an LLM tiebreaker. Three benefits: cheap (most drafts skip the LLM), auditable (we record which signal fired), and conservative (rules catch what LLMs miss, e.g. subtle commitments).

### 3. Confidence-aware routing
The classifier reports categorical confidence (`high`/`medium`/`low`) rather than a 0–1 score, because LLMs are not well-calibrated on numeric confidence. Low-confidence outputs route to human review even within auto-send-eligible categories — the classifier's own uncertainty is a safety mechanism.

### 4. Version-controlled prompts + eval harness
Prompts live as `.txt` files in `src/prompts/`. Every eval run hashes the prompt and pins the result, so "did this prompt change improve accuracy?" is answerable, not speculative. Run `python -m src.eval.run_evals` after any prompt edit.

### 5. Least-privilege OAuth that expands as features need it
Day 1 was `gmail.readonly` only. Day 4 added `gmail.modify` and `gmail.send` — the auth code auto-detects the scope change and forces re-consent. No upfront kitchen-sink permissions.

### 6. Proper email threading
Replies include `In-Reply-To`, `References`, and the Gmail `threadId`. Most tutorial agents skip this and the result lands as a new conversation in the recipient's inbox.

### 7. Writing-style profile from sent emails
`tools/style.py` pulls your recent sent emails, strips quoted-reply blocks and auto-responder junk, and asks Sonnet to extract a structured style profile (tone, sign-off, length, anchor phrases). The drafter loads it automatically. Drafts stop sounding like ChatGPT and start sounding like you.

### 8. Per-email cost & latency tracking
`usage_meter.py` is a thread-local accumulator: every API response's token counts are captured and rolled up per email. Stored in the audit log with USD cost computed from list prices. Stats tab projects annual cost at your current avg cost/email — talking point for "did you think about production?".

### 9. Idempotent state + audit trail as source of truth
The UI is a thin layer over `triage.db`. Every decision the agent made — classification, reasoning, risk signals, model, latency, tokens, cost, approval status — is in the audit log. You could swap Streamlit for FastAPI + React without touching the graph.

### 10. Defaults that protect the user
Auto-send is off by default. High-stakes is always queued regardless of risk signals. Failed auto-sends fall back to the approval queue instead of being lost.

---

## Setup

### Google Cloud Console
1. Create a project at https://console.cloud.google.com/
2. Enable the **Gmail API** (APIs & Services → Library)
3. OAuth consent screen: **External**, **Testing**, add your Gmail as a test user
4. Create credentials → **OAuth client ID** → **Desktop app** → download as `credentials.json` in project root

### Python
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

### Run
```bash
# UI
streamlit run src/app.py
# or use the launcher that handles PYTHONPATH for you
python run.py
```

First launch opens a browser for OAuth consent (you'll need to click through Google's "unverified app" warning since you haven't gone through their verification — that's normal for personal projects).

---

## Project structure

```
src/
├── app.py              Streamlit UI (5 tabs + top bar)
├── classifier.py       Haiku-based classification, tool_use
├── drafter.py          Sonnet-based draft generation, style-aware
├── usage_meter.py      Thread-local token + USD cost tracking
├── db.py               SQLite audit log + approval workflow
├── gmail_auth.py       OAuth with auto-detected scope changes
├── gmail_client.py     MIME-tree-aware message parsing
├── models.py           Pydantic models (Email, Classification, Draft, AuditRecord)
├── config.py           Constants, scopes, paths, mode flags
├── prompts/            Version-controlled prompts (.txt files)
│   ├── classify.txt
│   └── draft.txt
├── graph/              LangGraph state machine
│   ├── state.py        TriageState TypedDict
│   ├── nodes.py        Pure functions, one per node
│   ├── edges.py        Routing logic
│   └── builder.py      Graph assembly
├── tools/
│   ├── risk.py         Hybrid rule + LLM risk assessment
│   ├── gmail_actions.py Send (threaded), archive, label
│   └── style.py        Sent-email style profile extraction
└── eval/
    ├── golden_set.jsonl Hand-labeled email set (gitignored — personal)
    ├── label_helper.py  Interactive labeling CLI
    └── run_evals.py     Per-class precision/recall + confusion matrix

scripts/
├── day1_fetch.py       Smoke test: fetch + parse
└── day3_triage.py      CLI runner: process inbox without UI
```

---

## Eval results

Run on a personal hand-labeled set of N=30 emails:

| Metric | Value |
|---|---|
| Overall accuracy | _run `python -m src.eval.run_evals` to populate_ |
| Cost per email (avg) | _shown in Stats tab after processing_ |
| Latency per email (avg) | _shown in Stats tab after processing_ |

Eval runs are stored in `src/eval/results/` with model + prompt hash, so prompt iteration is measurable.

---

## What's interview-worthy

If asked to walk through this:

- **The risk-assessment design** — explain why pure LLM risk judgment is unreliable and how the rules-first approach is both cheaper and more defensible.
- **Cost tracking as a first-class concern** — token counts captured per API call via thread-local accumulator, persisted per email, projected to annual spend in the UI.
- **The eval harness** — version-controlled prompts hashed into result files, per-class F1 because false negatives on `needs_reply_high` are catastrophic.
- **The audit log as the source of truth** — UI is a view, not state. Swapping frontends would be trivial.
- **Confidence-aware routing** — the classifier's own uncertainty becomes part of the safety boundary.

---

## License

MIT
