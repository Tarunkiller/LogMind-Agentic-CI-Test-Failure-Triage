# LogMind — Agentic CI Test Failure Triage

An LLM agent that ingests CI test logs, triages failures,
and generates structured bug reports — without human intervention.
When uncertain, it flags for manual review rather than hallucinating.

**Built with:** Anthropic Claude API | Python | LLM Tool Use | Eval Harness

---

## How it works

```
RAW LOG → [parse_log] → [classify_failure] → confidence check
                                                    |
                              confidence >= 0.6     |  confidence < 0.6
                                    |               |        |
                          [generate_bug_report]     |  [flag_uncertainty]
                                    |               |        |
                              JSON report saved     |  manual_review
```

---

## Eval Results

| Metric               | Score    |
|----------------------|----------|
| Category accuracy    | 10/10    |
| Flag accuracy        | 10/10    |
| Hallucination-free   | 10/10    |
| Avg confidence       | 0.84     |

---

## Documented Failure Modes

- **Rule-based/No-API limitations:** In purely rule-based fallback mode, `classify_failure` relies on regex and keyword matching. It handles all 10 standard categories flawlessly but may lack context awareness on more obscure or unstructured error outputs.
- **Vague assertions:** Extremely minimal logs like `assertion_vague.txt` are correctly flagged via fallback, but an LLM agent with credits provides deeper reasoning as to *why* the assertion occurred.

---

## Recovery Paths

- Confidence < 0.6 → `flag_uncertainty` fires, no report generated
- Both `file` and `error_type` null → `flag_uncertainty` fires
- Log under 5 lines → `flag_uncertainty` fires
- Missing stack trace → `flag_uncertainty` fires

---

## Why `flag_uncertainty` matters

A hallucinated root cause in a bug tracker is worse than no triage at all.
Engineers who trust the system will chase phantom bugs. LogMind's
uncertainty detection ensures that low-confidence results never silently
enter the bug filing pipeline — they get routed to human review instead.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Usage

```bash
# Triage a single log file
python main.py --log data/sample_logs/assertion_error.txt

# Run the full eval harness (10 labeled logs)
python main.py --eval

# Triage all logs in a directory
python main.py --dir data/sample_logs/
```

---

## Project Structure

```
logmind/
├── main.py              # CLI entry point (--log, --eval, --dir flags)
├── agent.py             # Core agent loop with Anthropic tool-use API
├── tools.py             # 4 tool implementations (pure functions)
├── eval.py              # Evaluation harness — scores agent accuracy
├── data/
│   ├── sample_logs/     # 10 realistic CI log .txt files
│   └── eval_labels.json # Ground truth labels for eval scoring
├── outputs/
│   └── reports/         # Generated bug reports as JSON files
├── .env                 # ANTHROPIC_API_KEY (never commit this)
├── README.md
└── requirements.txt     # anthropic, python-dotenv
```

---

## Author

Padamati Tarun Krishna | NVIDIA Application 2026
