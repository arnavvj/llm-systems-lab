# 05 - Streaming Tokens

## Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

# Goal

Stream LLM output **one token at a time** using Python generators (`yield`) instead of blocking until the full response is ready.

Compare blocking generation vs streaming generation and measure **TTFT (Time To First Token)** — the metric behind perceived responsiveness in chat UIs.

---

# Concepts

- Python Generators (`yield`)
- Lazy Evaluation & Suspended Execution
- Blocking vs Streaming Generation
- Time To First Token (TTFT)
- Total Generation Time
- Streaming Events (token-by-token payloads)

---

# Architecture

```text
Blocking Generation

Prompt -> Full Decoding Loop -> Complete Response -> Return (user waits the whole time)


Streaming Generation

Prompt
  │
Forward Pass
  │
Decode Next Token
  │
yield {step, token_id, token}   <-- consumer prints immediately
  │
Append Token
  │
Repeat until EOS / max_new_tokens
```

---

# How to Run

```bash
python main.py
```

---

# Key Learnings

- A generator function pauses at `yield` and resumes exactly where it left off on the next iteration.
- `generate()` blocks until the entire response is produced before returning anything.
- `generate_stream()` yields each token as soon as it's decoded, enabling real-time output.
- Each yielded event carries structured data (`step`, `token_id`, `token`) — the shape of a streaming API payload.
- `print(..., end="", flush=True)` renders tokens as a continuous stream in the terminal.
- **TTFT** measures how quickly the first token appears; **total generation time** measures end-to-end latency.
- This yield-based loop is the core primitive behind SSE/WebSocket streaming in inference servers (vLLM, OpenAI, Fireworks, Baseten).

---

# What This Project Covers

✅ Python generator demo (`yield`)

✅ Chat-templated prompt (from Project 04)

✅ Blocking generation baseline

✅ Streaming generation with `yield`

✅ TTFT measurement

✅ Total generation time measurement

---

# Example Output

```text
GENERATOR DEMO...
Start
Received: 1
Resume
Received: 2
Resume
Received: 3
Done


STREAMING RESPONSE...

TTFT: 0.180s

The sky appears blue because of a phenomenon called Rayleigh scattering...

Total Generation Time: 3.412s
```

> **Note:** With streaming, the user starts reading after TTFT (~0.2s) instead of waiting for the full generation time — the total work is the same, but the perceived latency is dramatically lower.

---

# Next Project

**06 - Inference Server**

Wrap the streaming generator in a FastAPI server and stream tokens to clients over HTTP (Server-Sent Events) — the same interface exposed by production inference APIs.
