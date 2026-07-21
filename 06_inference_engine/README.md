# 06 - Inference Engine

## Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

# Goal

Wrap everything built so far (chat-templated prompts, blocking generation, streaming, KV caching) behind a single reusable `InferenceEngine` class, and add naive batching and a built-in benchmark — the first step toward a real inference server.

---

# Concepts

- Engine Abstraction (tokenizer + model + generation behind one interface)
- Blocking Generation (`generate()`)
- Streaming Generation (`stream_generate()`)
- Streaming Generation with KV Cache (`stream_generate_kv_cache()`)
- Naive Sequential Batching (`batch_generate()`)
- TTFT & Total Generation Time Benchmarking

---

# Architecture

```text
InferenceEngine
  │
  ├── __init__(model_name)             -> load tokenizer + model, pick device (mps/cuda/cpu)
  ├── _prepare_inputs(prompt)          -> apply_chat_template() -> input_ids, attention_mask
  │
  ├── generate(prompt)                 -> full decode loop, returns complete string
  ├── stream_generate(prompt)          -> yields tokens one at a time, no cache
  ├── stream_generate_kv_cache(prompt) -> yields tokens one at a time, reuses past_key_values
  ├── batch_generate(prompts)          -> generate() looped over prompts sequentially (fake batching)
  └── benchmark(prompt)                -> compares stream_generate() vs stream_generate_kv_cache()
```

---

# How to Run

```bash
python main.py
```

---

# Key Learnings

- Wrapping tokenization, generation, streaming, and caching behind one class turns each prior project's script into a reusable method call.
- `generate()`, `stream_generate()`, and `stream_generate_kv_cache()` all share the same `_prepare_inputs()` prompt-building step — the engine's decode strategy is the only thing that changes.
- `batch_generate()` here just loops `generate()` over prompts one at a time — it's a real batch **API** but a fake batch **implementation**, since each prompt still gets its own full forward passes.
- Real batching (vLLM, TGI, Fireworks, Baseten) runs multiple prompts through the same forward pass using continuous batching — that's the subject of a later project.
- `benchmark()` reuses the streaming methods to measure both **TTFT** (time to first token) and **total generation time**, so the engine's own generators double as the instrumentation layer.
- The KV-cached path consistently wins on total generation time; TTFT is close either way since the first token still requires a full forward pass over the prompt.

---

# What This Project Covers

✅ `InferenceEngine` class wrapping tokenizer + model

✅ Blocking generation (`generate`)

✅ Streaming generation, with and without KV cache

✅ Naive sequential batch generation (`batch_generate`)

✅ Built-in benchmarking (`benchmark`)

---

# Example Output

```text
============================================================

GENERATE...

The sky is blue because of a phenomenon known as Rayleigh scattering. This occurs
when sunlight enters Earth's atmosphere and encounters tiny molecules of gases such
as nitrogen and oxygen. These molecules scatter the shorter wavelengths of light,
such as blue and violet, in all directions, while the longer wavelengths, like red
and orange, are


============================================================

STREAM...

The sky is blue because of a phenomenon known as Rayleigh scattering...


============================================================

BATCH GENERATE...

Prompt 1:
The sky is blue because of a phenomenon known as Rayleigh scattering...

Prompt 2:
A transformer is a device that changes the voltage of an alternating current (AC)
signal by using electromagnetic induction, allowing it to step up or step down the
voltage while maintaining the same current.

Prompt 3:
Kubernetes is a container orchestration platform that allows you to manage and
scale your applications across a cluster of servers...


============================================================

BENCHMARK...

PROMPT for benchmarking: Why is the sky blue?

1. Vanilla Streaming (No KV Cache):
TTFT                 : 0.024s
Total Generation Time: 1.546s

2. Optimized Streaming (KV Cache):
TTFT                 : 0.021s
Total Generation Time: 0.873s

Stats calculated in streaming mode!
```

> **Note:** Same prompt, same model — the KV-cached stream finishes in roughly half the time of the vanilla stream, purely by avoiding recomputation over already-seen tokens.

---

# Next Project

**07 - FastAPI Inference Server**

Wrap `InferenceEngine` in a FastAPI server and expose `generate` / `stream_generate` over HTTP (Server-Sent Events) — the same interface shape exposed by production inference APIs.
