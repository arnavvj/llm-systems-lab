# 03 - KV Cache

## Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

# Goal

Optimize autoregressive LLM inference by caching attention **Keys** and **Values** instead of recomputing the entire prompt at every decoding step.

---

# Concepts

- Autoregressive Inference
- Self-Attention
- Key-Value (KV) Cache
- `past_key_values`
- DynamicCache
- Latency & Throughput

---

# Architecture

```text
Without KV Cache

Prompt
  │
Forward (all tokens)
  │
Append Token
  │
Repeat (all tokens)


With KV Cache

Prompt
  │
Forward (once)
  │
Cache Keys & Values
  │
Append Token
  │
Forward (new token only)
  │
Repeat
```

---

# How to Run

```bash
python main.py
```

---

# Key Learnings

- Vanilla decoding recomputes the entire prompt on every generation step.
- KV Cache stores attention **Keys** and **Values** from previous tokens.
- After the first forward pass, only the newly generated token is processed.
- The cache grows by **one token** after every decoding step.
- Modern Hugging Face models expose the cache through `DynamicCache`.

---

# What This Project Covers

✅ Baseline autoregressive decoding

✅ KV Cache (`past_key_values`)

✅ DynamicCache inspection

✅ Cache growth visualization

✅ Latency benchmark

---

# Key Learnings from the Code

- `use_cache=True` enables KV caching.
- `outputs.past_key_values` stores cached attention states.
- `DynamicCache.layers[i]` contains cached Keys and Values for each transformer layer.
- KV tensor shape is `(batch, heads, seq_len, head_dim)`.
- Cached sequence length grows by one token per decoding step.
- Reusing cached attention states significantly reduces inference latency.

---

# Results

```text
Vanilla Generation : 0.939 s
KV Cached Generation : 0.311 s

Speedup : ~3.02x
```

---

# Next Project

**04 - Chat Templates**

Learn why instruction-tuned LLMs expect structured conversations and build prompts using `tokenizer.apply_chat_template()`.