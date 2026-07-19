# 02 - Sampling Algorithms

## Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

# Goal

Implement the core decoding algorithms behind modern LLM inference using only **PyTorch** and **Hugging Face Transformers**.

Starting from model logits, build greedy decoding, temperature scaling, multinomial sampling, Top-k, and Top-p (nucleus) sampling without using `model.generate()`.

---

# Concepts

- Logits & Softmax
- Probability Distribution
- Greedy Decoding
- Temperature Scaling
- Multinomial Sampling
- Top-k Sampling
- Top-p (Nucleus) Sampling

---

# Architecture

```text
Logits
    │
Temperature Scaling
    │
Softmax
    │
Probability Distribution
    │
    ├── Greedy (argmax)
    ├── Multinomial
    ├── Top-k
    └── Top-p
```

---

# How to Run

```bash
python main.py
```

---

# Key Learnings

- Logits are raw scores; softmax converts them into probabilities.
- Softmax preserves ordering, so `argmax(logits) == argmax(softmax(logits))`.
- Lower temperature (<1) sharpens the distribution; higher temperature (>1) increases randomness.
- `torch.multinomial()` performs weighted random sampling from the probability distribution.
- Top-k samples from a **fixed** set of the `k` most likely tokens.
- Top-p samples from an **adaptive** set whose cumulative probability reaches `p`.
- `torch.topk()`, `torch.cumsum()`, and `torch.gather()` are the core PyTorch primitives behind production decoding algorithms.

---

# What This Project Covers

✅ Greedy Decoding

✅ Temperature Scaling

✅ Multinomial Sampling

✅ Top-k Sampling

✅ Top-p (Nucleus) Sampling

---

# Next Project

**03 - KV Cache**

Build an efficient autoregressive inference loop by caching attention key/value states instead of recomputing the entire prompt on every decoding step.