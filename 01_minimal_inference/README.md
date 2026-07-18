# 01 - Minimal LLM Inference

## Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

# Goal

Build the core autoregressive generation loop behind `model.generate()` using only **PyTorch** and **Hugging Face Transformers**.

This project intentionally avoids high-level generation APIs to understand how decoder-only LLMs perform inference one token at a time.

---

# Concepts

- Tokenization
- Decoder-only Transformers
- Autoregressive Decoding
- Causal Masking
- Forward Pass
- Logits
- Greedy Decoding (`argmax`)
- EOS Tokens

---

# Architecture

```text
Prompt -> Tokenizer -> input_ids -> Transformer Forward Pass -> Logits -> logits[:, -1, :] -> Greedy Decode (argmax) -> Append Token -> Repeat
```

---

# How to Run

```bash
python main.py
```

---

# Example Output

```text
Prompt:
Why is the sky blue?

Input Shape:
torch.Size([1, 6])

Logits Shape:
torch.Size([1, 6, 49152])

Next Token:
<|im_end|>
```

> **Note:** This project uses an instruction-tuned model (`SmolLM2-360M-Instruct`). Since we're providing a raw prompt instead of a chat-formatted prompt, the model predicts the end-of-message token (`<|im_end|>`). We'll revisit proper chat templates when building our inference server later in the series.

---

# What I Learned

- LLMs operate on **token IDs**, not raw text.
- A single forward pass processes the **entire prompt in parallel**.
- Causal masking prevents each token from attending to future tokens.
- The model produces logits for **every token position**, but inference only uses the final position (`logits[:, -1, :]`) to predict the next token.
- Greedy decoding selects the highest-scoring token using `torch.argmax()`.
- Autoregressive generation repeatedly performs:
  1. Forward pass
  2. Select next token
  3. Append token
  4. Repeat

---

# Next Project

**02 - Sampling Algorithms**

Replace greedy decoding with production decoding strategies:

- Temperature Sampling
- Top-k Sampling
- Top-p (Nucleus) Sampling

We'll compare how different decoding strategies affect generation quality, diversity, and determinism.