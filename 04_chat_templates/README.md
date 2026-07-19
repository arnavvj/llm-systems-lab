# 04 - Chat Templates

## Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

# Goal

Understand why instruction-tuned LLMs expect **structured, role-based conversations** and build proper prompts using `tokenizer.apply_chat_template()`.

Compare how the same model behaves when given a raw prompt versus a chat-formatted prompt, for both single-turn and multi-turn conversations.

---

# Concepts

- Instruction Tuning
- Chat Templates
- Role-Based Messages (`user` / `assistant`)
- Special Tokens (`<|im_start|>`, `<|im_end|>`)
- `apply_chat_template()`
- Generation Prompt (`add_generation_prompt=True`)
- Multi-Turn Context

---

# Architecture

```text
Raw Prompt

"Why is the sky blue?" -> Tokenizer -> Forward Pass -> Predicts <|im_end|> (model thinks the message just ended)


Chat-Formatted Prompt

messages = [{role, content}, ...]
        │
apply_chat_template()
        │
<|im_start|>user ... <|im_end|> <|im_start|>assistant
        │
Forward Pass -> Greedy Decode -> Assistant Response
```

---

# How to Run

```bash
python main.py
```

---

# Key Learnings

- Instruction-tuned models are trained on **structured conversations**, not raw text.
- A raw prompt makes the model predict `<|im_end|>` because it looks like a completed user message.
- `apply_chat_template()` reproduces the exact prompt format used during instruction tuning.
- Special tokens like `<|im_start|>` and `<|im_end|>` mark message boundaries and roles.
- `add_generation_prompt=True` appends `<|im_start|>assistant` so the model knows it's its turn to respond.
- Multi-turn chat works by re-sending the **entire conversation history** as one growing token sequence.
- The context length grows with every turn — the model itself is stateless.

---

# What This Project Covers

✅ Raw prompt vs chat-formatted prompt comparison

✅ `apply_chat_template()` usage

✅ Generation prompt (`add_generation_prompt=True`)

✅ Single-turn chat generation

✅ Multi-turn chat generation

✅ Context length growth inspection

---

# Example Output

```text
RAW PROMPT...

Raw Prompt Prediction:
'<|im_end|>'


SINGLE-TURN CHAT...

Decoded Prompt:

<|im_start|>user
Why is the sky blue?<|im_end|>
<|im_start|>assistant

CHAT Prompt Prediction:
'The'
```

> **Note:** The same model that dead-ends on a raw prompt produces a proper answer once the prompt is wrapped in the chat template — the format is the difference, not the model.

---

# Next Project

**05 - Inference Server**

Combine everything so far — tokenization, chat templates, KV caching, and sampling — into a minimal inference server that serves chat completions.