# Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

```bash
python main.py
```

No helper files needed — everything fits in one script.

---

# Goal

Understand how speculative decoding speeds up autoregressive generation by
using a small draft model to propose tokens and a larger target model to
verify them. Focus is on the algorithm, not reproducing DeepMind's exact
implementation.

---

# Concepts

- **Draft model** — small, cheap model that proposes several tokens ahead
- **Target model** — the model whose output we want; verifies the draft
- **Verification** — one target forward pass scores every drafted position at once
- **Accept / Reject** — a drafted token is kept if it matches the target's own greedy prediction; the first mismatch discards the rest
- **Rollback** — on rejection, use the target's own prediction at that position instead
- **Exact output guarantee** — accepted tokens are only ever tokens the target would have produced anyway, so the result matches plain greedy decoding, just in fewer target forward passes

---

# How It Works

```text
Prompt
   │
   ▼
Small Draft Model
   │
   ▼
Propose 4 Tokens
   │
   ▼
Large Target Model
   │
   ▼
Verify
 ┌──────────────┐
 │ Accept?      │
 └──────────────┘
   │        │
 Yes       No
   │        │
   ▼        ▼
Keep     Rollback
```

Not the exact algorithm from the paper (which uses rejection sampling to
preserve an arbitrary sampling distribution) — since everything here is
greedy: draft proposes `k` tokens, target verifies all of them in one
forward pass, accept the matching prefix, and fall back to the target's
own prediction at the first mismatch.

---

# Project Structure

```text
09_speculative_decoding/
│
├── main.py         # Config, models, greedy baseline, draft/verify/accept-reject loop
├── README.md
└── requirements.txt
```

`main.py` is organized into 8 numbered sections: config; load draft +
target models; `greedy_generate` (baseline); `draft_tokens`;
`verify_tokens` (verify + accept/reject); `speculative_generate` (the
full loop, compared against the baseline in `main()`).

---

# Models

- **Draft:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Target:** `HuggingFaceTB/SmolLM2-360M-Instruct`

Same family, same tokenizer — required for comparing predictions
position-for-position. Both are the Instruct variant, prompted through
the chat template — see Notes for why that match matters.

---

# How to Run

```bash
python main.py
```

Runs the baseline and speculative loop on the same prompt and reports
whether they match, plus timing/acceptance:

```text
Outputs match      : True
Baseline time      : 2.008s
Speculative time   : 2.099s
Speedup            : 0.96x
Rounds             : 21
Accepted / Drafted : 43/84 (51% acceptance)
```

Speedup is noisy at this model scale — acceptance rate is the more
stable signal.

---

# Notes

**Draft/target style matching matters more than raw capability.** A base
(non-instruct) 135M draft against the instruct 360M target gave ~0%
acceptance on factual prompts — base rambles, instruct answers directly,
so their greedy choices diverge immediately. Swapping to an Instruct
135M draft (same fine-tuning recipe, just smaller) got 51% acceptance on
the same kind of prompt.

**"Exact output" holds in ideal arithmetic, not always in floating
point.** Verifying `k` draft tokens computes an earlier position's logits
as a byproduct of one longer batched forward pass instead of a standalone
pass. Causal masking makes these mathematically equivalent but not always
bit-identical — matmul reduction order can differ by shape, producing
logit differences of a few tenths. Invisible almost everywhere, but at a
genuine near-tie it can flip argmax and diverge from the baseline. Not
run-to-run randomness (`torch.use_deterministic_algorithms(True)` doesn't
fix it) — an inherent floating-point property real inference engines
accept too.

**Benchmarks need a warm-up pass.** A model's first-ever forward pass
pays one-time cost (kernel compilation, allocator first-touch). Timing
the baseline before the speculative loop without warming up first makes
the baseline look artificially slow. `main()` runs one throwaway forward
pass per model before either timer starts.

---

# What I Learned

- Implemented draft → verify → accept/reject → rollback end to end.
- Causal masking is why one forward pass over a longer sequence can
  substitute for several shorter ones (same reason teacher forcing
  works in training).
- The "exact output" guarantee is an ideal-arithmetic claim — floating
  point can break it in practice at genuine near-ties, without that
  being a logic bug.
- Acceptance rate depends on draft/target style matching as much as
  raw capability.
- Warm up models before timing them, or first-call overhead skews the
  comparison.
