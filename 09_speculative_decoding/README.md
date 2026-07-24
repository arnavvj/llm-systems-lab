# Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Run it

```bash
python main.py
```

No helper files needed — everything fits comfortably in one script.

---

# Goal

Understand how speculative decoding speeds up autoregressive generation by
using a small draft model to propose tokens and a larger target model to
verify them.

The focus is on understanding the algorithm, not reproducing DeepMind's
exact implementation.

---

# Concepts

- **Draft model** — small, cheap model that proposes several tokens ahead
- **Target model** — the model whose output we actually want; verifies the draft
- **Verification** — one target forward pass scores every drafted position at once
- **Token acceptance** — a drafted token that matches the target's own greedy prediction is kept for free
- **Token rejection** — the first mismatch discards the rest of the draft
- **Rollback** — on rejection, fall back to the target's own prediction at that position instead of the draft's
- **Throughput** — tokens produced per second (this is what speculative decoding improves)
- **Exact output guarantee** — under greedy decoding, accepted tokens are only ever tokens the target model would have produced anyway, so the final text is identical to plain greedy decoding on the target model alone — just computed in fewer target forward passes

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

We don't implement the mathematically exact speculative decoding algorithm
from the paper (which uses rejection sampling to preserve an arbitrary
sampling distribution). Since everything here is greedy instead:

1. Draft proposes `k` greedy tokens.
2. Target recomputes those positions in a single forward pass.
3. Compare token-by-token.
4. Accept the matching prefix.
5. Reject at the first mismatch and continue with the target model's own
   prediction at that position.

Because both models decode greedily, "accept" only ever means "the target
would have produced this anyway" — so the output is exact by construction
(see the floating-point caveat in Notes), not just approximately close.

---

# Project Structure

```text
09_speculative_decoding/
│
├── main.py         # Config, models, greedy baseline, draft/verify/accept-reject loop
├── README.md
└── requirements.txt
```

### main.py

Organized into 8 sections, in order:

1. **Config** — model names, draft width `k`, device selection
2. / 3. **Load draft + target models** — both `SmolLM2-*-Instruct`, sharing a tokenizer
4. **`greedy_generate`** — plain one-token-per-forward-pass baseline
5. **`draft_tokens`** — draft model proposes up to `k` tokens ahead
6. / 7. **`verify_tokens`** — one target forward pass verifies all `k` positions at once; accepts the matching prefix, falls back to the target's own prediction at the first mismatch
8. **`speculative_generate`** — loops draft → verify → accept/reject until `max_new_tokens` or EOS; `main()` runs it against the plain baseline and reports the comparison

---

# Models

- **Draft:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Target:** `HuggingFaceTB/SmolLM2-360M-Instruct`

Both are part of the same SmolLM2 family and share a tokenizer/vocabulary —
that's what makes it valid to feed one token sequence to either model and
compare their predictions position-for-position.

Both are also the **Instruct** variant, and prompts go through the chat
template for both. This is deliberate, not incidental: draft and target
need to be fine-tuned the same way, not just share an architecture, for
acceptance rate to mean anything — see Notes.

---

# How to Run

```bash
python main.py
```

Runs the target model's plain greedy baseline and the speculative loop on
the same prompt, and reports whether their outputs match plus a timing/
acceptance-rate comparison:

```text
Prompt             : Do you know what's the 5th and the sixth planet in the solar system?
Baseline (target)  : ...
Speculative        : ...
Outputs match      : True
------------------------------------------------------------
Baseline time      : 2.008s
Speculative time   : 2.099s
Speedup            : 0.96x
Rounds             : 21
Accepted / Drafted : 43/84 (51% acceptance)
```

Speedup is noisy run-to-run at this model scale (a couple hundred million
parameters, single prompt) — acceptance rate is the more stable signal.

---

# Notes

### Draft/target style matching matters more than raw capability

An earlier version of this project paired the *base* (non-instruct) 135M
model with the instruct 360M target. It still worked correctly, but
acceptance rate was often 0% on short factual prompts — a base completion
model and an instruct-tuned model have different "personalities" (one
rambles/continues descriptively, the other answers directly), so their
greedy choices diverge immediately. Swapping in the Instruct 135M draft
(same fine-tuning recipe as the target, just smaller) turned that into a
51% acceptance rate on the same kind of prompt. This is standard practice
in production speculative decoding, not just a demo convenience.

### "Exact output" holds in ideal arithmetic, not always in floating point

Verifying `k` draft tokens computes the target's logits for an earlier
position as a byproduct of one *longer* batched forward pass, instead of a
standalone forward pass over just that prefix. Causal masking makes these
two computations mathematically equivalent, but not always bit-identical —
matmul reduction order can differ across different sequence-length shapes,
producing logit differences of a few tenths. That's invisible almost
everywhere, but at a genuine near-tie (two tokens within ~0.2 logits of
each other) it can flip which one wins argmax, and greedy decoding then
commits to a different continuation than the plain baseline would have.
This isn't run-to-run randomness — `torch.use_deterministic_algorithms(True)`
doesn't fix it — it's an inherent property of floating point across
different-shaped computations, and real inference engines (vLLM included)
accept the same caveat.

### Benchmarks need a warm-up pass

The very first forward pass either model ever runs pays extra one-time
cost (kernel compilation, allocator first-touch). Timing the baseline
before the speculative loop without a warm-up call first makes the
baseline look artificially slow and inflates the apparent speedup — on a
workload this small, that fixed cost can dominate the whole measurement.
`main()` runs one throwaway forward pass on each model before starting
either timer.

---

# What I Learned

- Implemented the draft → verify → accept/reject loop end to end: greedy
  drafting, single-pass batched verification, prefix acceptance, and
  rollback to the target's own prediction on mismatch.
- Understood *why* one forward pass over a longer sequence can substitute
  for several shorter ones — causal masking means a transformer computes
  logits for every position in parallel, and each one is only a function
  of the tokens at or before it (the same property that makes teacher
  forcing work during training).
- Learned that the "exact output" guarantee is a claim about ideal
  arithmetic, and floating-point non-associativity across differently
  shaped computations can occasionally break it in practice at genuine
  near-ties — without that being a bug in the accept/reject logic.
- Learned that acceptance rate depends on draft/target fine-tuning style
  matching as much as on raw model capability.
- Learned to warm up models before timing them — first-call overhead can
  dominate measurements on small workloads and produce misleading
  "speedups."
