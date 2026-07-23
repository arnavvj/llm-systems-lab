# Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Run the evaluation

```bash
python main.py
```

---

# Goal

Evaluate the inference engine built in Project 06/07 against a small benchmark dataset, combining correctness metrics (does the model answer correctly) with performance metrics (how fast does it answer).

This project demonstrates:

- Building a benchmark dataset of prompt/reference pairs
- Scoring generated text with naive and standard NLP metrics
- Measuring latency, time-to-first-token, and throughput per sample
- Aggregating per-sample scores into a single evaluation report

---

# Concepts

- Exact match & token overlap (Jaccard similarity)
- BLEU and ROUGE-L
- Time to first token (TTFT)
- Latency vs. throughput (tokens/second)
- KV-cache streaming generation

---

# Project Structure

```text
08_evaluations_benchmarks/
│
├── main.py         # Entry point: builds the dataset and runs the evaluation
├── engine.py        # Minimal KV-cached streaming inference engine
├── evaluator.py     # Runs the engine over a dataset, scores/times each sample, and holds the metric functions
├── README.md
└── requirements.txt
```

### engine.py

Wraps tokenization and generation. Exposes a single method,
`stream_generate_kv_cache`, which greedily decodes token by token while
reusing the KV cache so only the newly generated token is fed back into the
model on each step.

### evaluator.py

Metric functions plus the `Evaluator` that uses them:

- `exact_match` — 1 if prediction matches reference exactly, else 0
- `token_overlap` — Jaccard similarity between token sets
- `bleu_score` / `rouge_score` — standard NLP metrics via `sacrebleu` and `evaluate`
- `tokens_per_second` / `average` — throughput and mean helpers

`Evaluator` runs the engine once per sample, timing time-to-first-token and
total latency while collecting the generated tokens. The resulting prediction
text is scored against the reference, and per-sample results are averaged
into a final report.

---

# How to Run

```bash
python main.py
```

For each sample this prints the prompt, prediction, reference, and per-sample
scores, followed by an aggregate report:

```text
Samples           : 3
Exact Match       : 0.33
Token Overlap     : 0.50
BLEU              : 42.10
ROUGE-L           : 0.67
Average TTFT      : 0.045s
Average Latency   : 1.203s
Tokens / Second   : 18.42
```

---

# What I Learned

- Correctness metrics (exact match, overlap, BLEU, ROUGE) and performance
  metrics (TTFT, latency, tokens/second) measure different things and both
  matter for judging an inference engine.
- A single generation pass can produce both the text to score and the timing
  data to benchmark — no need to regenerate the same prompt twice.
- Naive metrics (exact match, overlap) are cheap sanity checks; BLEU/ROUGE
  give a more forgiving, standard measure of text similarity.
