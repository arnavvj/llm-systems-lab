"""
Project 08

Evaluator + Metrics
"""

import time

import evaluate
from sacrebleu.metrics import BLEU

_bleu = BLEU()
_rouge = evaluate.load("rouge")


def exact_match(prediction, reference):
    """1 if prediction exactly matches reference, else 0."""
    return int(prediction.strip() == reference.strip())


def token_overlap(prediction, reference):
    """Jaccard similarity between token sets."""
    prediction_tokens = set(prediction.lower().split())
    reference_tokens = set(reference.lower().split())

    union = prediction_tokens | reference_tokens
    if not union:
        return 1.0

    return len(prediction_tokens & reference_tokens) / len(union)


def bleu_score(prediction, reference):
    """BLEU score for a single prediction."""
    return _bleu.sentence_score(prediction, [reference]).score


def rouge_score(prediction, reference):
    """ROUGE-L score for a single prediction."""
    result = _rouge.compute(predictions=[prediction], references=[reference])
    return result["rougeL"]


def tokens_per_second(generated_tokens, generation_time):
    """Generation throughput in tokens per second."""
    return generated_tokens / generation_time if generation_time else 0


def average(values):
    """Arithmetic mean of a list of values, 0 if empty."""
    return sum(values) / len(values) if values else 0


class Evaluator:
    """Evaluate an inference engine on a benchmark dataset."""

    def __init__(self, engine):
        self.engine = engine

    def _run(self, prompt):
        """Generate a response, timing time-to-first-token and total latency."""

        start = time.perf_counter()
        first_token_time = None
        tokens = []

        for token in self.engine.stream_generate_kv_cache(prompt):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            tokens.append(token)

        end = time.perf_counter()
        latency = end - start

        return {
            "prediction": "".join(tokens),
            "ttft": first_token_time - start,
            "latency": latency,
            "tokens_per_second": tokens_per_second(len(tokens), latency),
        }

    def evaluate(self, dataset):
        """Run the benchmark and compute evaluation metrics."""

        scores = {key: [] for key in (
            "exact_match", "token_overlap", "bleu", "rouge",
            "ttft", "latency", "tokens_per_second",
        )}

        print("\nEvaluation\n")
        for sample in dataset:

            prompt, reference = sample["prompt"], sample["reference"]
            run = self._run(prompt)
            prediction = run["prediction"]

            scores["exact_match"].append(exact_match(prediction, reference))
            scores["token_overlap"].append(token_overlap(prediction, reference))
            scores["bleu"].append(bleu_score(prediction, reference))
            scores["rouge"].append(rouge_score(prediction, reference))
            scores["ttft"].append(run["ttft"])
            scores["latency"].append(run["latency"])
            scores["tokens_per_second"].append(run["tokens_per_second"])

            print(f"Prompt          : {prompt}")
            print(f"Prediction      : {prediction}")
            print(f"Reference       : {reference}")
            print(f"Exact Match     : {scores['exact_match'][-1]}")
            print(f"Token Overlap   : {scores['token_overlap'][-1]:.2f}")
            print(f"BLEU            : {scores['bleu'][-1]:.2f}")
            print(f"ROUGE-L         : {scores['rouge'][-1]:.2f}")
            print(f"TTFT            : {run['ttft']:.3f}s")
            print(f"Latency         : {run['latency']:.3f}s")
            print(f"Tokens / Second : {run['tokens_per_second']:.2f}")
            print("-" * 60)

        return {
            "samples": len(dataset),
            **{key: average(values) for key, values in scores.items()},
        }

    def print_report(self, results):
        """Print a formatted evaluation summary."""

        print("\n" + "=" * 60)
        print("\nEvaluation Report\n")

        print(f"Samples           : {results['samples']}")
        print(f"Exact Match       : {results['exact_match']:.2f}")
        print(f"Token Overlap     : {results['token_overlap']:.2f}")
        print(f"BLEU              : {results['bleu']:.2f}")
        print(f"ROUGE-L           : {results['rouge']:.2f}")
        print(f"Average TTFT      : {results['ttft']:.3f}s")
        print(f"Average Latency   : {results['latency']:.3f}s")
        print(f"Tokens / Second   : {results['tokens_per_second']:.2f}")
