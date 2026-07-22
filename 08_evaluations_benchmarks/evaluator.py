"""
Project 08

Evaluator
"""

import time

from metrics import (
    exact_match,
    token_overlap,
    average,
    tokens_per_second,
)


class Evaluator:

    def __init__(self, engine):
        self.engine = engine


    def evaluate(self, dataset):

        exact_matches = []
        overlaps = []
        latencies = []
        tps = []

        print("\nEvaluation\n")
        for sample in dataset:
            
            prompt = sample["prompt"]
            reference = sample["reference"]

            start = time.perf_counter()
            prediction = self.engine.generate(prompt)
            end = time.perf_counter()
            latency = end - start
            
            latencies.append(latency)   # <---- Latency

            exact_matches.append(       # <---- exact match bool
                exact_match(prediction, reference)
            )

            overlaps.append(    # <---- IoU
                token_overlap(prediction, reference)
            )

            generated_tokens = len(prediction.split())  # <---- num tokens and TPS
            tps.append(
                tokens_per_second(generated_tokens, latency)
            )

            print(f"Prompt     : {prompt}")
            print(f"Prediction : {prediction}")
            print(f"Reference  : {reference}")
            print()

        return {
            "samples": len(dataset),
            "exact_match": average(exact_matches),
            "token_overlap": average(overlaps),
            "latency": average(latencies),
            "tokens_per_second": average(tps),
        }


    def print_report(
        self,
        results,
    ):

        print("\n" + "=" * 60)
        print("\nEvaluation Report\n")

        print(f"Samples           : {results['samples']}")
        print(f"Exact Match       : {results['exact_match']:.2f}")
        print(f"Token Overlap     : {results['token_overlap']:.2f}")
        print(f"Average Latency   : {results['latency']:.3f}s")
        print(f"Tokens / Second   : {results['tokens_per_second']:.2f}")