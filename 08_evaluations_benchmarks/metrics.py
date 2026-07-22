"""
Project 08

Evaluation Metrics
"""

import time


"""
Project 08

Evaluation Metrics
"""

import time


# NAIVE METRICS
def exact_match(prediction, reference):
    """
    1 if prediction exactly matches reference, else 0.
    """
    return int(prediction.strip() == reference.strip())


def token_overlap(prediction, reference):
    """
    Simple Jaccard similarity between token sets.
    """
    prediction = set(prediction.lower().split())
    reference = set(reference.lower().split())

    intersection = prediction & reference
    union = prediction | reference

    if len(union) == 0:
        return 1.0

    return len(intersection) / len(union)


def average(values):
    """Return the arithmetic mean of a list of values."""
    if len(values) == 0:
        return 0
    return sum(values) / len(values)


def tokens_per_second(
    generated_tokens,
    generation_time,
):
    """Compute generation throughput in tokens per second."""
    if generation_time == 0:
        return 0

    return generated_tokens / generation_time