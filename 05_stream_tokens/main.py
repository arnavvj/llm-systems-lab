"""
Project 05

Streaming Inference

Agenda:
1. Learn Python generators.
2. Stream one token at a time.
3. Measure TTFT.
"""

"""
Project 05

Streaming Inference

Agenda:
1. Understand Python generators.
2. Stream LLM tokens one at a time using yield.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# (1) Generator Demo
print("\n\n" + "=" * 60)
print("\nGENERATOR DEMO...")

def count():
    print("Start")
    yield 1
    print("Resume")
    yield 2
    print("Resume")
    yield 3
    print("Done")

for x in count():
    print("Received:", x)