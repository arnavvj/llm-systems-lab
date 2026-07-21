"""
Project 06

Inference Engine

Agenda:
1. Wrap inference into an engine.
2. Reuse generate() and generate_stream().
3. Add batched generation.
"""

from engine import InferenceEngine


# (1) Config
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


# (2) Create Engine
engine = InferenceEngine(
    model_name=MODEL_NAME,
)


# (3) Generate
print("\n\n" + "=" * 60)
print("\nGENERATE...\n")
response = engine.generate("Why is the sky blue?")
print(response)


# (4) Stream
print("\n\n" + "=" * 60)
print("\nSTREAM...\n")

for token in engine.stream_generate("Why is the sky blue?"):
    print(token, end="", flush=True)


# (5) Batch Generate
print("\n\n" + "=" * 60)
print("\nBATCH GENERATE...\n")

prompts = [     # <---- This is a batch of input prompts
    "Why is the sky blue?",
    "Explain transformers in one sentence.",
    "What is Kubernetes?",
]
responses = engine.batch_generate(prompts)  # <---- NOTE: This is a mock example. Not real batching.

for i, response in enumerate(responses, start=1):
    print(f"\nPrompt {i}:")
    print(response)


# (6) Benchmark
print("\n\n" + "=" * 60)
print("\nBENCHMARK...\n")

engine.benchmark_streaming("Why is the sky blue?")

# The inference engine abstracts tokenization, generation,
# streaming, batching and benchmarking behind one interface.
# Production systems (vLLM, TGI, Fireworks, Baseten) expose a
# similar high-level abstraction while hiding scheduling,
# KV cache management and hardware-specific optimizations.