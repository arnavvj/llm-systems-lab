"""
Project 01

Minimal autoregressive (predicts future values in a seq from a weighted combo of its own past values) llm inference

Agenda: create the core generation loop behind model.generate() using only:
1. pytorch
2. huggingface transformer
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# config
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Load tokenizer + model
print(f"\nLOADING MODEL ON {DEVICE}..\n")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


# Tokenize
INPUT_PROMPT = "Why is the sky blue?"
inputs = tokenizer(
        INPUT_PROMPT,
        return_tensors = 'pt'   # <---- pytorch tensor otherwise just a list. Also expected by the model
    )

print("\n" + "=" * 60)
print("\n1.", inputs)
input_ids = inputs.input_ids.to(DEVICE)             # <---- must live on same device else error: Expected all tensors to be on same device.
attention_mask = inputs.attention_mask.to(DEVICE)

print("\n2. Input prompt", INPUT_PROMPT)

print("\n3. Input IDs", input_ids)

print("\n4. Input Shape", input_ids.shape)  # <---- torch.Size([1, 6]) i.e. (batch_size, sequence_length)
print("\n" + "=" * 60)