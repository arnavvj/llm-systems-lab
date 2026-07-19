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

# (1) config
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# (2) Load tokenizer + model
print("\n\n" + "=" * 60)
print(f"\nLOADING MODEL ON {DEVICE}..")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


# (3) Tokenize
# Decoder-only LLMs are autoregressive: given tokens [t1...tn], predict only the next token (tn+1).
# During training, causal masking hides future tokens so every position learns the same next-token prediction task used at inference.
print("\n\n" + "=" * 60)
print(f"\nTOKENIZER ON..")
INPUT_PROMPT = "Why is the sky blue?"
inputs = tokenizer(
        INPUT_PROMPT,
        return_tensors = 'pt'   # <---- pytorch tensor otherwise just a list. Also expected by the model
    )
print("\n1.", inputs)
input_ids = inputs.input_ids.to(DEVICE)             # <---- must live on same device else error: Expected all tensors to be on same device.
attention_mask = inputs.attention_mask.to(DEVICE)
print("\n2. Input prompt", INPUT_PROMPT)
print("\n3. Input IDs", input_ids)
print("\n4. Input Shape", input_ids.shape)  # <---- torch.Size([1, 6]) i.e. (batch_size, sequence_length)


# (4) Forward pass
# A single forward pass processes the entire prompt in parallel and produces logits for every token position.
# During inference, we only use the logits from the final position (logits[:, -1, :]) to generate the next token.
# 
# Causal Mask Example
# Prompt: "Why is the sky blue ?"
# Position 0 -> sees: Why                          -> predicts: is
# Position 1 -> sees: Why is                       -> predicts: the
# Position 2 -> sees: Why is the                   -> predicts: sky
# Position 3 -> sees: Why is the sky               -> predicts: blue
# Position 4 -> sees: Why is the sky blue          -> predicts: ?
# Position 5 -> sees: Why is the sky blue ?        -> predicts: Because
#
# Training computes all predictions in one forward pass (via causal masking).
# Inference only uses the last prediction, appends it, and repeats (autoregressive decoding).
print("\n\n" + "=" * 60)
print("\nFORWARD PASS...")
with torch.inference_mode():
    outputs = model(
        input_ids = input_ids,
        attention_mask = attention_mask,
    )
logits = outputs.logits     # <---- every produced token gets a "score". Not a probablity yet
print("\n1. Logits shape:", logits.shape)   # <---- (batch, seq_length, vocab) 3rd dim is vocab size
print("\n2. Last Token Logits Shape:", logits[:, -1, :].shape) # <---- Only interested in last token

next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
print("\n3. Next Token ID:", next_token_id.item())
print("\n4. Next Token decoded:", tokenizer.decode(next_token_id))


# (5) Loop for each new token: tokenize -> *forward -> get last logits -> pick best token -> append token -> repeat*
#                                          ^                                                                       |        
#                                          |-----------------------------------------------------------------------|
print("\n\n" + "=" * 60)
print("\nLOOP...\n")
MAX_NEW_TOKENS = 20
with torch.inference_mode():
    for step in range(MAX_NEW_TOKENS):
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )
        print(f"Step {step}: {input_ids}", end="")
        input_ids = torch.cat(
            [input_ids, next_token_id],
            dim = 1
        )
        print(f" -> Add {next_token_id} i.e. '{tokenizer.decode(next_token_id[0])}'; makes new seq length = {input_ids.shape}")
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (1,1),
                    device = DEVICE,
                    dtype = attention_mask.dtype,
                )
            ],
            dim = 1,
        )
        if next_token_id.item() == tokenizer.eos_token_id:  # circuit breaker
            print("\nEOS reached.")
            break
    
    print("\nGenerated Text: ", end="")
    print(
        tokenizer.decode(
            input_ids[0]
        ), end = "\n"
    )


# NOTE:
# Instruct models expect chat-formatted prompts (user/assistant turns). With a raw prompt,
# the highest-probability next token is often the EOS marker (<|im_end|>), so the loop
# exits immediately. Base LLMs do not enforce this conversational structure and generally
# continue autoregressive decoding until EOS or an external circuit breaker (e.g.,
# max_new_tokens, stop sequences, timeout).