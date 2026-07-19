import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# (1) config
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
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


# (4) TIME ANALYSIS OF VANILLA GENERATION LOOP
print("\n\n" + "=" * 60)
print("\nTIME ANALYSIS OF VANILLA LOOP...")

start = time.perf_counter()

MAX_NEW_TOKENS = 20
with torch.inference_mode():

    for _ in range(MAX_NEW_TOKENS):

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        input_ids = torch.cat(
            [input_ids, next_token_id],
            dim=1,
        )

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (1, 1),
                    device=DEVICE,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )

        if next_token_id.item() == tokenizer.eos_token_id:
            break

end = time.perf_counter()
vanilla_ts = end-start
print("\nGenerated Text:\n")
print(tokenizer.decode(input_ids[0]))
print(f"\nGeneration Time: {end} - {start} = {vanilla_ts:.3f}s")
# Baseline: every decoding step recomputes the entire prompt.
# Project 03 replaces this repeated work with a KV cache.


# (5) INTRODUCTION TO KV CACHE
print("\n\n" + "=" * 60)
print("\nTIME ANALYSIS OF KV CACHED LOOP...")

# Reset
input_ids = inputs.input_ids.to(DEVICE)
attention_mask = inputs.attention_mask.to(DEVICE)

start = time.perf_counter()

MAX_NEW_TOKENS, past_key_values = 20, None
with torch.inference_mode():

    # First forward pass over the entire prompt
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    input_ids = torch.cat([input_ids, next_token_id], dim=1)
    past_key_values = outputs.past_key_values
    print("\nLayers:", len(past_key_values))

    # Cache KVs from the first pass. Analyzing first layer
    print("\nKV Cache before loop...")
    print("\tLayer 0: ")
    layer0 = past_key_values.layers[0]
    keys, values = layer0.keys, layer0.values
    print("\t\tKey Shape  :", keys.shape)
    print("\t\tValue Shape:", values.shape, "\n")
    
    for _ in range(MAX_NEW_TOKENS-1):

        outputs = model(
            input_ids=next_token_id,
            past_key_values=past_key_values,
            use_cache=True,
        )
        # Consistent caching
        past_key_values = outputs.past_key_values
        print(f"\t\tStep {_+1}: cache length = {past_key_values.layers[0].get_seq_length()}")

        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

print("\nKV cache grows by one token per decoding step.")

end = time.perf_counter()
kv_ts = end-start
print("\nGenerated Text:\n")
print(tokenizer.decode(input_ids[0]))
print(f"\nGeneration Time: {end} - {start} = {kv_ts:.3f}s")

print(
    f"\nSpeedup: {vanilla_ts/kv_ts:.2f}x"
)

# Final KV dim check:
print("\nKV Cache after loop...")
print("\tLayer 0: ")
layer0 = past_key_values.layers[0]
keys, values = layer0.keys, layer0.values
print("\t\tKey Shape  :", keys.shape)
print("\t\tValue Shape:", values.shape)
# (batch_size, num_attention_heads, sequence_length, head_dimension)
print("\nShape = (batch, heads, seq_len, head_dim)")