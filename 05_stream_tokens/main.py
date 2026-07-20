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


# (2) Config
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# (3) Load tokenizer + model
print("\n\n" + "=" * 60)
print(f"\nLOADING MODEL ON {DEVICE}..")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


# (4) Chat Template
print("\n\n" + "=" * 60)
print("\nCHAT TEMPLATE...")

messages = [
    {
        "role": "user",
        "content": "Why is the sky blue?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
)

input_ids = inputs.input_ids.to(DEVICE)
attention_mask = inputs.attention_mask.to(DEVICE)
print(tokenizer.decode(input_ids[0]))


# (5) Generate
print("\n\n" + "=" * 60)
print("\nGENERATE...\n")

def generate(
    input_ids,
    attention_mask,
    max_new_tokens=20,
):
    initial_length = input_ids.shape[1]

    start = time.perf_counter()
    first_token_ts = None

    with torch.inference_mode():

        for step in range(max_new_tokens):

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if first_token_ts is None:
                first_token_ts = time.perf_counter()

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
                print("\nEOS reached.")
                break

    end = time.perf_counter()

    print("\nFinal Conversation:\n")
    print(tokenizer.decode(input_ids[0]))

    generated_ids = input_ids[0][initial_length:]

    print("\nAssistant Response:\n")
    print(tokenizer.decode(generated_ids))

    print(f"\nTTFT                 : {first_token_ts-start:.3f}s")
    print(f"Total Generation Time: {end-start:.3f}s")
    print(f"Final Context Length : {input_ids.shape[1]} tokens")

# demo 5 i.e. vanilla generate 
generate(
    input_ids,
    attention_mask,
)



# (6) Streaming Generator
def generate_stream(
    input_ids,
    attention_mask,
    max_new_tokens=40,
):

    first_token = True
    start = time.perf_counter()

    with torch.inference_mode():

        for step in range(max_new_tokens):

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            token = tokenizer.decode(next_token_id[0])

            if first_token:
                ttft = time.perf_counter() - start
                print(f"\n\nTTFT: {ttft:.3f}s\n")
                first_token = False

            yield {
                "step": step + 1,
                "token_id": next_token_id.item(),
                "token": token,
            }

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

    total = time.perf_counter() - start
    print(f"\n\nTotal Generation Time: {total:.3f}s")


# demo 6 i.e. consume stream generator
print("\n\n" + "=" * 60)
print("\nSTREAMING RESPONSE...\n")

for event in generate_stream(
    input_ids,
    attention_mask,
):
    print(
        event["token"],
        end="",     # <---- Comment and see how it prints
        flush=True,
    )

# generate() blocks until the entire response is produced before returning.
# generate_stream() "yields" one token at a time, enabling real-time streaming.
# This is the core primitive behind SSE/WebSockets used by inference servers (vLLM, Baseten, Fireworks, OpenAI).
# TTFT (Time To First Token) measures perceived responsiveness, while total generation time measures end-to-end latency.