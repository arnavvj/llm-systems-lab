"""
Project 04

Chat Templates

Agenda:
1. Compare raw prompts vs chat-formatted prompts.
2. Understand why instruct models expect role-based conversations.
3. Generate responses for both single-turn and multi-turn chats.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# (1) Config
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


# (3) Raw Prompt
print("\n\n" + "=" * 60)
print("\nRAW PROMPT...")

PROMPT = "Why is the sky blue?"

raw_inputs = tokenizer(PROMPT, return_tensors="pt")
raw_input_ids = raw_inputs.input_ids.to(DEVICE)
raw_attention_mask = raw_inputs.attention_mask.to(DEVICE)

print("\nDecoded Prompt:\n")
print(tokenizer.decode(raw_input_ids[0]))
print("\nRAW Prompt Tokens:", raw_input_ids.shape[1])

with torch.inference_mode():
    outputs = model(
        input_ids=raw_input_ids,
        attention_mask=raw_attention_mask,
    )
next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
print("\nRaw Prompt Prediction:")
print(repr(tokenizer.decode(next_token)))


# (4) Helper
def generate(chat_input_ids, chat_attention_mask, max_new_tokens=20):
    initial_length = chat_input_ids.shape[1]
    with torch.inference_mode():
        for step in range(max_new_tokens):
            
            outputs = model(
                input_ids=chat_input_ids,
                attention_mask=chat_attention_mask,
            )
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            print(
                f"Step {step+1:2}: "
                f"{next_token_id.item():5} "
                f"({repr(tokenizer.decode(next_token_id[0]))})"
            )

            chat_input_ids = torch.cat([chat_input_ids, next_token_id], dim=1)
            chat_attention_mask = torch.cat(
                [
                    chat_attention_mask,
                    torch.ones(
                        (1, 1),
                        device=DEVICE,
                        dtype=chat_attention_mask.dtype,
                    ),
                ],
                dim=1,
            )

            if next_token_id.item() == tokenizer.eos_token_id:
                print("\nEOS reached.")
                break

    print("\nFinal Conversation:\n")
    print(tokenizer.decode(chat_input_ids[0]))

    generated_ids = chat_input_ids[0][initial_length:]
    print("\nAssistant Response:")
    print(tokenizer.decode(generated_ids))
    print("\nInput Context Length:", initial_length, "tokens")
    print("Final Context Length:", chat_input_ids.shape[1], "tokens")
    print("Newly added as seen above:", chat_input_ids.shape[1] - initial_length, "tokens")


# (5) Single-turn Chat
print("\n\n" + "=" * 60)
print("\nSINGLE-TURN CHAT...")

messages = [
    {
        "role": "user",
        "content": PROMPT,
    }
]

chat_inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
    add_generation_prompt=True,
)

chat_input_ids = chat_inputs.input_ids.to(DEVICE)
chat_attention_mask = chat_inputs.attention_mask.to(DEVICE)

print("\nDecoded Prompt:\n")
print(tokenizer.decode(chat_input_ids[0]))
print("\nCHAT Prompt Tokens:", chat_input_ids.shape[1])

with torch.inference_mode():
    outputs = model(
        input_ids=chat_input_ids,
        attention_mask=chat_attention_mask,
    )
next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

print("\nCHAT Prompt Prediction:")
print(repr(tokenizer.decode(next_token)))
generate(chat_input_ids, chat_attention_mask)


# (6) Multi-turn Chat
print("\n\n" + "=" * 60)
print("\nMULTI-TURN CHAT...")

messages = [
    {
        "role": "user",
        "content": "Who discovered gravity?"
    },
    {
        "role": "assistant",
        "content": "Isaac Newton is widely credited with formulating the law of universal gravitation."
    },
    {
        "role": "user",
        "content": "When was he born?"
    }
]

chat_inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
    add_generation_prompt=True,
)

chat_input_ids = chat_inputs.input_ids.to(DEVICE)
chat_attention_mask = chat_inputs.attention_mask.to(DEVICE)

print("\nConversation:\n")
print(tokenizer.decode(chat_input_ids[0]))

generate(chat_input_ids, chat_attention_mask)
# Instruction-tuned LLMs expect structured conversations.
# apply_chat_template() reproduces the prompt format used during instruction tuning.