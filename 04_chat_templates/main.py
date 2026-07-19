"""
Project 04

Chat templates (instruct models expect user/assistant turns, not raw text)

Agenda: format prompts the way instruct models were trained on using:
1. tokenizer.apply_chat_template()
2. special tokens / role markers (e.g. <|im_start|>, <|im_end|>)
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


# (3) Tokenize and Prompt
# Decoder-only LLMs are autoregressive: given tokens [t1...tn], predict only the next token (tn+1).
# During training, causal masking hides future tokens so every position learns the same next-token prediction task used at inference.
print("\n\n" + "=" * 60)
print(f"\n===> TOKENIZER ON RAW INPUT FORMAT..")
INPUT_PROMPT = "Why is the sky blue?"
raw_inputs = tokenizer(
        INPUT_PROMPT,
        return_tensors = 'pt'
    )
print("\n1.", raw_inputs)
raw_input_ids = raw_inputs.input_ids.to(DEVICE)
raw_attention_mask = raw_inputs.attention_mask.to(DEVICE)
print("2. Input prompt:", INPUT_PROMPT)
print("3. Input IDs:", raw_input_ids)
print("4. Input Shape:", raw_input_ids.shape)
print("5. Tokenizer decoded:", tokenizer.decode(raw_inputs.input_ids[0]))
print("6. RAW Prompt Tokens :", raw_input_ids.shape[1])

outputs_from_raw = model(input_ids=raw_input_ids, attention_mask=raw_attention_mask)
next_token_from_raw = torch.argmax(outputs_from_raw.logits[:, -1, :], dim=-1)
print("7. Raw Prompt Prediction:", repr(tokenizer.decode(next_token_from_raw)))

print(f"\n====> TOKENIZER ON ROLE BASED MESSAGE FORMAT..")
messages = [
    {
        "role": "user",
        "content": INPUT_PROMPT,
    }
]
chat_inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
    add_generation_prompt=True,
)
print("\n1.", chat_inputs)
chat_input_ids = chat_inputs.input_ids.to(DEVICE)
chat_attention_mask = chat_inputs.attention_mask.to(DEVICE)
print("2. Input prompt:", INPUT_PROMPT)
print("3. Input IDs:", chat_input_ids)
print("4. Input Shape:", chat_input_ids.shape)
print("5. Tokenizer decoded:", tokenizer.decode(chat_inputs.input_ids[0]))
print("6. CHAT Prompt Tokens:", chat_input_ids.shape[1])

outputs_from_chat = model(input_ids=chat_input_ids, attention_mask=chat_attention_mask)
next_token_from_chat = torch.argmax(outputs_from_chat.logits[:, -1, :], dim=-1)
print("7. CHAT Prompt Prediction:", repr(tokenizer.decode(next_token_from_chat)))
# Instruction-tuned LLMs expect structured conversations rather than raw text.
# apply_chat_template() formats the conversation exactly as the model saw during training.


# (4) AUTOREGRESSIVE GENERATION
print("\n\n" + "=" * 60)
print("\nAUTOREGRESSIVE GENERATION...\n")

initial_length = chat_input_ids.shape[1]
MAX_NEW_TOKENS = 20

with torch.inference_mode():

    for step in range(MAX_NEW_TOKENS):

        outputs = model(input_ids=chat_input_ids, attention_mask=chat_attention_mask)

        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        print(
            f"Step {step+1:2}: "
            f"Add {next_token_id.item():5} "
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

print("\nFinal input looks like:\n")
print(tokenizer.decode(chat_input_ids[0]))
generated_ids = chat_input_ids[0][initial_length:]
print("\nAssistant Response:\n")
print(tokenizer.decode(generated_ids))