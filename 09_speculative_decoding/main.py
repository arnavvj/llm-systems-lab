"""
Project 09

Speculative Decoding

A small draft model greedily proposes several tokens ahead; a larger target
model verifies them in a single forward pass. Matching tokens are accepted
for free, and generation falls back to the target model's own prediction at
the first mismatch -- so the output is identical to running the target
model alone, just computed in fewer forward passes.
"""

import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# (1) Config

DRAFT_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
TARGET_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

NUM_DRAFT_TOKENS = 4
MAX_NEW_TOKENS = 64

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# (2) / (3) Load Draft + Target Models
# Both share the SmolLM2 tokenizer, so identical input_ids are valid input
# for either model -- that's what makes their predictions comparable.

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return model


tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
draft_model = load_model(DRAFT_MODEL_NAME)
target_model = load_model(TARGET_MODEL_NAME)


def encode(prompt):
    """Apply the chat template both instruct-tuned models expect."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    return inputs.input_ids.to(DEVICE)


# (4) Greedy Baseline

def greedy_generate(model, input_ids, max_new_tokens=MAX_NEW_TOKENS):
    """Plain autoregressive greedy decoding, one forward pass per token."""
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return input_ids


# (5) Draft Multiple Tokens

def draft_tokens(input_ids, k=NUM_DRAFT_TOKENS):
    """Greedily propose up to k tokens ahead with the draft model."""
    draft_ids = greedy_generate(draft_model, input_ids, max_new_tokens=k)
    return draft_ids[:, input_ids.shape[1]:]


# (6) / (7) Verify + Accept/Reject

def verify_tokens(input_ids, draft):
    """
    Verify draft tokens against the target model in a single forward pass.
    Accepts the longest prefix matching the target's own greedy prediction,
    then appends the target's prediction at the mismatch (or bonus) slot.
    """
    context_len = input_ids.shape[1]
    combined = torch.cat([input_ids, draft], dim=1)

    logits = target_model(combined).logits
    target_preds = torch.argmax(logits[:, context_len - 1:, :], dim=-1)

    accepted = 0
    for i in range(draft.shape[1]):
        if target_preds[0, i] == draft[0, i]:
            accepted += 1
        else:
            break

    bonus_token = target_preds[:, accepted:accepted + 1]
    return torch.cat([draft[:, :accepted], bonus_token], dim=1), accepted


# (8) Speculative Decoding Loop + Runtime Comparison

def speculative_generate(input_ids, max_new_tokens=MAX_NEW_TOKENS):
    """Loop draft -> verify -> accept/reject until max_new_tokens or EOS."""
    generated = input_ids
    stats = {"rounds": 0, "accepted": 0, "drafted": 0}

    while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
        draft = draft_tokens(generated)
        round_ids, accepted = verify_tokens(generated, draft)

        eos_pos = (round_ids[0] == tokenizer.eos_token_id).nonzero()
        if len(eos_pos) > 0:
            round_ids = round_ids[:, :eos_pos[0].item() + 1]

        generated = torch.cat([generated, round_ids], dim=1)
        stats["rounds"] += 1
        stats["accepted"] += accepted
        stats["drafted"] += draft.shape[1]

        if len(eos_pos) > 0:
            break

    stats["acceptance_rate"] = stats["accepted"] / stats["drafted"] if stats["drafted"] else 0
    return generated, stats


def main():
    prompt = "Do you knwo whats the 5th and the sixth planet in the solar system?"
    input_ids = encode(prompt)

    # Warm up both models once so the timed comparison below isn't skewed by
    # first-call overhead (kernel compilation, allocator first-touch, etc).
    with torch.inference_mode():
        draft_model(input_ids)
        target_model(input_ids)

    with torch.inference_mode():
        start = time.perf_counter()
        baseline_ids = greedy_generate(target_model, input_ids)
        baseline_time = time.perf_counter() - start

        start = time.perf_counter()
        spec_ids, stats = speculative_generate(input_ids)
        spec_time = time.perf_counter() - start

    baseline_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)
    spec_text = tokenizer.decode(spec_ids[0], skip_special_tokens=True)

    print(f"Prompt             : {prompt}")
    print(f"Baseline (target)  : {baseline_text}")
    print(f"Speculative        : {spec_text}")
    print(f"Outputs match      : {baseline_text == spec_text}")
    print("-" * 60)
    print(f"Baseline time      : {baseline_time:.3f}s")
    print(f"Speculative time   : {spec_time:.3f}s")
    print(f"Speedup            : {baseline_time / spec_time:.2f}x")
    print(f"Rounds             : {stats['rounds']}")
    print(f"Accepted / Drafted : {stats['accepted']}/{stats['drafted']} "
          f"({stats['acceptance_rate']:.0%} acceptance)")


if __name__ == "__main__":
    main()
