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


# (4) Forward pass
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

next_token_logits = logits[:, -1, :]
probs = torch.softmax(next_token_logits, dim=-1)
print("\nProbability Shape:", probs.shape)
print("Probability Sum:", probs.sum())

topk_probs, topk_ids = torch.topk(probs, k=10)
print("\nTop 10 Candidate Tokens:\n")
for rank, (token_id, prob) in enumerate(zip(topk_ids[0], topk_probs[0]), start=0):
    token = tokenizer.decode(token_id)
    print(
        f"{rank:2}. "
        f"{repr(token):15} "
        f"{prob.item():.6f}"
    )
# Logits are unnormalized scores over the vocabulary. 
# Softmax rescales logits into probabilities while preserving their ordering. Thats why it sums to 1.
# Therefore argmax(logits) == argmax(softmax(logits)).
argmax_logits = torch.argmax(next_token_logits, dim=-1)
argmax_probs = torch.argmax(probs, dim=-1)
print(f"\nargmax(logits) : {argmax_logits.item()} ==", f"{argmax_probs.item()} : argmax(softmax(logits))")
print("\nGreedy Token:", repr(tokenizer.decode(argmax_logits)))


# (5) Temperature + Stochastic Sampling
print("\n\n" + "=" * 60)
print("\nTEMPERATURE & STOCHASTIC SAMPLING...")

for iter, temperature in enumerate([0.25, 1.1, 1.89]):

    print(f"\n{iter+1} Temperature = {temperature}")

    scaled_logits = next_token_logits / temperature
    scaled_probs = torch.softmax(scaled_logits, dim=-1)

    topk_probs, topk_ids = torch.topk(scaled_probs, k=10)

    print("\n\tTop 10 Tokens")

    for rank, (token_id, prob) in enumerate(zip(topk_ids[0], topk_probs[0]), start=1,):
        token = repr(tokenizer.decode(token_id))

        print("\t"
            f"{rank:2}. "
            f"{token:15} "
            f"{prob.item():.6f}"
        )

    print(f"\n\tTop-1 Probability: {topk_probs[0][0].item():.6f}")

    sampled = torch.multinomial(scaled_probs, num_samples=1)

    print("\tSampled Token:", repr(tokenizer.decode(sampled[0])))
    
    print("\n\tGreedy Token:", repr(tokenizer.decode(torch.argmax(probs, dim=-1))))

    print("\tOne Sampled Token:", end = "")
    sampled = torch.multinomial(probs, num_samples=1)
    print(repr(tokenizer.decode(sampled[0])))
# Temperature controls randomness:
#   T < 1 -> sharper distribution -> deterministic
#   T > 1 -> flatter distribution -> diverse
#
# Greedy = always highest-probability token.
# Multinomial sampling randomly draws a token according to the probability distribution, allowing lower-probability tokens to be selected occasionally.