"""
Inference Engine

Wraps tokenization and KV-cached streaming generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class InferenceEngine:

    def __init__(self, model_name):

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_inputs(self, prompt):

        messages = [{"role": "user", "content": prompt}]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        return input_ids, attention_mask

    def stream_generate_kv_cache(self, prompt, max_new_tokens=64):
        """Greedy-decode token by token, reusing the KV cache across steps."""

        input_ids, attention_mask = self._prepare_inputs(prompt)

        with torch.inference_mode():

            # First pass processes the entire prompt and populates the KV cache.
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            for _ in range(max_new_tokens):

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                token = self.tokenizer.decode(next_token[0])
                yield token  # immediately stream token

                # Subsequent steps only feed the newly generated token,
                # avoiding repeated computation over the prompt.
                outputs = self.model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
