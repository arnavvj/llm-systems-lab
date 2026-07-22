"""
Inference Engine

Wraps tokenization, generation, streaming, batching and benchmarking.
"""

import time
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

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

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


    def generate(self, prompt, max_new_tokens=64):

        input_ids, attention_mask = self._prepare_inputs(prompt)
        initial_length = input_ids.shape[1]

        with torch.inference_mode():

            for _ in range(max_new_tokens):

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

                input_ids = torch.cat(
                    [input_ids, next_token],
                    dim=1,
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (1, 1),
                            device=self.device,
                            dtype=attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        generated = input_ids[0][initial_length:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )
    

    def batch_generate(         # <---- this is intentionally fake
        self,
        prompts,
        max_new_tokens=64,
    ):

        responses = []
        for prompt in prompts:

            responses.append(
                self.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                )
            )

        return responses
    # Sequential batching for simplicity.
    # Production inference engines (vLLM, TGI, Fireworks, Baseten)
    # execute prompts in the same forward pass using continuous batching.
    # To be continued in project 08


    def stream_generate(
        self,
        prompt,
        max_new_tokens=64,
    ):

        input_ids, attention_mask = self._prepare_inputs(prompt)

        with torch.inference_mode():

            for step in range(max_new_tokens):

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

                token = self.tokenizer.decode(next_token[0])
                yield token     # <---- immediately yield (i.e. stream produced token)

                input_ids = torch.cat(
                    [input_ids, next_token],
                    dim=1,
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (1, 1),
                            device=self.device,
                            dtype=attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )

                if next_token.item() == self.tokenizer.eos_token_id:
                    break


    def stream_generate_kv_cache(
        self,
        prompt,
        max_new_tokens=64,
    ):

        input_ids, attention_mask = self._prepare_inputs(prompt)
        past_key_values = None

        with torch.inference_mode():

            # First pass over the entire prompt
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values
            
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            for _ in range(max_new_tokens):

                token = self.tokenizer.decode(next_token[0])
                yield token      # <---- immediately stream token

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                outputs = self.model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values

                next_token = torch.argmax(
                    outputs.logits[:, -1, :],
                    dim=-1,
                    keepdim=True,
                )
    # First forward pass processes the entire prompt and populates the KV cache.
    # Subsequent decoding reuses the cached prompt state, so only the newly generated
    # token is fed back into the model, avoiding repeated computation over the prompt.


    def benchmark_streaming(
        self,
        prompt,
    ):
        
        print("PROMPT for benchmarking:", prompt)
        
        print("\n1. Vanilla Streaming (No KV Cache):")
        start = time.perf_counter()

        first_token, counter = None, 0
        for token in self.stream_generate(prompt):
            if first_token is None:
                first_token = time.perf_counter()
            counter += 1

        end = time.perf_counter()
        print(f"TTFT                 : {first_token-start:.3f}s")
        print(f"Total Generation Time: {end-start:.3f}s")
        print(f"TTFT compared to average time per token = {first_token-start:.5f}::{(end-first_token)/(counter-1):.5f}")


        print("\n2. Optimized Streaming (KV Cache):")
        start = time.perf_counter()

        first_token, counter = None, 0
        for token in self.stream_generate_kv_cache(prompt):
            if first_token is None:
                first_token = time.perf_counter()
            counter += 1

        end = time.perf_counter()

        print(f"TTFT                 : {first_token-start:.3f}s")
        print(f"Total Generation Time: {end-start:.3f}s")
        print(f"TTFT compared to average time per remaining token = {first_token-start:.5f}::{(end-first_token)/(counter-1):.5f}")

        print("\nStats calculated in streaming mode!")