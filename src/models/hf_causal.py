from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.base import GenerationResult


def resolve_dtype(dtype: str):
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


class HFCausalBackend:
    def __init__(self, model_path: str, dtype: str = "float16"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=resolve_dtype(dtype),
            device_map="auto",
        ).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def hidden_size(self) -> int:
        config = getattr(self.model, "config")
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return int(config.text_config.hidden_size)
        return int(self.model.get_input_embeddings().embedding_dim)

    def _decode_generation(
        self,
        output_ids: torch.Tensor,
        prompt_tokens: int,
        latency_sec: float,
    ) -> GenerationResult:
        if output_ids.shape[-1] > prompt_tokens:
            generated_ids = output_ids[0][prompt_tokens:]
        else:
            generated_ids = output_ids[0]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        generated_tokens = int(generated_ids.shape[-1])
        return GenerationResult(
            text=text,
            prompt_tokens=int(prompt_tokens),
            generated_tokens=generated_tokens,
            total_tokens=int(prompt_tokens + generated_tokens),
            latency_sec=round(latency_sec, 4),
        )

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self._decode_generation(output_ids, prompt_tokens, time.perf_counter() - start)

    def generate_with_soft_prompt(
        self,
        prompt: str,
        soft_prompt_embeds: torch.Tensor,
        max_new_tokens: int,
    ) -> GenerationResult:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        token_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
        if soft_prompt_embeds.ndim == 2:
            soft_prompt_embeds = soft_prompt_embeds.unsqueeze(0)
        soft_prompt_embeds = soft_prompt_embeds.to(
            device=token_embeds.device,
            dtype=token_embeds.dtype,
        )
        inputs_embeds = torch.cat([soft_prompt_embeds, token_embeds], dim=1)
        soft_mask = torch.ones(
            (inputs_embeds.shape[0], soft_prompt_embeds.shape[1]),
            device=inputs["attention_mask"].device,
            dtype=inputs["attention_mask"].dtype,
        )
        attention_mask = torch.cat([soft_mask, inputs["attention_mask"]], dim=1)

        prompt_tokens = int(inputs_embeds.shape[1])
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self._decode_generation(output_ids, prompt_tokens, time.perf_counter() - start)

