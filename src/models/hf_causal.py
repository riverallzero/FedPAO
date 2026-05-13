from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.base import GenerationResult, ScoringResult


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

    def score_continuations(
        self,
        prompt: str,
        continuations: dict[str, str],
        soft_prompt_embeds: torch.Tensor | None = None,
    ) -> ScoringResult:
        if not continuations:
            raise ValueError("continuations must not be empty")

        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        continuation_ids = {
            key: self.tokenizer(value, add_special_tokens=False)["input_ids"]
            for key, value in continuations.items()
        }
        if any(len(ids) == 0 for ids in continuation_ids.values()):
            raise ValueError("continuations must tokenize to at least one token")

        device = self.device
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        full_sequences = [
            torch.cat(
                [
                    prompt_ids,
                    torch.tensor(ids, dtype=prompt_ids.dtype),
                ]
            )
            for ids in continuation_ids.values()
        ]
        max_length = max(int(sequence.shape[-1]) for sequence in full_sequences)
        input_ids = torch.full(
            (len(full_sequences), max_length),
            int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(full_sequences), max_length),
            dtype=torch.long,
            device=device,
        )
        for row, sequence in enumerate(full_sequences):
            sequence = sequence.to(device)
            input_ids[row, : sequence.shape[-1]] = sequence
            attention_mask[row, : sequence.shape[-1]] = 1

        token_embeds = self.model.get_input_embeddings()(input_ids)
        soft_length = 0
        if soft_prompt_embeds is not None:
            if soft_prompt_embeds.ndim == 2:
                soft_prompt_embeds = soft_prompt_embeds.unsqueeze(0)
            soft_prompt_embeds = soft_prompt_embeds.to(
                device=token_embeds.device,
                dtype=token_embeds.dtype,
            )
            soft_prompt_embeds = soft_prompt_embeds.expand(token_embeds.shape[0], -1, -1)
            token_embeds = torch.cat([soft_prompt_embeds, token_embeds], dim=1)
            soft_length = int(soft_prompt_embeds.shape[1])
            soft_mask = torch.ones(
                (attention_mask.shape[0], soft_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(
                inputs_embeds=token_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
        log_probs = torch.log_softmax(logits, dim=-1)

        prompt_length = int(prompt_ids.shape[-1])
        scores = {}
        max_continuation_tokens = 0
        for row, (key, ids) in enumerate(continuation_ids.items()):
            max_continuation_tokens = max(max_continuation_tokens, len(ids))
            score = 0.0
            for offset, token_id in enumerate(ids):
                full_token_index = prompt_length + offset
                logits_index = soft_length + full_token_index - 1
                score += float(log_probs[row, logits_index, int(token_id)].item())
            scores[key] = score

        prompt_tokens = prompt_length + soft_length
        return ScoringResult(
            scores=scores,
            prompt_tokens=prompt_tokens,
            continuation_tokens=max_continuation_tokens,
            total_tokens=prompt_tokens + max_continuation_tokens,
            latency_sec=round(time.perf_counter() - start, 4),
        )

    def score_choices_with_soft_prompt(
        self,
        prompt: str,
        choices: list[str],
        soft_prompt_embeds: torch.Tensor | None = None,
    ) -> ScoringResult:
        return self.score_continuations(
            prompt,
            {choice: f" {choice}" for choice in choices},
            soft_prompt_embeds,
        )

    def score_answer_with_soft_prompt(
        self,
        prompt: str,
        answer: str,
        soft_prompt_embeds: torch.Tensor | None = None,
    ) -> ScoringResult:
        return self.score_continuations(
            prompt,
            {"answer": f" {answer}"},
            soft_prompt_embeds,
        )
