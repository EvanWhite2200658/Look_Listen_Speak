# src/language/response_generator.py

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.language.schemas import ResponseRequest, ResponseResult


class QwenResponseGenerator:
    """
    Modular LLM response generator using Qwen2.5-3B-Instruct.

    This module only converts text input into text output.
    It does not decide when the system may respond and must remain
    off the critical turn-taking path.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        default_system_prompt: str = (
            "You are a helpful conversational assistant. "
            "Respond naturally and briefly in spoken dialogue style."
        ),
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.default_system_prompt = default_system_prompt

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(None if torch_dtype == "auto" else torch_dtype),
            device_map=device_map,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, request: ResponseRequest) -> ResponseResult:

        messages = self._build_messages(request)


        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        device = next(self.model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        prompt_tokens = int(model_inputs["input_ids"].shape[-1])


        start = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start

        new_tokens = generated_ids[0][model_inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return ResponseResult(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=int(new_tokens.shape[-1]),
            generation_time_s=elapsed,
            model_name=self.model_name,
        )

    def _build_messages(self, request: ResponseRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        system_prompt = request.system_prompt or self.default_system_prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for turn in request.conversation_history:
            messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": request.user_text})
        return messages