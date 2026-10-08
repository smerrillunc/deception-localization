"""A transformers-backed object exposing the slice of the vLLM API this repo uses.

Steering edits head outputs through forward hooks, which vLLM does not expose, so
the steering experiment runs generation through transformers instead. Installing
this module as ``sys.modules["vllm"]`` lets `deception_miner` and `utils` run
unchanged, preserving prompt construction, parsing and labelling exactly as in
the localization pipeline.

Surface reproduced (the only symbols the miner touches):
    LLM(model=..., max_model_len=..., seed=..., gpu_memory_utilization=...)
        .chat(conversations, sampling_params=...) -> list[RequestOutput]
        .generate(prompts, sampling_params=...)   -> list[RequestOutput]
        .get_tokenizer()
    SamplingParams(n=, temperature=, top_p=, max_tokens=, seed=,
                   repetition_penalty=, stop=)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    # Accept and ignore anything else vLLM callers might pass.
    _extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, temperature: float = 1.0, top_p: float = 1.0,
                 max_tokens: int = 512, repetition_penalty: float = 1.0,
                 seed: Optional[int] = None, **kwargs: Any) -> None:
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.repetition_penalty = float(repetition_penalty)
        self.seed = seed
        self._extra = dict(kwargs)


class _Completion:
    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class _RequestOutput:
    __slots__ = ("outputs",)

    def __init__(self, text: str) -> None:
        self.outputs = [_Completion(text)]


def _is_single_conversation(x: Any) -> bool:
    """A conversation is a list of {role, content} dicts; a batch is a list of those."""
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], dict))


class LLM:
    def __init__(self, model: str, max_model_len: int = 12288, seed: int = 0,
                 gpu_memory_utilization: float = 0.9, dtype: Any = None, **kwargs: Any) -> None:
        self.model_name = model
        self.max_model_len = int(max_model_len) if max_model_len else 12288
        self._seed = int(seed)

        logging.info("[hf_vllm_shim] loading tokenizer/model: %s", model)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        torch_dtype = torch.bfloat16 if dtype is None else dtype
        attn = os.environ.get("SHIM_ATTN_IMPL", "sdpa")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch_dtype, trust_remote_code=True,
                attn_implementation=attn,
            )
        except Exception:
            logging.exception("[hf_vllm_shim] attn=%s failed; retrying eager", attn)
            self.model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch_dtype, trust_remote_code=True,
                attn_implementation="eager",
            )
        self.model.to("cuda")
        self.model.eval()
        logging.info("[hf_vllm_shim] model ready on %s", next(self.model.parameters()).device)

    def get_tokenizer(self):
        return self.tokenizer

    def _render(self, conversation: Sequence[Dict[str, Any]]) -> str:
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )

    @torch.no_grad()
    def chat(self, messages: Any, sampling_params: Any = None, **kwargs: Any) -> List[_RequestOutput]:
        # Normalize conversations.
        if _is_single_conversation(messages):
            conversations = [messages]
        else:
            conversations = list(messages)
        n = len(conversations)

        # Normalize sampling params to a per-conversation list.
        if sampling_params is None:
            params = [SamplingParams()] * n
        elif isinstance(sampling_params, (list, tuple)):
            params = list(sampling_params)
        else:
            params = [sampling_params] * n
        if len(params) < n:
            params = params + [params[-1]] * (n - len(params))

        p0 = params[0]
        max_new = max(1, int(getattr(p0, "max_tokens", 512)))
        temperature = float(getattr(p0, "temperature", 1.0))
        top_p = float(getattr(p0, "top_p", 1.0))
        rep = float(getattr(p0, "repetition_penalty", 1.0))
        seed = getattr(p0, "seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))

        prompts = [self._render(c) for c in conversations]
        max_prompt_len = max(1, self.max_model_len - max_new)
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_prompt_len,
        ).to("cuda")
        input_len = enc["input_ids"].shape[1]

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=rep if rep and rep != 1.0 else None,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        out_ids = self.model.generate(**enc, **gen_kwargs)
        gen_ids = out_ids[:, input_len:]
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
        return [_RequestOutput(t) for t in texts]
