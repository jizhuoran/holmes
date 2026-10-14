"""Hugging Face Llama baseline loader and generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass(frozen=True)
class LlamaLoadConfig:
    model_id_or_path: str
    device_map: str = "auto"
    torch_dtype: str = "float16"  # float16 | bfloat16 | float32
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None  # e.g. "sdpa" or "eager"


def _resolve_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key in ("float16", "fp16"):
        return torch.float16
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    if key in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def resolve_model_path(
    model_id: Optional[str] = None,
    local_dir: Optional[Union[str, Path]] = None,
    default_id: str = "meta-llama/Meta-Llama-3-8B",
) -> str:
    if local_dir is not None:
        path = Path(local_dir).expanduser().resolve()
        if path.is_dir():
            return str(path)
        raise FileNotFoundError(f"Local model directory not found: {path}")
    return model_id or default_id


def load_llama_baseline(
    cfg: LlamaLoadConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    dtype = _resolve_dtype(cfg.torch_dtype)
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": cfg.device_map,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id_or_path, **model_kwargs)
    model.eval()
    return model, tokenizer


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> List[str]:
    outputs: List[str] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs.append(text)
    return outputs


def prefill_then_decode_one_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> Dict[str, Any]:
    """
    Explicit prefill (full prompt) + single-token decode with KV cache.
    Useful for validating the two-phase inference path used in LLM serving.
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        prefill_out = model(input_ids=input_ids, use_cache=True)
        past = prefill_out.past_key_values
        last_logits = prefill_out.logits[:, -1, :]
        next_id = torch.argmax(last_logits, dim=-1, keepdim=True)
        decode_out = model(input_ids=next_id, past_key_values=past, use_cache=True)
    next_token = tokenizer.decode(next_id[0], skip_special_tokens=True)
    return {
        "prefill_tokens": int(input_ids.shape[1]),
        "next_token": next_token,
        "next_token_id": int(next_id.item()),
        "decode_logits_shape": list(decode_out.logits.shape),
    }
