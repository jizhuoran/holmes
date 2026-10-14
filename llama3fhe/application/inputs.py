"""Turning a prompt or a fixture into the model's plaintext input.

Everything the runner needs before encryption: resolving token ids from a
prompt or a JSON fixture, the padding token, and the synthetic hidden states
used when no checkpoint is available. Nothing here is encrypted yet — the
boundary is deliberate, because these are the only places a real prompt exists
in the clear.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..layouts import AttentionPairLayout, FeatureMajorPrefillLayout
from ..model import LayerWeights
from .checkpoint import CheckpointMetadata


def _padding_token(metadata: CheckpointMetadata, tokenizer) -> int:
    for value in (
        getattr(tokenizer, "pad_token_id", None),
        metadata.pad_token_id,
        getattr(tokenizer, "eos_token_id", None),
    ):
        if value is not None:
            return int(value)
    eos = metadata.eos_token_id
    if isinstance(eos, tuple) and eos:
        return int(eos[0])
    if isinstance(eos, int):
        return eos
    raise ValueError("checkpoint has no EOS/PAD token for 128-token padding.")


def _load_prompt_ids(
    prompt: str,
    *,
    model_dir: Path,
    metadata: CheckpointMetadata,
    seq_len: int,
    pad_side: str,
    add_special_tokens: bool,
) -> tuple[np.ndarray, object, int]:
    """Tokenize and pad a prompt that fits one encrypted Prefill page."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "--prompt requires transformers; use --input-ids-file otherwise."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ids = np.asarray(
        tokenizer.encode(
            str(prompt), add_special_tokens=bool(add_special_tokens)
        ),
        dtype=np.int64,
    )
    raw_count = int(ids.size)
    if raw_count == 0:
        raise ValueError("prompt tokenization produced no tokens.")
    if ids.size > int(seq_len):
        raise ValueError(
            f"prompt has {ids.size} tokens, exceeding the {seq_len}-token "
            "single-page Prefill limit; use the variable-length Prefill runner."
        )
    if ids.size < int(seq_len):
        padding = np.full(
            (int(seq_len) - ids.size,),
            _padding_token(metadata, tokenizer),
            dtype=np.int64,
        )
        ids = (
            np.concatenate((padding, ids))
            if str(pad_side) == "left"
            else np.concatenate((ids, padding))
        )
    return ids, tokenizer, raw_count


def _pad_input_ids(
    ids: np.ndarray, *, seq_len: int, pad_side: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Repeat a valid token in padding slots and return its public mask."""

    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    seq_len = int(seq_len)
    if not 1 <= ids.size <= seq_len:
        raise ValueError(
            f"single-page Prefill needs 1..{seq_len} token IDs, got {ids.size}."
        )
    if pad_side not in {"left", "right"}:
        raise ValueError(f"pad_side must be left or right, got {pad_side!r}.")
    valid = np.zeros(seq_len, dtype=bool)
    valid_count = int(ids.size)
    padding = np.full(seq_len - valid_count, ids[-1], dtype=np.int64)
    if pad_side == "left":
        valid[-valid_count:] = True
        return np.concatenate((padding, ids)), valid, seq_len - 1
    valid[:valid_count] = True
    return np.concatenate((ids, padding)), valid, valid_count - 1


def split_variable_prefill_ids(
    ids: np.ndarray, *, page_size: int = 128
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Prepare one encrypted Prefill page and retain every later token."""

    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        raise ValueError("Prefill needs at least one token ID.")
    page, valid, _ = _pad_input_ids(
        ids[: int(page_size)], seq_len=page_size, pad_side="right"
    )
    occupied = min(int(ids.size), int(page_size))
    return page, valid, occupied, ids[int(page_size) :].copy()


def split_prefill_pages(
    ids: np.ndarray, *, page_size: int = 128
) -> tuple[tuple[np.ndarray, np.ndarray, int], ...]:
    """Split every input token into fixed-width pages with a valid-slot mask."""

    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    page_size = int(page_size)
    if ids.size == 0:
        raise ValueError("Prefill needs at least one token ID.")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}.")
    pages = []
    for start in range(0, int(ids.size), page_size):
        chunk = ids[start : start + page_size]
        page, valid, _ = _pad_input_ids(
            chunk, seq_len=page_size, pad_side="right"
        )
        pages.append((page, valid, int(chunk.size)))
    return tuple(pages)


def _synthetic_inputs(
    seed: int,
    *,
    layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> tuple[np.ndarray, LayerWeights]:
    """Create deterministic full-shape data for the one-layer system preset."""

    rng = np.random.default_rng(int(seed))
    seq_len, hidden_dim = int(layout.seq_len), int(layout.hidden_dim)
    hidden = rng.normal(
        scale=0.01, size=(seq_len, hidden_dim)
    ).astype(np.float32)

    def weight(shape, scale=0.002):
        return rng.normal(scale=scale, size=shape).astype(np.float32)

    kv_dim = int(attention_layout.key_value_heads) * int(
        attention_layout.head_dim
    )
    return hidden, LayerWeights(
        input_layernorm=rng.normal(
            1.0, 0.01, size=(hidden_dim,)
        ).astype(np.float32),
        q_proj=weight((hidden_dim, hidden_dim)),
        k_proj=weight((kv_dim, hidden_dim)),
        v_proj=weight((kv_dim, hidden_dim)),
        o_proj=weight((hidden_dim, hidden_dim)),
        post_attention_layernorm=rng.normal(
            1.0, 0.01, size=(hidden_dim,)
        ).astype(np.float32),
        gate_proj=weight((14336, hidden_dim)),
        up_proj=weight((14336, hidden_dim)),
        down_proj=weight((hidden_dim, 14336)),
    )
