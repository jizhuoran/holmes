from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attention import AttentionPairLayout


def _token_heads(
    token: np.ndarray, *, heads: int, head_dim: int, name: str
) -> np.ndarray:
    token = np.asarray(token, dtype=np.float32)
    shaped = (int(heads), int(head_dim))
    flat = (int(heads) * int(head_dim),)
    if token.shape == shaped:
        return token
    if token.shape == flat:
        return token.reshape(shaped)
    if token.shape == (1, *shaped):
        return token[0]
    if token.shape == (1, flat[0]):
        return token.reshape(shaped)
    raise ValueError(
        f"{name} token must have shape {shaped} or {flat}, got {token.shape}."
    )


def empty_c_page(layout: AttentionPairLayout, *, dtype=np.float64) -> np.ndarray:
    """Zero C-form payload for one KV pair-cipher page."""

    return np.zeros(
        (int(layout.key_value_pair_count), int(layout.slots)), dtype=dtype
    )


@dataclass
class KvCachePage:
    """One C-form K/V page with at most ``layout.seq_len`` occupied tokens."""

    layout: AttentionPairLayout
    occupied: int = 0
    k_c: np.ndarray = field(init=False)
    v_c: np.ndarray = field(init=False)
    rope_start: int = 0

    def __post_init__(self) -> None:
        self.k_c = empty_c_page(self.layout)
        self.v_c = empty_c_page(self.layout)
        occupied = int(self.occupied)
        if occupied:
            raise ValueError("construct an empty page and append tokens into it.")

    @property
    def remaining(self) -> int:
        return int(self.layout.seq_len) - int(self.occupied)

    def append(self, key_token: np.ndarray, value_token: np.ndarray) -> int:
        """Write the next token column. Returns the absolute token index on this page."""

        if int(self.occupied) >= int(self.layout.seq_len):
            raise ValueError("C-form KV page is full.")
        token_index = int(self.occupied)
        heads = int(self.layout.key_value_heads)
        head_dim = int(self.layout.head_dim)
        key = _token_heads(key_token, heads=heads, head_dim=head_dim, name="key")
        value = _token_heads(
            value_token, heads=heads, head_dim=head_dim, name="value"
        )
        # ``pack_token_c`` returns a full-page sparse payload.  The previous
        # ``self.page = self.page + payload`` form allocated two additional
        # 4×32768 arrays for every appended token (O(S²) host traffic during
        # prefill/prefix construction).  Pages are private and append-only,
        # so an in-place add is equivalent and keeps peak memory constant.
        self.k_c += self.layout.pack_token_c(
            key, token_index=token_index, kv=True, name="key"
        )
        self.v_c += self.layout.pack_token_c(
            value, token_index=token_index, kv=True, name="value"
        )
        self.occupied = token_index + 1
        return token_index

    def packed_keys(self) -> np.ndarray:
        return np.asarray(self.k_c)

    def packed_values(self) -> np.ndarray:
        return np.asarray(self.v_c)

    def logical_keys(self) -> np.ndarray:
        keys = self.layout.unpack_key_c(self.k_c)
        return keys[: int(self.occupied)]

    def logical_values(self) -> np.ndarray:
        values = self.layout.unpack_value_c(self.v_c)
        return values[: int(self.occupied)]


@dataclass
class LayerKvCache:
    """Append-only C-form KV cache for one transformer layer."""

    layout: AttentionPairLayout
    pages: list[KvCachePage] = field(default_factory=list)
    rope_start: int = 0

    @property
    def occupied(self) -> int:
        return int(sum(int(page.occupied) for page in self.pages))

    @classmethod
    def from_prefill(
        cls,
        key: np.ndarray,
        value: np.ndarray,
        *,
        layout: AttentionPairLayout,
    ) -> "LayerKvCache":
        """Fill one C page from a Prefill window of RoPE'd K/V."""

        cache = cls(layout=layout)
        key = layout._logical_heads(
            key, heads=int(layout.key_value_heads), name="key"
        )
        value = layout._logical_heads(
            value, heads=int(layout.key_value_heads), name="value"
        )
        if int(key.shape[0]) != int(value.shape[0]):
            raise ValueError("prefill K/V token counts must match.")
        if int(key.shape[0]) > int(layout.seq_len):
            raise ValueError("prefill KV exceeds one C page.")
        for token in range(int(key.shape[0])):
            cache.append(key[token], value[token])
        return cache

    def append(self, key_token: np.ndarray, value_token: np.ndarray) -> tuple[int, int]:
        """Append one token. Returns ``(page_index, token_index_in_page)``."""

        if not self.pages or int(self.pages[-1].remaining) <= 0:
            page_rope = int(self.rope_start) + int(self.occupied)
            self.pages.append(
                KvCachePage(layout=self.layout, rope_start=page_rope)
            )
        page_index = len(self.pages) - 1
        token_index = self.pages[-1].append(key_token, value_token)
        return page_index, token_index


__all__ = [
    "KvCachePage",
    "LayerKvCache",
    "empty_c_page",
]
