from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AttentionQKVPayload:
    query_c: np.ndarray
    key_delta: np.ndarray
    value_delta: np.ndarray


@dataclass(frozen=True)
class AttentionPairLayout:
    """Two-head interleaved Attention ``C/Delta`` working layouts."""

    seq_len: int = 128
    head_dim: int = 128
    query_heads: int = 32
    key_value_heads: int = 8
    slots: int = 32768

    def __post_init__(self) -> None:
        for name in ("seq_len", "head_dim", "query_heads", "key_value_heads", "slots"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        if int(self.query_heads) % 2 or int(self.key_value_heads) % 2:
            raise ValueError("two-head interleaving requires even Q and KV head counts.")
        if int(self.query_heads) % int(self.key_value_heads):
            raise ValueError("query_heads must be divisible by key_value_heads for GQA.")
        if int(self.seq_len) != int(self.head_dim):
            raise ValueError("the current C/Delta design requires seq_len == head_dim.")
        if 2 * int(self.seq_len) * int(self.head_dim) > int(self.slots):
            raise ValueError(
                "one two-head payload does not fit: "
                f"2*{self.seq_len}*{self.head_dim} > slots={self.slots}."
            )
        if int(self.slots) % self.payload_slots:
            raise ValueError(
                f"slots={self.slots} must be divisible by the two-head payload size={self.payload_slots}."
            )

    @property
    def payload_slots(self) -> int:
        return 2 * int(self.seq_len) * int(self.head_dim)

    @property
    def payload_repetitions(self) -> int:
        return int(self.slots) // self.payload_slots

    @property
    def gqa_ratio(self) -> int:
        return int(self.query_heads) // int(self.key_value_heads)

    @property
    def query_pair_count(self) -> int:
        return int(self.query_heads) // 2

    @property
    def key_value_pair_count(self) -> int:
        return int(self.key_value_heads) // 2


    def query_head_pairs(self) -> tuple[tuple[int, int], ...]:
        ratio = self.gqa_ratio
        pairs: list[tuple[int, int]] = []
        for kv_pair in range(self.key_value_pair_count):
            q_base = 2 * ratio * kv_pair
            for offset in range(ratio):
                pairs.append((q_base + offset, q_base + ratio + offset))
        return tuple(pairs)

    def key_value_head_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((2 * pair, 2 * pair + 1) for pair in range(self.key_value_pair_count))

    def _logical_heads(self, values: np.ndarray, *, heads: int, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        flat_shape = (int(self.seq_len), int(heads) * int(self.head_dim))
        head_shape = (int(self.seq_len), int(heads), int(self.head_dim))
        if values.shape == flat_shape:
            return values.reshape(head_shape)
        if values.shape == head_shape:
            return values
        raise ValueError(f"{name} must have shape {flat_shape} or {head_shape}, got {values.shape}.")

    def _repeat_payload(self, base: np.ndarray) -> np.ndarray:
        base = np.asarray(base)
        if base.ndim != 2 or base.shape[1] != self.payload_slots:
            raise ValueError(f"base payload must have shape [pairs, {self.payload_slots}], got {base.shape}.")
        return np.tile(base, (1, self.payload_repetitions))

    def _base_payload(self, packed: np.ndarray, *, pairs: int, name: str) -> np.ndarray:
        packed = np.asarray(packed)
        expected = (int(pairs), int(self.slots))
        if packed.shape != expected:
            raise ValueError(f"{name} payload must have shape {expected}, got {packed.shape}.")
        base = packed[:, : self.payload_slots]
        for repetition in range(1, self.payload_repetitions):
            start = repetition * self.payload_slots
            if not np.allclose(
                base,
                packed[:, start : start + self.payload_slots],
                atol=1e-5,
                rtol=1e-5,
            ):
                raise ValueError(f"{name} repeated payload copy {repetition} does not match copy 0.")
        return base

    def c_slot(self, feature: int, token: int, lane: int) -> int:
        """C-form slot ``2 * (feature * S + token) + lane`` inside one payload."""

        feature = int(feature)
        token = int(token)
        lane = int(lane)
        if not 0 <= feature < int(self.head_dim):
            raise IndexError(f"feature={feature} is outside [0, {self.head_dim}).")
        if not 0 <= token < int(self.seq_len):
            raise IndexError(f"token={token} is outside [0, {self.seq_len}).")
        if lane not in (0, 1):
            raise IndexError(f"lane={lane} must be 0 or 1.")
        return 2 * (feature * int(self.seq_len) + token) + lane

    def _pack_c(
        self,
        values: np.ndarray,
        *,
        heads: int,
        head_pairs: tuple[tuple[int, int], ...],
        name: str,
        dtype=np.float64,
    ) -> np.ndarray:
        values = self._logical_heads(values, heads=int(heads), name=name)
        packed = np.zeros((len(head_pairs), self.payload_slots), dtype=dtype)
        tokens = np.arange(int(self.seq_len), dtype=np.int64)
        features = np.arange(int(self.head_dim), dtype=np.int64)
        for pair_index, pair_heads in enumerate(head_pairs):
            for lane, head in enumerate(pair_heads):
                slots = 2 * (
                    features[:, None] * int(self.seq_len) + tokens[None, :]
                ) + lane
                packed[pair_index, slots.reshape(-1)] = values[:, head, :].T.reshape(-1)
        return self._repeat_payload(packed)

    def _unpack_c(
        self,
        packed: np.ndarray,
        *,
        heads: int,
        head_pairs: tuple[tuple[int, int], ...],
        name: str,
    ) -> np.ndarray:
        packed = self._base_payload(packed, pairs=len(head_pairs), name=name)
        values = np.zeros(
            (int(self.seq_len), int(heads), int(self.head_dim)), dtype=np.float32
        )
        tokens = np.arange(int(self.seq_len), dtype=np.int64)
        features = np.arange(int(self.head_dim), dtype=np.int64)
        for pair_index, pair_heads in enumerate(head_pairs):
            for lane, head in enumerate(pair_heads):
                slots = 2 * (
                    features[:, None] * int(self.seq_len) + tokens[None, :]
                ) + lane
                values[:, head, :] = packed[pair_index, slots.reshape(-1)].reshape(
                    int(self.head_dim), int(self.seq_len)
                ).T
        return values.reshape(int(self.seq_len), -1)

    def pack_query_c(self, query: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        return self._pack_c(
            query,
            heads=int(self.query_heads),
            head_pairs=self.query_head_pairs(),
            name="query",
            dtype=dtype,
        )

    def unpack_query_c(self, packed: np.ndarray) -> np.ndarray:
        return self._unpack_c(
            packed,
            heads=int(self.query_heads),
            head_pairs=self.query_head_pairs(),
            name="Q_C",
        )

    def pack_key_c(self, key: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        return self._pack_c(
            key,
            heads=int(self.key_value_heads),
            head_pairs=self.key_value_head_pairs(),
            name="key",
            dtype=dtype,
        )

    def unpack_key_c(self, packed: np.ndarray) -> np.ndarray:
        return self._unpack_c(
            packed,
            heads=int(self.key_value_heads),
            head_pairs=self.key_value_head_pairs(),
            name="K_C",
        )

    def pack_value_c(self, value: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        return self._pack_c(
            value,
            heads=int(self.key_value_heads),
            head_pairs=self.key_value_head_pairs(),
            name="value",
            dtype=dtype,
        )

    def unpack_value_c(self, packed: np.ndarray) -> np.ndarray:
        return self._unpack_c(
            packed,
            heads=int(self.key_value_heads),
            head_pairs=self.key_value_head_pairs(),
            name="V_C",
        )

    def pack_token_c(
        self,
        token: np.ndarray,
        *,
        token_index: int,
        kv: bool = True,
        name: str | None = None,
        dtype=np.float64,
    ) -> np.ndarray:
        """Pack one token into an otherwise-zero C page at ``token_index``."""

        token_index = int(token_index)
        if not 0 <= token_index < int(self.seq_len):
            raise IndexError(
                f"token_index={token_index} is outside [0, {self.seq_len})."
            )
        heads = int(self.key_value_heads if kv else self.query_heads)
        head_pairs = (
            self.key_value_head_pairs() if kv else self.query_head_pairs()
        )
        label = str(name or ("key" if kv else "query"))
        pages = np.zeros(
            (int(self.seq_len), heads, int(self.head_dim)), dtype=np.float32
        )
        token = np.asarray(token, dtype=np.float32)
        shaped = (heads, int(self.head_dim))
        flat = (heads * int(self.head_dim),)
        if token.shape == shaped:
            pages[token_index] = token
        elif token.shape == flat:
            pages[token_index] = token.reshape(shaped)
        elif token.shape == (1, *shaped):
            pages[token_index] = token[0]
        elif token.shape == (1, flat[0]):
            pages[token_index] = token.reshape(shaped)
        else:
            raise ValueError(
                f"{label} token must have shape {shaped}, {flat}, "
                f"{(1, *shaped)}, or {(1, flat[0])}, got {token.shape}."
            )
        return self._pack_c(
            pages,
            heads=heads,
            head_pairs=head_pairs,
            name=label,
            dtype=dtype,
        )

    def pack_query_c_feature_folded(
        self, query: np.ndarray, *, scale: float = 0.5, dtype=np.complex128
    ) -> np.ndarray:
        return self._pack_c_feature_folded(
            query,
            heads=int(self.query_heads),
            head_pairs=self.query_head_pairs(),
            name="query",
            scale=scale,
            dtype=dtype,
        )

    def pack_key_c_feature_folded(
        self, key: np.ndarray, *, scale: float = 0.5, dtype=np.complex128
    ) -> np.ndarray:
        """Same C-grid as Q folding, for Prefill cache snapshots of K."""

        return self._pack_c_feature_folded(
            key,
            heads=int(self.key_value_heads),
            head_pairs=self.key_value_head_pairs(),
            name="key",
            scale=scale,
            dtype=dtype,
        )

    def _pack_c_feature_folded(
        self,
        values: np.ndarray,
        *,
        heads: int,
        head_pairs: tuple[tuple[int, int], ...],
        name: str,
        scale: float,
        dtype,
    ) -> np.ndarray:
        if int(self.head_dim) % 2:
            raise ValueError(f"complex {name} feature folding requires even head_dim.")
        values = self._logical_heads(values, heads=int(heads), name=name)
        packed = np.zeros((len(head_pairs), self.payload_slots), dtype=dtype)
        tokens = np.arange(int(self.seq_len), dtype=np.int64)
        features = np.arange(int(self.head_dim), dtype=np.int64)
        paired_features = (features + int(self.head_dim) // 2) % int(self.head_dim)
        slots_base = 2 * (features[:, None] * int(self.seq_len) + tokens[None, :])
        for pair_index, pair_heads in enumerate(head_pairs):
            for lane, head in enumerate(pair_heads):
                real = values[:, head, :].T
                imaginary = values[:, head, paired_features].T
                packed[pair_index, (slots_base + lane).reshape(-1)] = (
                    float(scale) * (real + 1j * imaginary)
                ).reshape(-1)
        return self._repeat_payload(packed)

    def pack_decode_scores(self, scores: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        """Pack 1-query (or N-query row-0) scores onto the C token axis.

        Logical shape is ``[query_heads, seq_len]``.  Feature row 0 holds the
        scores; remaining feature rows stay zero.  This is the Decode Softmax
        and PV token axis, not ``S_Delta``.
        """

        scores = np.asarray(scores, dtype=np.float32)
        expected = (int(self.query_heads), int(self.seq_len))
        if scores.shape != expected:
            raise ValueError(f"decode scores must have shape {expected}, got {scores.shape}.")
        packed = np.zeros((self.query_pair_count, self.payload_slots), dtype=dtype)
        tokens = np.arange(int(self.seq_len), dtype=np.int64)
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                packed[pair_index, 2 * tokens + lane] = scores[head]
        return self._repeat_payload(packed)

    def unpack_decode_scores(self, packed: np.ndarray) -> np.ndarray:
        packed = self._base_payload(
            packed, pairs=self.query_pair_count, name="decode_scores"
        )
        scores = np.zeros(
            (int(self.query_heads), int(self.seq_len)), dtype=np.float32
        )
        tokens = np.arange(int(self.seq_len), dtype=np.int64)
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                scores[head] = packed[pair_index, 2 * tokens + lane]
        return scores

    def decode_prefix_mask(
        self, occupied: int | None = None, *, dtype=np.float64
    ) -> np.ndarray:
        """Public 1/0 prefix in ``pack_decode_scores`` slot order."""

        occupied = int(self.seq_len if occupied is None else occupied)
        if occupied < 0 or occupied > int(self.seq_len):
            raise ValueError(
                f"occupied={occupied} outside [0, {self.seq_len}]."
            )
        mask = np.zeros(
            (int(self.query_heads), int(self.seq_len)), dtype=np.float32
        )
        mask[:, :occupied] = 1.0
        return self.pack_decode_scores(mask, dtype=dtype)

    def decode_token0_mask(self, *, dtype=np.float64) -> np.ndarray:
        """Select token column 0 in ``pack_decode_scores`` slot order."""

        mask = np.zeros(
            (int(self.query_heads), int(self.seq_len)), dtype=np.float32
        )
        mask[:, 0] = 1.0
        return self.pack_decode_scores(mask, dtype=dtype)

    def decode_softmax_token_rotations(self) -> tuple[int, ...]:
        """Rotate-add tree along the C token axis (stride 2, not Delta ``2S``).

        After the fold, the full row sum lives on token column 0.  Alg2
        broadcasts it back with the negated rotations.
        """

        return tuple(
            2 * (1 << bit)
            for bit in range(int(self.seq_len).bit_length() - 1)
        )

    def decode_softmax_token_broadcast_rotations(self) -> tuple[int, ...]:
        """Copy token column 0 onto every C token column (negative stride 2)."""

        return tuple(-step for step in self.decode_softmax_token_rotations())

    # -- compact Decode score pages merged into one ciphertext -------------
    #
    # A compact score page uses feature rows ``0 .. query_pair_count-1`` of the
    # ``head_dim`` rows; the other rows are free.  Page ``g`` of a multi-page
    # Softmax can therefore live on rows ``g*query_pair_count ..`` of a single
    # ciphertext (slot offset ``g * decode_page_merge_stride``), so one exp /
    # bootstrap / Alg2 round serves every page.

    @property
    def decode_page_merge_stride(self) -> int:
        """Slots between two merged score pages (``query_pair_count`` rows)."""

        return 2 * int(self.seq_len) * int(self.query_pair_count)

    @property
    def decode_page_merge_groups(self) -> int:
        """How many compact score pages fit in one ciphertext (8 for Llama-3)."""

        return int(self.head_dim) // int(self.query_pair_count)

    def decode_page_merge_rotations(self) -> tuple[int, ...]:
        """Merge (``-g*stride``), un-merge (``+g*stride``) and the group
        rotate-add tree (``stride * 2^b``) of the merged-page Softmax."""

        stride = self.decode_page_merge_stride
        groups = self.decode_page_merge_groups
        rotations: list[int] = []
        for group in range(1, groups):
            rotations.append(-stride * group)
            rotations.append(stride * group)
        bit = 1
        while bit < groups:
            rotations.append(stride * bit)
            bit <<= 1
        return tuple(dict.fromkeys(rotations))

    def decode_feature_broadcast_rotations(self) -> tuple[int, ...]:
        """Copy feature row 0 onto every C feature row (negative ``2S``)."""

        stride = 2 * int(self.seq_len)
        return tuple(
            -stride * (1 << bit)
            for bit in range(int(self.head_dim).bit_length() - 1)
        )

    def decode_qk_feature_rotations(self) -> tuple[int, ...]:
        """QK feature-reduce onto Decode score row 0 (positive ``2S``)."""

        stride = 2 * int(self.seq_len)
        return tuple(
            stride * (1 << bit)
            for bit in range(int(self.head_dim).bit_length() - 1)
        )

    def pack_decode_output(
        self, output: np.ndarray, *, dtype=np.float64
    ) -> np.ndarray:
        """Pack a 1-query attention output onto token column 0 of Q_C."""

        output = np.asarray(output, dtype=np.float32)
        heads = int(self.query_heads)
        dim = int(self.head_dim)
        if output.shape == (1, heads, dim):
            token = output[0]
        elif output.shape == (heads, dim):
            token = output
        elif output.shape == (1, heads * dim):
            token = output.reshape(heads, dim)
        elif output.shape == (heads * dim,):
            token = output.reshape(heads, dim)
        else:
            raise ValueError(
                "decode output must be one token of "
                f"{(heads, dim)}, got {output.shape}."
            )
        rows = np.zeros(
            (int(self.seq_len), heads, dim), dtype=np.float32
        )
        rows[0] = token
        return self.pack_query_c(rows, dtype=dtype)

    def pack_query_broadcast_c(
        self, query: np.ndarray, *, dtype=np.float64
    ) -> np.ndarray:
        """Replicate one query token across every C token column for Decode QK."""

        query = np.asarray(query, dtype=np.float32)
        heads = int(self.query_heads)
        dim = int(self.head_dim)
        if query.shape == (heads, dim):
            token = query
        elif query.shape == (1, heads, dim):
            token = query[0]
        elif query.shape == (1, heads * dim):
            token = query.reshape(heads, dim)
        elif query.shape == (heads * dim,):
            token = query.reshape(heads, dim)
        else:
            raise ValueError(
                "broadcast query must be one token of "
                f"{(heads, dim)}, got {query.shape}."
            )
        rows = np.repeat(token[None, :, :], int(self.seq_len), axis=0)
        return self.pack_query_c(rows, dtype=dtype)

    def _require_compact_q(self) -> None:
        if int(self.query_pair_count) > int(self.seq_len):
            raise ValueError(
                "compact Q stores one pair per token column, "
                f"need query_pair_count={self.query_pair_count} "
                f"<= seq_len={self.seq_len}."
            )
        if int(self.query_pair_count) > int(self.head_dim):
            raise ValueError(
                "compact scores store one pair per feature row, "
                f"need query_pair_count={self.query_pair_count} "
                f"<= head_dim={self.head_dim}."
            )

    def _one_query_token(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        heads = int(self.query_heads)
        dim = int(self.head_dim)
        if query.shape == (heads, dim):
            return query
        if query.shape == (1, heads, dim):
            return query[0]
        if query.shape == (1, heads * dim):
            return query.reshape(heads, dim)
        if query.shape == (heads * dim,):
            return query.reshape(heads, dim)
        raise ValueError(
            f"compact query must be one token of {(heads, dim)}, got {query.shape}."
        )

    def pack_compact_query_c(
        self, query: np.ndarray, *, dtype=np.float64
    ) -> np.ndarray:
        """One ciphertext: token column ``pair`` holds Q pair ``pair`` (no broadcast)."""

        self._require_compact_q()
        token = self._one_query_token(query)
        packed = np.zeros((self.payload_slots,), dtype=dtype)
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                for feature in range(int(self.head_dim)):
                    packed[self.c_slot(feature, pair_index, lane)] = token[
                        int(head), feature
                    ]
        return self._repeat_payload(packed.reshape(1, -1))

    def unpack_compact_query_c(self, packed: np.ndarray) -> np.ndarray:
        packed = self._base_payload(packed, pairs=1, name="compact_Q")
        token = np.zeros(
            (int(self.query_heads), int(self.head_dim)), dtype=np.float32
        )
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                for feature in range(int(self.head_dim)):
                    token[int(head), feature] = packed[
                        0, self.c_slot(feature, pair_index, lane)
                    ]
        return token

    def pack_compact_decode_scores(
        self, scores: np.ndarray, *, dtype=np.float64
    ) -> np.ndarray:
        """One ciphertext: feature row ``pair`` holds the 1-query score row."""

        self._require_compact_q()
        scores = np.asarray(scores, dtype=np.float32)
        expected = (int(self.query_heads), int(self.seq_len))
        if scores.shape != expected:
            raise ValueError(
                f"compact scores must have shape {expected}, got {scores.shape}."
            )
        packed = np.zeros((self.payload_slots,), dtype=dtype)
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                for token in range(int(self.seq_len)):
                    packed[self.c_slot(pair_index, token, lane)] = scores[
                        int(head), token
                    ]
        return self._repeat_payload(packed.reshape(1, -1))

    def unpack_compact_decode_scores(self, packed: np.ndarray) -> np.ndarray:
        packed = self._base_payload(packed, pairs=1, name="compact_scores")
        scores = np.zeros(
            (int(self.query_heads), int(self.seq_len)), dtype=np.float32
        )
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                for token in range(int(self.seq_len)):
                    scores[int(head), token] = packed[
                        0, self.c_slot(pair_index, token, lane)
                    ]
        return scores

    def compact_token_column_mask(
        self, token_index: int, *, dtype=np.float64
    ) -> np.ndarray:
        """Select one C token column on every feature row (compact Q isolate)."""

        token_index = int(token_index)
        packed = np.zeros((self.payload_slots,), dtype=dtype)
        for feature in range(int(self.head_dim)):
            for lane in (0, 1):
                packed[self.c_slot(feature, token_index, lane)] = 1.0
        return self._repeat_payload(packed.reshape(1, -1))

    def compact_feature_row_mask(
        self, pair_index: int, *, dtype=np.float64
    ) -> np.ndarray:
        """Select one compact-score feature row on every token column."""

        pair_index = int(pair_index)
        packed = np.zeros((self.payload_slots,), dtype=dtype)
        for token in range(int(self.seq_len)):
            for lane in (0, 1):
                packed[self.c_slot(pair_index, token, lane)] = 1.0
        return self._repeat_payload(packed.reshape(1, -1))

    def compact_decode_prefix_mask(
        self, occupied: int | None = None, *, dtype=np.float64
    ) -> np.ndarray:
        mask = np.zeros(
            (int(self.query_heads), int(self.seq_len)), dtype=np.float32
        )
        occupied = int(self.seq_len if occupied is None else occupied)
        if occupied < 0 or occupied > int(self.seq_len):
            raise ValueError(
                f"occupied={occupied} outside [0, {self.seq_len}]."
            )
        mask[:, :occupied] = 1.0
        return self.pack_compact_decode_scores(mask, dtype=dtype)

    def compact_decode_token0_mask(self, *, dtype=np.float64) -> np.ndarray:
        mask = np.zeros(
            (int(self.query_heads), int(self.seq_len)), dtype=np.float32
        )
        mask[:, 0] = 1.0
        return self.pack_compact_decode_scores(mask, dtype=dtype)

    def compact_q_column_rotations(self) -> tuple[int, ...]:
        """Move compact-Q token column ``p`` to column 0 (``+2p``)."""

        return tuple(
            2 * pair for pair in range(1, int(self.query_pair_count))
        )

    def compact_score_row_rotations(self) -> tuple[int, ...]:
        """Move feature row 0 onto compact-score row ``p`` (``-2 S p``)."""

        stride = 2 * int(self.seq_len)
        return tuple(
            -stride * pair for pair in range(1, int(self.query_pair_count))
        )

    def _pack_delta(self, values: np.ndarray, *, name: str, dtype=np.float64) -> np.ndarray:
        values = self._logical_heads(values, heads=int(self.key_value_heads), name=name)
        packed = np.zeros((self.key_value_pair_count, self.payload_slots), dtype=dtype)
        queries = np.arange(int(self.seq_len), dtype=np.int64)
        features = np.arange(int(self.head_dim), dtype=np.int64)
        source_tokens = (queries[None, :] + features[:, None]) % int(self.seq_len)
        slots_base = 2 * (features[:, None] * int(self.seq_len) + queries[None, :])
        for pair_index, heads in enumerate(self.key_value_head_pairs()):
            for lane, head in enumerate(heads):
                packed[pair_index, (slots_base + lane).reshape(-1)] = values[source_tokens, head, features[:, None]].reshape(-1)
        return self._repeat_payload(packed)


    def pack_key_delta(self, key: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        return self._pack_delta(key, name="key", dtype=dtype)

    def pack_key_delta_feature_folded(
        self, key: np.ndarray, *, dtype=np.complex128
    ) -> np.ndarray:
        """Pack ``K_Delta[d] - i*K_Delta[d+d_h/2]`` for folded QK."""

        if int(self.head_dim) % 2:
            raise ValueError("complex K feature folding requires even head_dim.")
        values = self._logical_heads(
            key, heads=int(self.key_value_heads), name="key"
        )
        packed = np.zeros(
            (self.key_value_pair_count, self.payload_slots), dtype=dtype
        )
        queries = np.arange(int(self.seq_len), dtype=np.int64)
        features = np.arange(int(self.head_dim), dtype=np.int64)
        paired_features = (features + int(self.head_dim) // 2) % int(
            self.head_dim
        )
        source_tokens = (queries[None, :] + features[:, None]) % int(
            self.seq_len
        )
        slots_base = 2 * (
            features[:, None] * int(self.seq_len) + queries[None, :]
        )
        for pair_index, heads in enumerate(self.key_value_head_pairs()):
            for lane, head in enumerate(heads):
                real = values[source_tokens, head, features[:, None]]
                imaginary = values[
                    source_tokens, head, paired_features[:, None]
                ]
                packed[pair_index, (slots_base + lane).reshape(-1)] = (
                    real - 1j * imaginary
                ).reshape(-1)
        return self._repeat_payload(packed)


    def pack_value_delta(self, value: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        return self._pack_delta(value, name="value", dtype=dtype)


    def pack_score_delta(self, scores: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        """Pack logical scores ``[Q heads, query, key]`` as ``S_Delta[d,q]``."""

        scores = np.asarray(scores, dtype=np.float32)
        expected = (int(self.query_heads), int(self.seq_len), int(self.seq_len))
        if scores.shape != expected:
            raise ValueError(f"scores must have shape {expected}, got {scores.shape}.")
        packed = np.zeros((self.query_pair_count, self.payload_slots), dtype=dtype)
        queries = np.arange(int(self.seq_len), dtype=np.int64)
        deltas = np.arange(int(self.seq_len), dtype=np.int64)
        keys = (queries[None, :] + deltas[:, None]) % int(self.seq_len)
        slots_base = 2 * (deltas[:, None] * int(self.seq_len) + queries[None, :])
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                packed[pair_index, (slots_base + lane).reshape(-1)] = scores[head, queries[None, :], keys].reshape(-1)
        return self._repeat_payload(packed)

    def unpack_score_delta(self, packed: np.ndarray) -> np.ndarray:
        packed = self._base_payload(packed, pairs=self.query_pair_count, name="S_Delta/P_Delta")
        scores = np.zeros(
            (int(self.query_heads), int(self.seq_len), int(self.seq_len)), dtype=np.float32
        )
        queries = np.arange(int(self.seq_len), dtype=np.int64)
        deltas = np.arange(int(self.seq_len), dtype=np.int64)
        keys = (queries[None, :] + deltas[:, None]) % int(self.seq_len)
        slots_base = 2 * (deltas[:, None] * int(self.seq_len) + queries[None, :])
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                scores[head, queries[None, :], keys] = packed[
                    pair_index, (slots_base + lane).reshape(-1)
                ].reshape(int(self.seq_len), int(self.seq_len))
        return scores

    def pack_prefix_scores_prefill(
        self, scores: np.ndarray, *, dtype=np.float64
    ) -> np.ndarray:
        """Pack prefix scores ``[Hq, Q, k]`` into C form: feature ``j``, token ``query``.

        Occupies feature rows ``0..k-1`` on the same 16 pair-ciphers as ``Q_C``.
        Content scores stay in ``S_Δ``; this object is never concatenated with them.
        """

        scores = np.asarray(scores, dtype=np.float32)
        expected_hq = int(self.query_heads)
        expected_q = int(self.seq_len)
        if scores.ndim != 3 or scores.shape[0] != expected_hq or scores.shape[1] != expected_q:
            raise ValueError(
                "prefix scores must have shape "
                f"({expected_hq}, {expected_q}, k), got {scores.shape}."
            )
        prefix_k = int(scores.shape[2])
        if not 1 <= prefix_k <= int(self.head_dim):
            raise ValueError(
                f"prefix_k={prefix_k} must lie in [1, {self.head_dim}]."
            )
        packed = np.zeros((self.query_pair_count, self.payload_slots), dtype=dtype)
        queries = np.arange(expected_q, dtype=np.int64)
        seq = int(self.seq_len)
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                for prefix_j in range(prefix_k):
                    slots = 2 * (prefix_j * seq + queries) + lane
                    packed[pair_index, slots] = scores[int(head), :, prefix_j]
        return self._repeat_payload(packed)

    def unpack_prefix_scores_prefill(
        self, packed: np.ndarray, *, prefix_k: int
    ) -> np.ndarray:
        prefix_k = int(prefix_k)
        if not 1 <= prefix_k <= int(self.head_dim):
            raise ValueError(
                f"prefix_k={prefix_k} must lie in [1, {self.head_dim}]."
            )
        packed = self._base_payload(
            packed, pairs=self.query_pair_count, name="S_prefix/P_prefix"
        )
        scores = np.zeros(
            (int(self.query_heads), int(self.seq_len), prefix_k), dtype=np.float32
        )
        queries = np.arange(int(self.seq_len), dtype=np.int64)
        seq = int(self.seq_len)
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                for prefix_j in range(prefix_k):
                    slots = 2 * (prefix_j * seq + queries) + lane
                    scores[int(head), :, prefix_j] = packed[pair_index, slots]
        return scores

    def prefix_compact_row_count(self, prefix_k: int) -> int:
        """Rows needed to multiplex every query pair's Prefix partition."""

        prefix_k = int(prefix_k)
        if not 1 <= prefix_k <= int(self.head_dim):
            raise ValueError(
                f"prefix_k={prefix_k} must lie in [1, {self.head_dim}]."
            )
        rows = int(self.query_pair_count) * prefix_k
        if rows > int(self.head_dim):
            raise ValueError(
                "compact Prefix scores do not fit in one ciphertext: "
                f"query_pairs={self.query_pair_count}, prefix_k={prefix_k}, "
                f"rows={rows}, capacity={self.head_dim}."
            )
        return rows

    def pack_prefix_scores_compact(
        self, scores: np.ndarray, *, dtype=np.float64
    ) -> np.ndarray:
        """Pack every Prefix query pair into one C-grid ciphertext.

        Pair ``p`` owns rows ``p*k .. (p+1)*k-1``.  Tokens remain queries and
        the two lanes remain the two query heads in that pair.  Production
        ``k=2`` therefore occupies 32 of 128 rows instead of materializing 16
        mostly-empty ciphertexts.
        """

        scores = np.asarray(scores, dtype=np.float32)
        expected_hq = int(self.query_heads)
        expected_q = int(self.seq_len)
        if (
            scores.ndim != 3
            or scores.shape[0] != expected_hq
            or scores.shape[1] != expected_q
        ):
            raise ValueError(
                "prefix scores must have shape "
                f"({expected_hq}, {expected_q}, k), got {scores.shape}."
            )
        prefix_k = int(scores.shape[2])
        self.prefix_compact_row_count(prefix_k)
        packed = np.zeros((1, self.payload_slots), dtype=dtype)
        queries = np.arange(expected_q, dtype=np.int64)
        seq = int(self.seq_len)
        for pair_index, heads in enumerate(self.query_head_pairs()):
            base_row = int(pair_index) * prefix_k
            for lane, head in enumerate(heads):
                for prefix_j in range(prefix_k):
                    slots = 2 * ((base_row + prefix_j) * seq + queries) + lane
                    packed[0, slots] = scores[int(head), :, prefix_j]
        return self._repeat_payload(packed)

    def unpack_prefix_scores_compact(
        self, packed: np.ndarray, *, prefix_k: int
    ) -> np.ndarray:
        """Inverse of :meth:`pack_prefix_scores_compact`."""

        prefix_k = int(prefix_k)
        self.prefix_compact_row_count(prefix_k)
        packed = self._base_payload(
            packed, pairs=1, name="compact S_prefix/P_prefix"
        )
        scores = np.zeros(
            (int(self.query_heads), int(self.seq_len), prefix_k),
            dtype=np.float32,
        )
        queries = np.arange(int(self.seq_len), dtype=np.int64)
        seq = int(self.seq_len)
        for pair_index, heads in enumerate(self.query_head_pairs()):
            base_row = int(pair_index) * prefix_k
            for lane, head in enumerate(heads):
                for prefix_j in range(prefix_k):
                    slots = 2 * ((base_row + prefix_j) * seq + queries) + lane
                    scores[int(head), :, prefix_j] = packed[0, slots]
        return scores

    def c_feature_row_mask(self, feature: int, *, dtype=np.float64) -> np.ndarray:
        """1 on one C-form feature row (all tokens, both lanes) of every Q pair."""

        feature = int(feature)
        if not 0 <= feature < int(self.head_dim):
            raise ValueError(
                f"feature={feature} is outside [0, {self.head_dim})."
            )
        packed = np.zeros((self.query_pair_count, self.payload_slots), dtype=dtype)
        seq = int(self.seq_len)
        tokens = np.arange(seq, dtype=np.int64)
        for pair_index in range(self.query_pair_count):
            for lane in (0, 1):
                slots = 2 * (feature * seq + tokens) + lane
                packed[pair_index, slots] = 1.0
        return self._repeat_payload(packed)

    def c_prefix_score_mask(self, prefix_k: int, *, dtype=np.float64) -> np.ndarray:
        """1 on C-form prefix score rows ``feature=0..k-1`` (all queries)."""

        prefix_k = int(prefix_k)
        if not 1 <= prefix_k <= int(self.head_dim):
            raise ValueError(
                f"prefix_k={prefix_k} must lie in [1, {self.head_dim}]."
            )
        packed = np.zeros((self.query_pair_count, self.payload_slots), dtype=dtype)
        seq = int(self.seq_len)
        queries = np.arange(seq, dtype=np.int64)
        for pair_index in range(self.query_pair_count):
            for lane in (0, 1):
                for prefix_j in range(prefix_k):
                    slots = 2 * (prefix_j * seq + queries) + lane
                    packed[pair_index, slots] = 1.0
        return self._repeat_payload(packed)

    def c_content_score_causal_mask(self, *, dtype=np.float64) -> np.ndarray:
        """C-form content scores ``feature=key``, ``token=query``: ``key <= query``."""

        packed = np.zeros((self.query_pair_count, self.payload_slots), dtype=dtype)
        seq = int(self.seq_len)
        queries = np.arange(seq, dtype=np.int64)
        for pair_index in range(self.query_pair_count):
            for lane in (0, 1):
                for key in range(seq):
                    slots = 2 * (key * seq + queries) + lane
                    packed[pair_index, slots] = (key <= queries).astype(dtype)
        return self._repeat_payload(packed)

    def pack_output_c(self, output: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        return self.pack_query_c(output, dtype=dtype)

    def unpack_output_c(self, packed: np.ndarray) -> np.ndarray:
        return self.unpack_query_c(packed)

    def pack_qkv(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, *, dtype=np.float64) -> AttentionQKVPayload:
        return AttentionQKVPayload(
            query_c=self.pack_query_c(query, dtype=dtype),
            key_delta=self.pack_key_delta(key, dtype=dtype),
            value_delta=self.pack_value_delta(value, dtype=dtype),
        )
