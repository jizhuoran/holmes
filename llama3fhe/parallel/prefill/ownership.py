"""Static data ownership for the one-process-per-GPU (T2) runtime.

Everything in the layer is indexed by *global* ciphertext positions of the
feature-major (FM) stream and by global attention query pairs.  The mapping
from those indices to ranks is fixed for the whole run and is built from one
number, ``ciphers_per_rank = cipher_count / world_size`` (an even number):

* rank ``r`` owns the adjacent FM ciphertexts ``r*n .. r*n + n-1``
  (``n = ciphers_per_rank``).  With the production layout (``S=128``,
  ``D=4096``, 32768 slots, 16 FM ciphertexts) that is tokens
  ``8nr .. 8n(r+1)-1``; QKV/MLP token shards, RMSNorm ``cipher_indices`` and
  residual adds all use exactly these indices
  (:func:`llama3fhe.operators.linear.shard.qkv_source_shard.qkv_token_shard_plan`,
  :func:`llama3fhe.operators.linear.shard.mlp_token_shard.mlp_token_shard_plan`);
* inside attention rank ``r`` owns the global query pairs ``r*n .. r*n + n-1``
  (16 pairs for 32 query heads, so the query-pair ownership coincides with the
  FM cipher ownership) and, after the PairFM reduce-scatter, the ``n/2``
  ``W_O`` carriers ``r*n/2 .. (r+1)*n/2 - 1``
  (:func:`llama3fhe.operators.attention.distributed.distributed_output.build_pairfm_reduce_scatter_plan`).

A *T2 slot* is the unit the PairFM algebra is written in: one adjacent FM pair
``(2s, 2s+1)``, the query pairs ``(2s, 2s+1)`` and the ``W_O`` carrier ``s``
(``s`` = logical slot, 8 of them in the production layout).  A rank owns
``slots_per_rank = n / 2`` consecutive slots (:meth:`RankOwnership.slot_indices`):
one slot per rank on an 8-card host (``world_size=8``), two on a 4-card host
(``world_size=4``, one process per card).  The station code works on the
rank's whole cipher list; only the PairFM source half iterates over slots.

The GQA roles are derived, not stored: query pair ``p`` reads key pair
``p // gqa_ratio`` (:meth:`AttentionPairLayout.query_head_pairs`) and key pair
``k`` lives in the complex V carrier ``k // 2`` (``2*Re`` = V pair ``2c``,
``2*Im`` = V pair ``2c+1``).  With the production ratio 4 and two query pairs
per rank (8 ranks), rank ``r`` needs key pair ``r // 2`` (shared with its
neighbour) and V carrier ``r // 4`` (shared by four ranks); with four query
pairs per rank (4 ranks) rank ``r`` needs exactly key pair ``r`` and V carrier
``r // 2``.  The T2 phase-1 attention input duplicates the K work on every
consumer and builds each V carrier as the sum of one own-source partial per
rank (:func:`key_pairs_for_rank`, :func:`value_carriers_for_rank`,
:func:`value_group_ranks`).

This module is pure Python so the mapping can be unit-tested on the CPU and
so both the launcher (host side) and the workers agree on it without
communication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


def _check_rank(rank: int, world_size: int) -> int:
    if isinstance(rank, bool):
        raise TypeError("rank must be an integer, not bool")
    rank = int(rank)
    if rank < 0 or rank >= int(world_size):
        raise IndexError(f"rank {rank} is outside [0, {world_size})")
    return rank


@dataclass(frozen=True)
class RankOwnership:
    """What one rank owns for the whole layer."""

    rank: int
    #: Global FM ciphertext indices resident on this rank (adjacent, even count).
    cipher_indices: tuple[int, ...]
    #: First token and token count carried by ``cipher_indices``.
    token_start: int
    token_count: int
    #: Global attention query-pair indices this rank finishes (QK/Softmax/PV).
    query_pair_indices: tuple[int, ...]
    #: PairFM target carriers / W_O carriers owned after the reduce-scatter
    #: (carrier ``c`` holds FM ciphers ``2c, 2c+1``; one per owned slot).
    target_carrier_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        ciphers = tuple(int(v) for v in self.cipher_indices)
        if not ciphers or len(ciphers) % 2:
            raise ValueError("a rank owns a positive even number of FM ciphertexts")
        if ciphers != tuple(range(ciphers[0], ciphers[0] + len(ciphers))):
            raise ValueError(f"owned FM ciphertexts must be adjacent, got {ciphers}")
        if ciphers[0] % 2:
            raise ValueError(f"owned FM ciphertexts must start at an even index, got {ciphers}")
        carriers = tuple(int(v) for v in self.target_carrier_indices)
        if not carriers:
            # derived: carrier c <-> FM pair (2c, 2c+1)
            carriers = tuple(range(ciphers[0] // 2, (ciphers[-1] + 1) // 2))
        if len(carriers) != len(ciphers) // 2:
            raise ValueError(
                f"{len(ciphers)} owned ciphertexts need {len(ciphers) // 2} target carriers, got {carriers}"
            )
        object.__setattr__(self, "cipher_indices", ciphers)
        object.__setattr__(self, "query_pair_indices", tuple(int(v) for v in self.query_pair_indices))
        object.__setattr__(self, "target_carrier_indices", carriers)

    @property
    def token_stop(self) -> int:
        return int(self.token_start) + int(self.token_count)

    @property
    def ciphers_per_rank(self) -> int:
        return len(self.cipher_indices)

    @property
    def slots_per_rank(self) -> int:
        """T2 slots (adjacent FM pairs / W_O carriers) owned by this rank."""

        return len(self.cipher_indices) // 2

    @property
    def slot_indices(self) -> tuple[int, ...]:
        """Logical slot indices ``s`` owned by this rank (FM pair ``2s, 2s+1``).

        Identical to :attr:`target_carrier_indices` in the production layout;
        the PairFM source half is written per slot (``source_logical_rank``).
        """

        return tuple(int(v) // 2 for v in self.cipher_indices[0::2])

    @property
    def cipher_pair(self) -> tuple[int, int]:
        if len(self.cipher_indices) != 2:
            raise ValueError("cipher_pair requires exactly two owned ciphertexts (one slot per rank)")
        return int(self.cipher_indices[0]), int(self.cipher_indices[1])

    @property
    def target_carrier_index(self) -> int:
        """The single W_O carrier of a one-slot rank (``world_size == 8``)."""

        if len(self.target_carrier_indices) != 1:
            raise ValueError(
                "target_carrier_index requires exactly one owned carrier; "
                f"rank {self.rank} owns {self.target_carrier_indices} (use target_carrier_indices)"
            )
        return int(self.target_carrier_indices[0])

    def cipher_pairs(self) -> tuple[tuple[int, int], ...]:
        """Owned FM ciphertexts two at a time (one adjacent pair per slot)."""

        return tuple(
            (int(self.cipher_indices[i]), int(self.cipher_indices[i + 1]))
            for i in range(0, len(self.cipher_indices), 2)
        )

    def slot_query_pairs(self) -> tuple[tuple[int, int], ...]:
        """Owned query pairs grouped by slot (``(2s, 2s+1)`` for slot ``s``).

        Requires the production identity ``query_pair_count == cipher_count``
        (16 query pairs for 16 FM ciphertexts), which is what makes the
        PairFM source shards (two adjacent query pairs each) coincide with the
        owned slots.
        """

        pairs = tuple(int(v) for v in self.query_pair_indices)
        if len(pairs) != len(self.cipher_indices):
            raise ValueError(
                f"rank {self.rank}: {len(pairs)} query pairs do not map one-to-one onto "
                f"{len(self.cipher_indices)} FM ciphertexts; slot grouping is undefined"
            )
        grouped = tuple((pairs[i], pairs[i + 1]) for i in range(0, len(pairs), 2))
        for slot, group in zip(self.slot_indices, grouped, strict=True):
            if group != (2 * slot, 2 * slot + 1):
                raise ValueError(
                    f"rank {self.rank}: slot {slot} expects query pairs {(2 * slot, 2 * slot + 1)}, got {group}"
                )
        return grouped

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": int(self.rank),
            "cipher_indices": [int(v) for v in self.cipher_indices],
            "token_start": int(self.token_start),
            "token_count": int(self.token_count),
            "query_pair_indices": [int(v) for v in self.query_pair_indices],
            "target_carrier_indices": [int(v) for v in self.target_carrier_indices],
            "slot_indices": [int(v) for v in self.slot_indices],
        }


@dataclass(frozen=True)
class WorldOwnership:
    """The complete static assignment for ``world_size`` ranks."""

    world_size: int
    ciphers_per_rank: int
    tokens_per_cipher: int
    query_pairs_per_rank: int
    ranks: tuple[RankOwnership, ...]

    def __post_init__(self) -> None:
        if len(self.ranks) != int(self.world_size):
            raise ValueError("ranks must contain one entry per rank")
        for expected, item in enumerate(self.ranks):
            if int(item.rank) != expected:
                raise ValueError("ranks must be ordered 0..world_size-1")

    def __iter__(self) -> Iterator[RankOwnership]:
        return iter(self.ranks)

    def __len__(self) -> int:
        return len(self.ranks)

    def __getitem__(self, rank: int) -> RankOwnership:
        return self.ranks[_check_rank(rank, self.world_size)]

    @property
    def cipher_count(self) -> int:
        return int(self.world_size) * int(self.ciphers_per_rank)

    @property
    def slots_per_rank(self) -> int:
        return int(self.ciphers_per_rank) // 2

    @property
    def slot_count(self) -> int:
        """Logical T2 slots (= PairFM source shards = W_O carriers)."""

        return self.cipher_count // 2

    @property
    def query_pair_count(self) -> int:
        return int(self.world_size) * int(self.query_pairs_per_rank)

    @property
    def seq_len(self) -> int:
        return self.cipher_count * int(self.tokens_per_cipher)

    def owner_of_cipher(self, cipher_index: int) -> int:
        index = int(cipher_index)
        if index < 0 or index >= self.cipher_count:
            raise IndexError(
                f"cipher index {index} is outside [0, {self.cipher_count})"
            )
        return index // int(self.ciphers_per_rank)

    def owner_of_query_pair(self, pair_index: int) -> int:
        index = int(pair_index)
        if index < 0 or index >= self.query_pair_count:
            raise IndexError(
                f"query pair {index} is outside [0, {self.query_pair_count})"
            )
        return index // int(self.query_pairs_per_rank)

    def owner_of_target_carrier(self, carrier_index: int) -> int:
        index = int(carrier_index)
        if index < 0 or index >= self.slot_count:
            raise IndexError(
                f"target carrier {index} is outside [0, {self.slot_count})"
            )
        return index // self.slots_per_rank

    def owner_of_slot(self, slot_index: int) -> int:
        return self.owner_of_target_carrier(slot_index)

    def peers(self, rank: int) -> tuple[int, ...]:
        rank = _check_rank(rank, self.world_size)
        return tuple(other for other in range(int(self.world_size)) if other != rank)

    def to_dict(self) -> dict[str, object]:
        return {
            "world_size": int(self.world_size),
            "ciphers_per_rank": int(self.ciphers_per_rank),
            "tokens_per_cipher": int(self.tokens_per_cipher),
            "query_pairs_per_rank": int(self.query_pairs_per_rank),
            "ranks": [item.to_dict() for item in self.ranks],
        }


def cipher_indices_for_rank(
    rank: int, *, world_size: int = 8, ciphers_per_rank: int = 2
) -> tuple[int, ...]:
    rank = _check_rank(rank, world_size)
    per = int(ciphers_per_rank)
    return tuple(range(rank * per, rank * per + per))


def owner_of_cipher(
    cipher_index: int, *, world_size: int = 8, ciphers_per_rank: int = 2
) -> int:
    index = int(cipher_index)
    total = int(world_size) * int(ciphers_per_rank)
    if index < 0 or index >= total:
        raise IndexError(f"cipher index {index} is outside [0, {total})")
    return index // int(ciphers_per_rank)


def attention_query_pairs_for_rank(
    rank: int, *, world_size: int = 8, query_pairs_per_rank: int = 2
) -> tuple[int, ...]:
    rank = _check_rank(rank, world_size)
    per = int(query_pairs_per_rank)
    return tuple(range(rank * per, rank * per + per))


def owner_of_query_pair(
    pair_index: int, *, world_size: int = 8, query_pairs_per_rank: int = 2
) -> int:
    index = int(pair_index)
    total = int(world_size) * int(query_pairs_per_rank)
    if index < 0 or index >= total:
        raise IndexError(f"query pair {index} is outside [0, {total})")
    return index // int(query_pairs_per_rank)


def key_pairs_for_query_pairs(query_pair_indices, *, gqa_ratio: int = 4) -> tuple[int, ...]:
    """Key/value pair indices read by the given query pairs (``p // gqa_ratio``)."""

    ratio = int(gqa_ratio)
    if ratio <= 0:
        raise ValueError("gqa_ratio must be positive")
    return tuple(sorted({int(pair) // ratio for pair in query_pair_indices}))


def value_carriers_for_key_pairs(key_pair_indices) -> tuple[int, ...]:
    """Complex V carriers holding the given key/value pairs (``k // 2``)."""

    return tuple(sorted({int(pair) // 2 for pair in key_pair_indices}))


def key_pairs_for_rank(ownership: "RankOwnership", *, gqa_ratio: int = 4) -> tuple[int, ...]:
    return key_pairs_for_query_pairs(ownership.query_pair_indices, gqa_ratio=gqa_ratio)


def value_carriers_for_rank(ownership: "RankOwnership", *, gqa_ratio: int = 4) -> tuple[int, ...]:
    return value_carriers_for_key_pairs(key_pairs_for_rank(ownership, gqa_ratio=gqa_ratio))


def value_group_ranks(world: "WorldOwnership", carrier: int, *, gqa_ratio: int = 4) -> tuple[int, ...]:
    """Ranks that consume V carrier ``carrier`` (they receive one partial per rank)."""

    carrier = int(carrier)
    return tuple(
        item.rank
        for item in world
        if carrier in value_carriers_for_rank(item, gqa_ratio=gqa_ratio)
    )


def build_world_ownership(
    *,
    world_size: int = 8,
    ciphers_per_rank: int = 2,
    tokens_per_cipher: int = 8,
    query_pair_count: int | None = None,
) -> WorldOwnership:
    """Build the static T2 assignment.

    ``query_pair_count`` defaults to ``world_size * ciphers_per_rank`` (16 for
    the production layout: 32 query heads -> 16 pairs), which is what keeps
    the attention query-pair ownership identical to the FM cipher ownership
    and makes the PairFM reduce-scatter ``ciphers_per_rank / 2`` carriers per
    rank (one on 8 ranks, two on 4 ranks).
    """

    world_size = int(world_size)
    ciphers_per_rank = int(ciphers_per_rank)
    tokens_per_cipher = int(tokens_per_cipher)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if ciphers_per_rank <= 0 or ciphers_per_rank % 2:
        raise ValueError("ciphers_per_rank must be a positive even number")
    if tokens_per_cipher <= 0:
        raise ValueError("tokens_per_cipher must be positive")
    pairs = (
        world_size * ciphers_per_rank
        if query_pair_count is None
        else int(query_pair_count)
    )
    if pairs <= 0 or pairs % world_size:
        raise ValueError(
            f"query_pair_count {pairs} must be a positive multiple of world_size {world_size}"
        )
    pairs_per_rank = pairs // world_size
    slots_per_rank = ciphers_per_rank // 2
    ranks = tuple(
        RankOwnership(
            rank=rank,
            cipher_indices=cipher_indices_for_rank(
                rank, world_size=world_size, ciphers_per_rank=ciphers_per_rank
            ),
            token_start=rank * ciphers_per_rank * tokens_per_cipher,
            token_count=ciphers_per_rank * tokens_per_cipher,
            query_pair_indices=attention_query_pairs_for_rank(
                rank, world_size=world_size, query_pairs_per_rank=pairs_per_rank
            ),
            # carrier c holds FM pair (2c, 2c+1): the rank's slots
            target_carrier_indices=tuple(range(rank * slots_per_rank, (rank + 1) * slots_per_rank)),
        )
        for rank in range(world_size)
    )
    return WorldOwnership(
        world_size=world_size,
        ciphers_per_rank=ciphers_per_rank,
        tokens_per_cipher=tokens_per_cipher,
        query_pairs_per_rank=pairs_per_rank,
        ranks=ranks,
    )


def ownership_from_layouts(
    feature_layout, attention_layout=None, *, world_size: int = 8
) -> WorldOwnership:
    """Derive the assignment from the production layout objects.

    ``feature_layout`` is a :class:`llama3fhe.layouts.feature_major.FeatureMajorPrefillLayout`
    (``cipher_count``, ``tokens_per_cipher``); ``attention_layout`` an optional
    :class:`llama3fhe.layouts.attention.AttentionPairLayout`
    (``query_pair_count``).  Only duck-typed attributes are used so the
    function stays importable without the layout modules.
    """

    cipher_count = int(feature_layout.cipher_count)
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if cipher_count % world_size or (cipher_count // world_size) % 2:
        raise ValueError(
            f"{cipher_count} FM ciphertexts do not split into an even number per rank over "
            f"{world_size} ranks (world_size must divide {cipher_count // 2})"
        )
    pairs = None if attention_layout is None else int(attention_layout.query_pair_count)
    return build_world_ownership(
        world_size=world_size,
        ciphers_per_rank=cipher_count // world_size,
        tokens_per_cipher=int(feature_layout.tokens_per_cipher),
        query_pair_count=pairs,
    )


__all__ = [
    "RankOwnership",
    "WorldOwnership",
    "attention_query_pairs_for_rank",
    "build_world_ownership",
    "cipher_indices_for_rank",
    "key_pairs_for_query_pairs",
    "key_pairs_for_rank",
    "owner_of_cipher",
    "owner_of_query_pair",
    "ownership_from_layouts",
    "value_carriers_for_key_pairs",
    "value_carriers_for_rank",
    "value_group_ranks",
]
