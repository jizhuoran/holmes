"""Static data ownership for the multi-card PairFold128 *decode* step (T2-Decode).

Decode moves one token through the layer: the hidden state is a single
feature-major ciphertext (``FeatureMajorPrefillLayout(seq_len=1)``), so the
prefill ownership (:mod:`llama3fhe.parallel.prefill.ownership`, one FM cipher slice
per rank) has nothing to hand out.  The decode step is instead split along
two axes that are already independent inside the single-card kernels:

* **query streams** -- the PairFold attention core iterates over
  ``(group, query_offset)`` pairs (``key_value_groups * gqa_ratio = 8`` with
  the production layout).  Every stream reads the *whole* KV cache but writes
  its own segment of the compact score page and contributes one folded-O
  partial that the kernel sums.  Rank ``r`` finishes the streams
  ``r*8/W .. (r+1)*8/W - 1`` (``W`` = world size, ``W | 8``), so the rank's
  folded-O is a *partial* the ranks add up after the attention core;
* **BSGS diagonal chunks** -- the linear operators (QKV, MLP gate/up/down)
  encode/multiply block-diagonal chunks of the packed weight and the partial
  products of disjoint chunk sets add up to the full product (same level /
  scale after the operator's own rescale).  Rank ``r`` computes the chunk
  indices ``r*C/W .. (r+1)*C/W - 1`` (``C`` chunks, ``W | C``).

For the MLP the intermediate features are also paged (4 pages of 4096 with
the production layout): with ``W <= pages`` a rank owns whole pages and only
its down-projection partial has to be reduced; with ``W > pages`` the ranks
sharing a page split the gate/up and down chunks of that page
(:attr:`DecodeRankOwnership.mlp_chunk_split`) and exchange their gate/up
partials with the page partners before the SiLU.

Everything else in the layer (RMSNorm + bootstrap, KV extraction and cache
append, softmax on the rank's own score segment, W_O, residual adds) is
computed redundantly on every rank: every rank therefore holds the same
hidden state and the same KV cache at the end of the layer, which is exactly
the input of the next layer / the next token.

Pure Python: unit-tested on the CPU, and the launcher and the workers agree
on it without communication.
"""

from __future__ import annotations

from dataclasses import dataclass

#: ``(key_value_groups, gqa_ratio)`` of the production PairFold128 layout.
DEFAULT_KEY_VALUE_GROUPS = 2
DEFAULT_GQA_RATIO = 4
#: Intermediate MLP pages of the production layout (14336 features / 4096 per page).
DEFAULT_MLP_PAGES = 4


def _check_rank(rank: int, world_size: int) -> tuple[int, int]:
    if isinstance(rank, bool) or isinstance(world_size, bool):
        raise TypeError("rank and world_size must be integers, not bool")
    rank, world_size = int(rank), int(world_size)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise IndexError(f"rank {rank} is outside [0, {world_size})")
    return rank, world_size


def contiguous_shard(count: int, position: int, k: int) -> tuple[int, ...]:
    """Indices ``position*count/k .. (position+1)*count/k - 1`` (``k | count``)."""

    count, position, k = int(count), int(position), int(k)
    if k <= 0 or position < 0 or position >= k:
        raise ValueError(f"position {position} outside a group of {k}")
    if count % k:
        raise ValueError(f"{count} items do not split evenly over {k} ranks")
    width = count // k
    return tuple(range(position * width, (position + 1) * width))


@dataclass(frozen=True)
class DecodeRankOwnership:
    """What one rank computes in the decode step (everything else is redundant)."""

    rank: int
    world_size: int
    key_value_groups: int = DEFAULT_KEY_VALUE_GROUPS
    gqa_ratio: int = DEFAULT_GQA_RATIO
    mlp_pages: int = DEFAULT_MLP_PAGES

    def __post_init__(self) -> None:
        rank, world_size = _check_rank(self.rank, self.world_size)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "world_size", world_size)
        streams = int(self.key_value_groups) * int(self.gqa_ratio)
        if streams <= 0 or streams % world_size:
            raise ValueError(
                f"{streams} query streams ({self.key_value_groups} groups x gqa {self.gqa_ratio}) "
                f"do not split evenly over {world_size} ranks"
            )
        pages = int(self.mlp_pages)
        if pages <= 0:
            raise ValueError("mlp_pages must be positive")
        if world_size <= pages:
            if pages % world_size:
                raise ValueError(f"{pages} MLP pages do not split evenly over {world_size} ranks")
        elif world_size % pages:
            raise ValueError(f"{world_size} ranks do not split evenly over {pages} MLP pages")

    # ------------------------------------------------------------ attention
    @property
    def stream_count(self) -> int:
        return int(self.key_value_groups) * int(self.gqa_ratio)

    @property
    def streams_per_rank(self) -> int:
        return self.stream_count // int(self.world_size)

    @property
    def query_streams(self) -> tuple[tuple[int, int], ...]:
        """``(group, query_offset)`` pairs this rank finishes (QK, softmax segment, PV)."""

        return tuple(
            divmod(stream, int(self.gqa_ratio))
            for stream in contiguous_shard(self.stream_count, self.rank, self.world_size)
        )

    @property
    def is_full_attention(self) -> bool:
        return int(self.world_size) == 1

    # ------------------------------------------------------------ KV refresh
    def kv_refresh_indices(self, cipher_count: int = 4) -> tuple[int, ...]:
        """K/V ciphertexts of the new token this rank bootstraps before the cache append.

        The limb drop of the token (``K0, K1, V0, V1`` -> ``kv_limbs``) is one
        independent bootstrap per ciphertext; rank ``r`` refreshes the indices
        ``i`` with ``i % W == r`` and the ranks all-gather the results (ranks
        beyond ``cipher_count`` contribute nothing).  Single rank: all of them.
        """

        count = int(cipher_count)
        if count <= 0:
            raise ValueError(f"cipher_count must be positive, got {count}")
        return tuple(i for i in range(count) if i % int(self.world_size) == int(self.rank))

    # --------------------------------------------------------------- linear
    def chunk_indices(self, chunk_count: int) -> tuple[int, ...]:
        """BSGS chunk indices of this rank for an operator with ``chunk_count`` chunks."""

        return contiguous_shard(chunk_count, self.rank, self.world_size)

    # ------------------------------------------------------------------ MLP
    @property
    def mlp_chunk_split(self) -> int:
        """Ranks sharing every MLP page (1 when a rank owns whole pages)."""

        world_size, pages = int(self.world_size), int(self.mlp_pages)
        return 1 if world_size <= pages else world_size // pages

    @property
    def mlp_page_indices(self) -> tuple[int, ...]:
        """Intermediate pages this rank evaluates (possibly shared with partners)."""

        world_size, pages = int(self.world_size), int(self.mlp_pages)
        if world_size <= pages:
            return contiguous_shard(pages, self.rank, world_size)
        return (self.rank // self.mlp_chunk_split,)

    @property
    def mlp_chunk_index(self) -> int:
        """Position of this rank among the partners of its page (0 when unsplit)."""

        return 0 if self.mlp_chunk_split == 1 else self.rank % self.mlp_chunk_split

    @property
    def mlp_partners(self) -> tuple[int, ...]:
        """Ranks (including self, ascending) that share this rank's MLP page."""

        split = self.mlp_chunk_split
        if split == 1:
            return (self.rank,)
        first = (self.rank // split) * split
        return tuple(range(first, first + split))

    def mlp_chunk_indices(self, chunk_count: int) -> tuple[int, ...]:
        """Chunks of the shared page this rank computes (all of them when unsplit)."""

        return contiguous_shard(chunk_count, self.mlp_chunk_index, self.mlp_chunk_split)


def decode_world(world_size: int, **layout) -> tuple[DecodeRankOwnership, ...]:
    """Ownership of every rank (the launcher's view)."""

    return tuple(DecodeRankOwnership(rank, world_size, **layout) for rank in range(int(world_size)))


__all__ = [
    "DEFAULT_GQA_RATIO",
    "DEFAULT_KEY_VALUE_GROUPS",
    "DEFAULT_MLP_PAGES",
    "DecodeRankOwnership",
    "contiguous_shard",
    "decode_world",
]
