"""A world of one rank, for running the multi-card decode kernels in-process.

The decode collectives (:mod:`.decode_collectives`) return their inputs
untouched when ``transport.world_size == 1``, so the single-process decode
graph (:func:`llama3fhe.graph.decode.evaluate_decode`) can reuse the
station kernels -- today the shifted-lane MLP
(:func:`llama3fhe.operators.linear.shard.decode_mlp_shard.decode_mlp_shard_fhe`)
-- by handing them this transport and ``DecodeRankOwnership(rank=0,
world_size=1)``.  Nothing is ever sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..prefill.transport import TransportStats


@dataclass
class SingleRankTransport:
    rank: int = 0
    world_size: int = 1
    stats: TransportStats = field(default_factory=TransportStats)
    device_payloads: bool = True

    def __post_init__(self) -> None:
        if int(self.rank) != 0 or int(self.world_size) != 1:
            raise ValueError("SingleRankTransport is rank 0 of a world of one.")

    def _unexpected(self, name: str):
        raise RuntimeError(f"SingleRankTransport.{name}: a world of one never communicates.")

    def all_gather(self, tag, payload):
        return [payload]

    def all_to_all(self, tag, payloads_by_dst):
        return list(payloads_by_dst)

    def barrier(self, tag: str) -> None:
        return None

    def send(self, *args, **kwargs):
        self._unexpected("send")

    def recv(self, *args, **kwargs):
        self._unexpected("recv")

    def close(self) -> None:
        return None


__all__ = ["SingleRankTransport"]
