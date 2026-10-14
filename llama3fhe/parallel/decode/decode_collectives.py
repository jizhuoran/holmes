"""Ciphertext all-reduce helpers for the multi-card decode step.

The transport (:class:`~llama3fhe.parallel.prefill.transport.Transport`) only has
``all_gather`` / ``all_to_all``; a decode station reduces a partial
ciphertext by gathering every rank's partial and adding them locally in rank
order.  CKKS addition is exact modular addition of the RNS limbs, so every
rank ends up with the *bit-identical* sum whatever the order -- the ranks
stay in lock step without any further exchange.

Two flavours:

* :func:`all_reduce_ciphers` -- world-wide, one ``all_gather`` round for all
  ciphertexts of the list (QKV carrier partials, folded-O partials, MLP down
  partials);
* :func:`group_all_reduce_ciphers` -- among a subset of ranks, one
  ``all_to_all`` round whose payloads to ranks outside the group are empty
  (the MLP gate/up partials of the ranks sharing one intermediate page).

Both *consume* the input ciphertexts (released after the sum, or returned
as-is when the group has one member) and return owned ciphertexts.  With
``aligned=True`` the partials may sit at different levels and are aligned by
:func:`llama3fhe.operators.attention.pairfold.pairfold_decode.add_pairfold_output_contributions_fhe`
(folded-O contributions); otherwise they must share level and scale and are
added with ``homo_add`` (the linear operators' partials after the operator's
own rescale, or before it with ``rescale=False`` -- the caller rescales the
sum once).
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from llama3fhe.backend import release_all, release_if_supported

from ..wire import CipherPacket, pack_cipher, unpack_cipher


def packet_bytes(value: Any) -> int:
    """Bytes of a packet / nested sequence of packets (``None`` counts 0)."""

    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(packet_bytes(item) for item in value)
    nbytes = getattr(value, "nbytes", None)
    return int(nbytes) if nbytes is not None else 0


def _record(comm: dict[str, Any] | None, tag: str, *, sent: int, received: int, seconds: float) -> None:
    if comm is None:
        return
    comm[f"{tag}_sent"] = int(comm.get(f"{tag}_sent", 0)) + int(sent)
    comm[f"{tag}_received"] = int(comm.get(f"{tag}_received", 0)) + int(received)
    comm[f"{tag}_seconds"] = float(comm.get(f"{tag}_seconds", 0.0)) + float(seconds)
    comm[f"{tag}_rounds"] = int(comm.get(f"{tag}_rounds", 0)) + 1


def _sum_in_rank_order(
    own: Sequence[Any],
    foreign: dict[int, Sequence[CipherPacket]],
    *,
    rank: int,
    members: Sequence[int],
    crypto_context,
    aligned: bool,
) -> list[Any]:
    """``sum over members`` per list position; consumes ``own``."""

    count = len(own)
    outputs: list[Any] = []
    if aligned:
        from llama3fhe.operators.attention.pairfold.pairfold_decode import add_pairfold_output_contributions_fhe

    for index in range(count):
        terms: list[tuple[Any, bool]] = []  # (cipher, owned_by_us)
        for member in members:
            if member == rank:
                terms.append((own[index], True))
            else:
                packets = foreign[member]
                if len(packets) != count:
                    raise RuntimeError(
                        f"rank {member} sent {len(packets)} ciphertexts, expected {count} ({index=})"
                    )
                terms.append((unpack_cipher(packets[index], device=crypto_context.device), True))
        if len(terms) == 1:
            outputs.append(terms[0][0])
            continue
        if aligned:
            (combined,) = add_pairfold_output_contributions_fhe(
                [cipher for cipher, _ in terms], crypto_context=crypto_context
            )
            release_all([cipher for cipher, _ in terms])
            outputs.append(combined)
            continue
        accumulator, acc_owned = terms[0]
        for cipher, owned in terms[1:]:
            combined = crypto_context.fhe.homo_add(accumulator, cipher, crypto_context.context)
            if acc_owned:
                release_if_supported(accumulator)
            if owned:
                release_if_supported(cipher)
            accumulator, acc_owned = combined, True
        outputs.append(accumulator)
    return outputs


def all_reduce_ciphers(
    transport,
    tag: str,
    ciphers: Sequence[Any],
    *,
    crypto_context,
    comm: dict[str, Any] | None = None,
    aligned: bool = False,
) -> list[Any]:
    """World-wide sum of ``ciphers[i]`` over all ranks (one ``all_gather`` round).

    Consumes ``ciphers``; returns owned ciphertexts.  With a single rank the
    inputs are returned unchanged and no collective is issued.
    """

    ciphers = list(ciphers)
    world_size = int(transport.world_size)
    rank = int(transport.rank)
    if world_size == 1 or not ciphers:
        return ciphers
    started = time.perf_counter()
    host = not getattr(transport, "device_payloads", False)
    packets = tuple(pack_cipher(cipher, tag=f"{tag}/{rank}/{i}", host=host) for i, cipher in enumerate(ciphers))
    gathered = transport.all_gather(tag, packets)
    foreign = {src: tuple(gathered[src] or ()) for src in range(world_size) if src != rank}
    outputs = _sum_in_rank_order(
        ciphers, foreign, rank=rank, members=range(world_size), crypto_context=crypto_context, aligned=aligned
    )
    _record(
        comm,
        tag,
        sent=packet_bytes(packets) * (world_size - 1),
        received=sum(packet_bytes(v) for v in foreign.values()),
        seconds=time.perf_counter() - started,
    )
    return outputs


def group_all_reduce_ciphers(
    transport,
    tag: str,
    ciphers: Sequence[Any],
    *,
    group: Sequence[int],
    crypto_context,
    comm: dict[str, Any] | None = None,
    aligned: bool = False,
) -> list[Any]:
    """Sum of ``ciphers[i]`` over the ranks in ``group`` (one ``all_to_all`` round).

    Every rank of the world must call this (same order, same tag); ranks that
    are not in ``group`` take part in the round with empty payloads and get
    their (unchanged) inputs back.  Consumes ``ciphers``.
    """

    ciphers = list(ciphers)
    world_size = int(transport.world_size)
    rank = int(transport.rank)
    group = tuple(sorted(int(member) for member in group))
    if world_size == 1:
        return ciphers
    started = time.perf_counter()
    in_group = rank in group and len(group) > 1
    payloads_by_dst: list[Any] = [None] * world_size
    if in_group and ciphers:
        host = not getattr(transport, "device_payloads", False)
        packets = tuple(pack_cipher(cipher, tag=f"{tag}/{rank}/{i}", host=host) for i, cipher in enumerate(ciphers))
        for peer in group:
            if peer != rank:
                payloads_by_dst[peer] = packets
    received = transport.all_to_all(tag, payloads_by_dst)
    if not in_group or not ciphers:
        _record(comm, tag, sent=0, received=0, seconds=time.perf_counter() - started)
        return ciphers
    foreign = {src: tuple(received[src] or ()) for src in group if src != rank}
    outputs = _sum_in_rank_order(
        ciphers, foreign, rank=rank, members=group, crypto_context=crypto_context, aligned=aligned
    )
    _record(
        comm,
        tag,
        sent=sum(packet_bytes(p) for dst, p in enumerate(payloads_by_dst) if dst != rank),
        received=sum(packet_bytes(v) for v in foreign.values()),
        seconds=time.perf_counter() - started,
    )
    return outputs


def all_gather_ciphers(
    transport,
    tag: str,
    owned: dict[int, Any],
    *,
    count: int,
    crypto_context,
    comm: dict[str, Any] | None = None,
) -> list[Any]:
    """Assemble ``count`` ciphertexts every rank computed a disjoint subset of.

    ``owned`` maps index -> ciphertext this rank produced (kept as-is in the
    result); the other indices arrive from the rank that owns them (one
    ``all_gather`` round, every index must be contributed by exactly one
    rank).  Returns ``count`` owned ciphertexts, identical on every rank.
    Used for the per-ciphertext bootstraps the ranks split (KV refresh).
    """

    world_size = int(transport.world_size)
    rank = int(transport.rank)
    count = int(count)
    owned = {int(k): v for k, v in dict(owned).items()}
    if any(k < 0 or k >= count for k in owned):
        raise IndexError(f"owned indices {sorted(owned)} outside [0, {count})")
    if world_size == 1:
        if len(owned) != count:
            raise RuntimeError(f"single rank must own all {count} ciphertexts, has {sorted(owned)}")
        return [owned[i] for i in range(count)]
    started = time.perf_counter()
    host = not getattr(transport, "device_payloads", False)
    payload = tuple((index, pack_cipher(cipher, tag=f"{tag}/{rank}/{index}", host=host)) for index, cipher in sorted(owned.items()))
    gathered = transport.all_gather(tag, payload)
    outputs: list[Any] = [None] * count
    source: dict[int, int] = {}
    for src in range(world_size):
        for index, packet in tuple(gathered[src] or ()):
            index = int(index)
            if index in source:
                raise RuntimeError(f"ciphertext {index} contributed by ranks {source[index]} and {src}")
            source[index] = src
            if src == rank:
                outputs[index] = owned[index]
            else:
                outputs[index] = unpack_cipher(packet, device=crypto_context.device)
    missing = [i for i in range(count) if i not in source]
    if missing:
        raise RuntimeError(f"no rank contributed ciphertexts {missing}")
    _record(
        comm,
        tag,
        sent=sum(packet_bytes(p) for _, p in payload) * (world_size - 1),
        received=sum(packet_bytes(p) for src in range(world_size) if src != rank for _, p in tuple(gathered[src] or ())),
        seconds=time.perf_counter() - started,
    )
    return outputs


def broadcast_ciphers(
    transport,
    tag: str,
    ciphers: Sequence[Any] | None,
    *,
    root: int = 0,
    crypto_context,
    comm: dict[str, Any] | None = None,
) -> list[Any]:
    """``root``'s ciphertexts to every rank (one ``all_gather`` round).

    Only ``root`` contributes a payload; the other ranks pass ``None`` / an
    empty list and get *owned* copies back.  ``root`` gets its own inputs back
    unchanged.  Used to hand the client's ciphertexts (decode hidden state,
    KV pages) to all ranks so that the redundant stations start from
    bit-identical inputs.
    """

    world_size = int(transport.world_size)
    rank = int(transport.rank)
    root = int(root)
    own = list(ciphers or ())
    if world_size == 1:
        return own
    started = time.perf_counter()
    host = not getattr(transport, "device_payloads", False)
    packets = None
    if rank == root:
        packets = tuple(pack_cipher(cipher, tag=f"{tag}/{root}/{i}", host=host) for i, cipher in enumerate(own))
    gathered = transport.all_gather(tag, packets)
    if rank == root:
        _record(comm, tag, sent=packet_bytes(packets) * (world_size - 1), received=0, seconds=time.perf_counter() - started)
        return own
    root_packets = tuple(gathered[root] or ())
    outputs = [unpack_cipher(packet, device=crypto_context.device) for packet in root_packets]
    _record(comm, tag, sent=0, received=packet_bytes(root_packets), seconds=time.perf_counter() - started)
    return outputs


__all__ = ["all_gather_ciphers", "all_reduce_ciphers", "broadcast_ciphers", "group_all_reduce_ciphers", "packet_bytes"]
