from __future__ import annotations

from types import SimpleNamespace

import pytest

from llama3fhe.parallel.prefill import stations


class _Cipher:
    def __init__(self, limbs: int) -> None:
        self.state = SimpleNamespace(cur_limbs=int(limbs), scale_degree=1)
        self.is_ext = False


def test_linear_entry_returns_inputs_untouched_when_already_at_floor() -> None:
    ciphers = (_Cipher(13), _Cipher(13))
    out, owned = stations.linear_entry(ciphers, input_limbs=13, context=object())
    assert out == ciphers
    assert owned is False


def test_linear_entry_skips_when_no_floor_or_below_floor() -> None:
    ciphers = (_Cipher(11),)
    assert stations.linear_entry(ciphers, input_limbs=None, context=object()) == (ciphers, False)
    assert stations.linear_entry(ciphers, input_limbs=13, context=object()) == (ciphers, False)
    assert stations.linear_entry((), input_limbs=13, context=object()) == ((), False)


def test_linear_entry_drops_to_floor_and_marks_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_drop(ciphers, *, target_limbs, crypto_context):
        calls.append((len(ciphers), int(target_limbs)))
        return tuple(_Cipher(target_limbs) for _ in ciphers)

    import llama3fhe.operators.primitives as primitives

    monkeypatch.setattr(primitives, "drop_to_limbs", fake_drop)
    out, owned = stations.linear_entry((_Cipher(18), _Cipher(18)), input_limbs=13, context=object())
    assert owned is True
    assert calls == [(2, 13)]
    assert [c.state.cur_limbs for c in out] == [13, 13]


def test_mlp_input_limbs_matches_graph_formula() -> None:
    from llama3fhe.approx.chebyshev import chebyshev_ps_mul_depth

    # degree-7 Chebyshev: depth is whatever the PS scheduler says; the station
    # must add the 6-limb suffix and refund one limb when the domain map is fused.
    coefficients = [0.5, 0.25, 0.0, -0.125, 0.0, 0.0625, 0.0, -0.03125]
    depth = chebyshev_ps_mul_depth(coefficients)
    entry = {"coefficients": coefficients}
    assert stations.mlp_input_limbs(entry) == depth + stations.MLP_SUFFIX_LIMBS
    assert stations.mlp_input_limbs(entry, fuse_silu_domain_map=True) == depth + stations.MLP_SUFFIX_LIMBS - 1


def test_qkv_floor_is_production_attention_budget() -> None:
    assert stations.QKV_INPUT_LIMBS == 13
    assert stations.MLP_SUFFIX_LIMBS == 6


# ----------------------------------------------------------------- pairfm_exchange routing
class _Partial:
    """CPU stand-in for a pre-rescale partial: remembers (source rank, target carrier, value)."""

    def __init__(self, src: int, target: int, value: float = 1.0) -> None:
        self.src = int(src)
        self.target = int(target)
        self.value = float(value)
        self.state = SimpleNamespace(cur_limbs=13, scale_degree=2)
        self.released = False

    def release(self) -> None:
        self.released = True


class _Packet:
    def __init__(self, partial: _Partial, tag: str) -> None:
        self.partial = partial
        self.tag = tag
        self.nbytes = 8


def _fake_pairfm(monkeypatch: pytest.MonkeyPatch, *, world_size: int):
    """Patch the device halves + the wire so pairfm_exchange runs on plain objects."""

    import llama3fhe.operators.attention.distributed.distributed_output as dist
    from llama3fhe.parallel import wire

    def fake_rank_partials(groups, *, slot_indices, **_):
        slots = tuple(int(s) for s in slot_indices)
        # rank id = owner of the first slot; each partial carries the *slot set* as value
        rank = slots[0] * world_size // 8
        return SimpleNamespace(
            partial_ciphers=tuple(_Partial(rank, target, value=sum(slots)) for target in range(8)),
            slot_indices=slots,
            stage_seconds={"partial_online": 1.0},
            slot_wall_seconds=tuple(0.5 for _ in slots),
        )

    reduced: list = []

    def fake_reduce(partials, *, crypto_context, synchronize=True):
        partials = list(partials)
        for p in partials:
            p.released = True
        out = _Partial(-1, partials[0].target, value=sum(p.value for p in partials))
        out.summands = tuple((p.src, p.target) for p in partials)
        reduced.append(out)
        return out

    monkeypatch.setattr(dist, "pairfm_rank_partials_fhe", fake_rank_partials)
    monkeypatch.setattr(dist, "pairfm_target_reduce_fhe", fake_reduce)
    monkeypatch.setattr(wire, "pack_cipher", lambda cipher, *, tag="", host=True: _Packet(cipher, tag))
    monkeypatch.setattr(wire, "unpack_cipher", lambda packet, *, device=None: packet.partial)
    monkeypatch.setattr(stations, "_sync", lambda _context: None)
    return reduced


def _run_pairfm_world(monkeypatch: pytest.MonkeyPatch, world_size: int, *, after_exchange=None):
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
    from llama3fhe.parallel.prefill.ownership import ownership_from_layouts

    feature = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()
    world = ownership_from_layouts(feature, attention, world_size=world_size)
    reduced = _fake_pairfm(monkeypatch, world_size=world_size)
    hook = None if after_exchange is None else (lambda: after_exchange(reduced))
    context = SimpleNamespace(device="cpu")
    ws = int(world_size)

    # Drive the collective by hand from one thread: first run every rank up to
    # its all_to_all (which stores the payload row and aborts), then re-run
    # with a transport that answers from the collected table.
    payload_table: dict[int, list] = {}

    class _Stop(Exception):
        pass

    class _Collect:
        world_size = ws

        def __init__(self, rank):
            self.rank = rank

        def all_to_all(self, tag, payloads_by_dst):
            payload_table[self.rank] = list(payloads_by_dst)
            raise _Stop()

    for rank in world:
        items = tuple((p, _Cipher(4)) for p in rank.query_pair_indices)
        with pytest.raises(_Stop):
            stations.pairfm_exchange(items, ownership=rank, world=world, transport=_Collect(rank.rank),
                                     feature_layout=feature, attention_layout=attention, context=context)
    assert sorted(payload_table) == list(range(world_size))

    class _Answer:
        world_size = ws

        def __init__(self, rank):
            self.rank = rank

        def all_to_all(self, tag, payloads_by_dst):
            return [payload_table[src][self.rank] for src in range(ws)]

    results = []
    for rank in world:
        items = tuple((p, _Cipher(4)) for p in rank.query_pair_indices)
        results.append(stations.pairfm_exchange(items, ownership=rank, world=world, transport=_Answer(rank.rank),
                                                feature_layout=feature, attention_layout=attention, context=context,
                                                after_exchange=hook))
    return world, payload_table, results


@pytest.mark.parametrize("world_size", [8, 4, 2])
def test_pairfm_exchange_routes_one_packet_per_owned_carrier(monkeypatch: pytest.MonkeyPatch, world_size: int) -> None:
    world, payloads, results = _run_pairfm_world(monkeypatch, world_size)
    spr = 8 // world_size
    for src in world:
        row = payloads[src.rank]
        assert row[src.rank] is None
        for dst in world:
            if dst.rank == src.rank:
                continue
            packets = row[dst.rank]
            # one packet per carrier the destination owns, in its carrier order
            assert [p.partial.target for p in packets] == list(dst.target_carrier_indices)
            assert all(p.partial.src == src.rank for p in packets)
            assert len(packets) == spr
    for rank, result in zip(world, results, strict=True):
        assert result.carrier_indices == rank.target_carrier_indices
        assert len(result.carriers) == spr
        assert result.output_limbs == tuple(13 for _ in range(spr))
        for carrier, index in zip(result.carriers, rank.target_carrier_indices, strict=True):
            # every rank contributed exactly one partial for this carrier
            assert sorted(s for s, _ in carrier.summands) == list(range(world_size))
            assert {t for _, t in carrier.summands} == {index}
            # sum of slot-set values over all ranks = sum over all 8 slots
            assert carrier.value == pytest.approx(sum(range(8)))
        if spr == 1:
            assert result.carrier is result.carriers[0]
            assert result.carrier_index == rank.target_carrier_index
        else:
            with pytest.raises(ValueError):
                result.carrier
        assert result.comm_bytes["partial_sent"] == 8 * spr * (world_size - 1)
        assert result.comm_bytes["partial_received"] == 8 * spr * (world_size - 1)
        assert result.source_slot_seconds == tuple(0.5 for _ in range(spr))
        assert result.ciphers == result.carriers
        result.release()
        assert result.ciphers == ()


def test_pairfm_exchange_hook_runs_before_local_target_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    """The KV gather gate is released after PairFM transport, before reduce."""

    world_size = 2
    spr = 8 // world_size
    observed: list[int] = []
    _run_pairfm_world(monkeypatch, world_size, after_exchange=lambda reduced: observed.append(len(reduced)))
    assert observed == [0, spr]


def test_pairfm_exchange_rejects_foreign_query_pairs() -> None:
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
    from llama3fhe.parallel.prefill.ownership import ownership_from_layouts

    feature = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()
    world = ownership_from_layouts(feature, attention, world_size=4)
    rank = world[1]
    wrong = tuple((p, _Cipher(4)) for p in (0, 1, 2, 3))
    with pytest.raises(ValueError, match="own query pairs"):
        stations.pairfm_exchange(wrong, ownership=rank, world=world, transport=None,
                                 feature_layout=feature, attention_layout=attention, context=None)


# ---------------------------------------------------------------------------- w_o_station
def test_w_o_station_accepts_one_carrier_or_the_carrier_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    import llama3fhe.operators.linear.prefill.output as output
    from llama3fhe.parallel.prefill.ownership import build_world_ownership

    calls: list = []

    def fake(carriers, weight, *, carrier_indices, **kwargs):
        calls.append((tuple(carriers), tuple(carrier_indices)))
        return tuple(f"fm{2 * c + k}" for c in carrier_indices for k in (0, 1))

    monkeypatch.setattr(output, "w_o_carrier_shards_fhe", fake)
    eight = build_world_ownership()
    out = stations.w_o_station("c5", None, ownership=eight[5], layout=None, attention_layout=None, context=None)
    assert out == ("fm10", "fm11")
    assert calls[-1] == (("c5",), (5,))
    four = build_world_ownership(world_size=4, ciphers_per_rank=4)
    out = stations.w_o_station(("c2", "c3"), None, ownership=four[1], layout=None, attention_layout=None, context=None)
    assert out == ("fm4", "fm5", "fm6", "fm7")
    assert calls[-1] == (("c2", "c3"), (2, 3))
    with pytest.raises(ValueError, match="carriers"):
        stations.w_o_station("c2", None, ownership=four[1], layout=None, attention_layout=None, context=None)


def test_pairfm_exchange_releases_everything_when_a_reduce_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """world 4, rank 1 owns carriers (2, 3): the second reduce raises -> the first
    carrier, the not-yet-reduced summands and nothing else leak."""

    import llama3fhe.operators.attention.distributed.distributed_output as dist
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
    from llama3fhe.parallel.prefill.ownership import ownership_from_layouts

    feature = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()
    world = ownership_from_layouts(feature, attention, world_size=4)
    rank = world[1]
    _fake_pairfm(monkeypatch, world_size=4)

    made: list[_Partial] = []
    produced: list[_Partial] = []

    def fake_rank_partials(groups, *, slot_indices, **_):
        partials = tuple(_Partial(1, target) for target in range(8))
        made.extend(partials)
        return SimpleNamespace(partial_ciphers=partials, slot_indices=tuple(slot_indices),
                               stage_seconds={}, slot_wall_seconds=(0.0, 0.0))

    calls = {"n": 0}

    def failing_reduce(partials, *, crypto_context, synchronize=True):
        partials = list(partials)
        for p in partials:
            p.released = True  # the real reduce consumes its inputs
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        out = _Partial(-1, partials[0].target)
        produced.append(out)
        return out

    monkeypatch.setattr(dist, "pairfm_rank_partials_fhe", fake_rank_partials)
    monkeypatch.setattr(dist, "pairfm_target_reduce_fhe", failing_reduce)

    foreign = {src: [_Partial(src, c) for c in rank.target_carrier_indices] for src in (0, 2, 3)}

    class _T:
        world_size = 4

        def all_to_all(self, tag, payloads_by_dst):
            return [None if src == 1 else tuple(_Packet(p, "") for p in foreign[src]) for src in range(4)]

    items = tuple((p, _Cipher(4)) for p in rank.query_pair_indices)
    with pytest.raises(RuntimeError, match="boom"):
        stations.pairfm_exchange(items, ownership=rank, world=world, transport=_T(), feature_layout=feature,
                                 attention_layout=attention, context=SimpleNamespace(device="cpu"))
    # own partials: the 6 sent ones released in step 3, the 2 kept consumed by the reduces
    assert all(p.released for p in made)
    # every foreign partial was handed to a reduce (consumed) or released by the except path
    assert all(p.released for group in foreign.values() for p in group)
    # the carrier produced by the first reduce is released too
    assert produced and all(p.released for p in produced)
