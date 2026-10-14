"""CPU tests of the pipelined plaintext encode in :meth:`LinearOperator.apply` (``encode_depth > 0``).

The CUDA streams and events are faked; the checks are about the schedule --
which chunk is encoded when, what the MAC stream waits on, how plaintexts are
handed back to the allocator -- and about the results matching the
sequential schedule.
"""

from __future__ import annotations

import pytest

from llama3fhe.operators.linear import LinearOperator, LinearOperatorConfig
from llama3fhe.operators.linear import operator as linear_operator_impl
from llama3fhe.operators.linear.config import default_linear_encode_depth


# ------------------------------------------------------------------ fakes
class FakeEvent:
    def __init__(self, enable_timing=False):
        self.enable_timing = bool(enable_timing)
        self.recorded_on = None

    def record(self, stream=None):
        self.recorded_on = "current" if stream is None else stream

    def elapsed_time(self, other):
        assert self.recorded_on is not None and other.recorded_on is not None
        return 1.0  # milliseconds


class FakeStream:
    def __init__(self, name):
        self.name = name
        self.waited = []
        self.synchronized = 0

    def wait_event(self, event):
        self.waited.append(event)

    def synchronize(self):
        self.synchronized += 1

    def __repr__(self):
        return f"FakeStream({self.name})"


class FakeComponent:
    def __init__(self, log):
        self.log = log
        self.recorded = []

    def record_stream(self, stream):
        self.recorded.append(stream)
        self.log.append(("record_stream", stream.name))


class FakePlaintext:
    def __init__(self, chunk_index, log):
        self.chunk_index = int(chunk_index)
        self.cv = [FakeComponent(log), FakeComponent(log)]


class FakeContext:
    context = object()

    def __init__(self, device):
        self.device = device

    @staticmethod
    def level_for_cipher(cipher) -> int:
        del cipher
        return 3


class FakeWeight:
    """Records the order of encode requests and on which stream they were issued."""

    closed = False

    def __init__(self, crypto_context, dimension, log):
        self.crypto_context = crypto_context
        self.dimension = int(dimension)
        self.log = log
        self.sequential_calls = 0
        self.async_calls = 0

    def prepare(self, operator_config):
        self.log.append(("prepare", str(operator_config.raw_weight_policy)))
        return 0.0

    def plaintext(self, plan, chunk, *, level, operator_config):
        del plan, level, operator_config
        self.sequential_calls += 1
        self.log.append(("encode", int(chunk.index), None))
        return FakePlaintext(chunk.index, self.log)

    def plaintext_async(self, plan, chunk, *, level, operator_config, stream):
        del plan, level, operator_config
        self.async_calls += 1
        self.log.append(("encode", int(chunk.index), stream.name))
        started, ready = FakeEvent(enable_timing=True), FakeEvent(enable_timing=True)
        started.record(stream)
        ready.record(stream)
        ready.chunk_index = int(chunk.index)
        return FakePlaintext(chunk.index, self.log), started, ready


@pytest.fixture
def fake_backend(monkeypatch):
    """Fake ``fhe`` kernels + fake CUDA streams/events; returns the shared event log."""

    import easyfhe

    log: list = []
    main = FakeStream("main")
    side = FakeStream("side")

    def mac_sum(cipher, baby_offsets, plaintexts, giant_offset, giant_count, context, **kwargs):
        del baby_offsets, giant_offset, giant_count, context
        assert kwargs["baby_rotations"] is not None
        log.append(("mac", plaintexts.chunk_index, cipher))
        return ("mac", cipher, plaintexts.chunk_index)

    def rotate(value, steps, context):
        del context
        return ("rot", value, int(steps))

    def add(left, right, context):
        del context
        return ("add", left, right)

    def rescale(value, context):
        del context
        return ("rescale", value)

    def prepare_babies(cipher, offsets, context, **kwargs):
        del offsets, context, kwargs
        log.append(("baby", cipher))
        return (("baby", cipher),)

    monkeypatch.setattr(linear_operator_impl.fhe, "hoisted_mac_sum", mac_sum)
    monkeypatch.setattr(linear_operator_impl.fhe, "homo_rotate", rotate)
    monkeypatch.setattr(linear_operator_impl.fhe, "homo_add", add)
    monkeypatch.setattr(linear_operator_impl.fhe, "rescale", rescale)
    monkeypatch.setattr(linear_operator_impl.fhe, "prepare_hoisted_baby_rotations", prepare_babies)
    monkeypatch.setattr(linear_operator_impl, "sync_device", lambda device: log.append(("sync", device)))
    monkeypatch.setattr(linear_operator_impl, "_encode_stream", lambda device: side)
    monkeypatch.setattr(easyfhe.cuda, "current_stream", lambda device=None: main)
    monkeypatch.setattr(easyfhe.cuda, "Event", FakeEvent)
    return {"log": log, "main": main, "side": side}


CHUNKS = 4  # dimension 32, token_lanes 2, baby_steps 4 -> 8 giants; 8 rows per chunk -> 2 giants per chunk


def _operator(device, depth, *, max_plaintext_rows=8):
    config = LinearOperatorConfig(
        baby_steps=4, baby_anchor_step=4, max_plaintext_rows=max_plaintext_rows, encode_depth=depth
    )
    context = FakeContext(device)
    return LinearOperator(crypto_context=context, token_lanes=2, config=config), context


def _run(fake_backend, device, depth, inputs=("c0", "c1")):
    operator, context = _operator(device, depth)
    log = fake_backend["log"]
    del log[:]
    weight = FakeWeight(context, dimension=32, log=log)
    outputs, profile = operator.apply(tuple(inputs), weight)
    operator.close()
    return outputs, profile, list(log), weight


# ------------------------------------------------------------------ tests
def test_encode_depth_default_comes_from_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_FHE_LINEAR_ENCODE_DEPTH", raising=False)
    assert default_linear_encode_depth() == 0
    assert LinearOperatorConfig().encode_depth == 0
    monkeypatch.setenv("LLAMA_FHE_LINEAR_ENCODE_DEPTH", "2")
    assert LinearOperatorConfig().encode_depth == 2
    assert LinearOperatorConfig(encode_depth=0).encode_depth == 0
    monkeypatch.setenv("LLAMA_FHE_LINEAR_ENCODE_DEPTH", "-1")
    with pytest.raises(ValueError):
        default_linear_encode_depth()
    monkeypatch.setenv("LLAMA_FHE_LINEAR_ENCODE_DEPTH", "two")
    with pytest.raises(ValueError):
        default_linear_encode_depth()
    with pytest.raises(ValueError):
        LinearOperatorConfig(encode_depth=-1)


def test_cpu_device_keeps_the_sequential_schedule(fake_backend) -> None:
    outputs, profile, log, weight = _run(fake_backend, "cpu", depth=2)
    assert weight.async_calls == 0
    assert weight.sequential_calls == CHUNKS
    assert "pipelined_chunks" not in profile
    assert len(outputs) == 2
    # encode -> sync -> MAC -> sync per chunk
    encodes = [entry for entry in log if entry[0] == "encode"]
    assert [entry[1] for entry in encodes] == list(range(CHUNKS))


def test_pipelined_schedule_keeps_depth_chunks_ahead(fake_backend) -> None:
    depth = 2
    outputs, profile, log, weight = _run(fake_backend, "cuda:0", depth=depth)
    main, side = fake_backend["main"], fake_backend["side"]

    assert weight.sequential_calls == 0
    assert weight.async_calls == CHUNKS
    encodes = [entry for entry in log if entry[0] == "encode"]
    assert all(entry[2] == "side" for entry in encodes)
    assert [entry[1] for entry in encodes] == list(range(CHUNKS))

    # the MAC stream waits on chunk i's ready event before MAC'ing chunk i
    waited = [event.chunk_index for event in main.waited]
    assert waited == list(range(CHUNKS))

    # at most ``depth`` encodes are issued ahead of the MACs: encode(i + depth) comes after mac(i)
    order = [entry for entry in log if entry[0] in ("encode", "mac")]
    first_mac_of = {}
    encode_at = {}
    for position, entry in enumerate(order):
        if entry[0] == "encode":
            encode_at[entry[1]] = position
        else:
            first_mac_of.setdefault(entry[1], position)
    for chunk in range(CHUNKS):
        if chunk + depth < CHUNKS:
            assert encode_at[chunk + depth] > first_mac_of[chunk]
        if chunk >= depth:
            assert encode_at[chunk] > first_mac_of[chunk - depth]
    # the first ``depth`` encodes precede the baby precompute (which overlaps them on the MAC stream)
    positions = [index for index, entry in enumerate(log) if entry[0] in ("encode", "baby")]
    kinds = [log[index][0] for index in positions]
    assert kinds[:depth] == ["encode"] * depth
    assert kinds[depth : depth + 2] == ["baby", "baby"]

    # every plaintext component is handed to the allocator as read by the MAC stream
    records = [entry for entry in log if entry[0] == "record_stream"]
    assert len(records) == CHUNKS * 2 and all(entry[1] == "main" for entry in records)
    # one stream synchronisation closes the loop (the rescale's device sync follows)
    assert main.synchronized == 1
    assert side.synchronized == 0

    assert profile["pipelined_chunks"] == float(CHUNKS)
    assert profile["pipeline_wall"] >= 0.0
    assert profile["encode_issue_wait"] >= 0.0
    assert profile["baby_cache_misses"] == 2.0
    for chunk in range(CHUNKS):
        assert profile[f"pack_encode.chunk_{chunk}"] == pytest.approx(0.001)
        assert profile[f"online.chunk_{chunk}"] == pytest.approx(0.001)
    assert profile["pack_encode"] == pytest.approx(0.001 * CHUNKS)
    assert profile["online"] == pytest.approx(0.001 * CHUNKS)
    assert profile["baby_precompute"] == pytest.approx(0.001)
    assert len(outputs) == 2


def test_pipelined_results_match_the_sequential_schedule(fake_backend) -> None:
    sequential, _, _, _ = _run(fake_backend, "cuda:0", depth=0)
    for depth in (1, 2, 3, CHUNKS, 50):
        pipelined, profile, _, _ = _run(fake_backend, "cuda:0", depth=depth)
        assert pipelined == sequential
        assert profile["pipelined_chunks"] == float(CHUNKS)


def test_pipelined_depth_is_capped_by_the_chunk_count(fake_backend) -> None:
    _, profile, log, weight = _run(fake_backend, "cuda:0", depth=50)
    assert weight.async_calls == CHUNKS
    encodes = [index for index, entry in enumerate(log) if entry[0] == "encode"]
    macs = [index for index, entry in enumerate(log) if entry[0] == "mac"]
    # everything encoded first, then the MACs
    assert max(encodes) < min(macs)
    assert profile["pipelined_chunks"] == float(CHUNKS)
