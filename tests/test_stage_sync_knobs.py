"""``T2_STAGE_SYNC`` (profile-only device syncs) and the shared T2 CLI knobs (``--bs-cache-mode``, ``--no-stage-sync``,
``--attn-in-overlap`` / ``--no-attn-in-overlap``)."""

from __future__ import annotations

import argparse
import os

import pytest

from llama3fhe import backend
from llama3fhe.parallel.presets import preset_by_name


_KNOB_ENVS = ("T2_STAGE_SYNC", "T2_ATTN_IN_OVERLAP")


@pytest.fixture(autouse=True)
def _reset_stage_sync():
    saved = {name: os.environ.pop(name, None) for name in _KNOB_ENVS}
    backend.set_stage_sync(None)
    try:
        yield
    finally:
        for name, value in saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value
        backend.set_stage_sync(None)


def test_stage_sync_defaults_on_and_follows_env(monkeypatch):
    assert backend.stage_sync_enabled() is True
    monkeypatch.setenv("T2_STAGE_SYNC", "0")
    assert backend.stage_sync_enabled() is True  # cached until reset
    backend.set_stage_sync(None)
    assert backend.stage_sync_enabled() is False
    backend.set_stage_sync(True)
    assert backend.stage_sync_enabled() is True


def test_profile_sync_is_a_no_op_when_disabled(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(backend, "synchronize_device", lambda device: calls.append(str(device)))
    backend.set_stage_sync(True)
    backend.profile_sync_device("cpu")
    assert calls == ["cpu"]


def test_key_tracer_load_timing_uses_profile_sync_switch(monkeypatch):
    from llama3fhe.parallel.prefill.keys import KeyTracer

    calls: list[str] = []
    monkeypatch.setattr(backend, "profile_sync_device", lambda device: calls.append(str(device)))

    class Native:
        device = "cuda:0"

    tracer = KeyTracer(Native())
    backend.set_stage_sync(True)
    tracer._sync()
    backend.set_stage_sync(False)
    tracer._sync()
    assert calls == ["cuda:0"]
    calls.clear()
    backend.set_stage_sync(False)
    backend.profile_sync_device("cpu")
    assert calls == ["cpu"]


def test_hot_path_modules_use_the_profile_sync():
    """The operators' stage-timer syncs route through the switch; the semantic syncs do not."""

    import llama3fhe.operators.attention.distributed.distributed_input as distributed_input
    import llama3fhe.operators.attention.distributed.distributed_output as distributed_output
    import llama3fhe.operators.attention.prefill.operator as attention_operator
    import llama3fhe.operators.linear.operator as linear_operator
    import llama3fhe.operators.norm.rmsnorm as rmsnorm
    import llama3fhe.parallel.prefill.transport as transport

    for module in (linear_operator, rmsnorm, distributed_input, distributed_output, attention_operator):
        assert module.sync_device is backend.profile_sync_device
    # the attention operator keeps a real sync for the key-eviction barriers
    assert attention_operator.synchronize_device is backend.synchronize_device
    # the pinned transport's ``_synchronize`` must stay a real device sync
    import inspect

    assert "synchronize_device" in inspect.getsource(transport.PinnedStagingTransport._synchronize)
    assert "profile_sync" not in inspect.getsource(transport.PinnedStagingTransport._synchronize)


def test_runtime_knobs_apply_to_preset_and_env(monkeypatch):
    from bench.t2_layer import add_runtime_knob_args, apply_runtime_knobs, attn_in_overlap_requested, stage_sync_requested

    parser = argparse.ArgumentParser()
    add_runtime_knob_args(parser)
    base = preset_by_name("rtx6000pro-high")

    args = parser.parse_args([])
    same = apply_runtime_knobs(args, base)
    assert same == base and same.bootstrap.cache_mode == "plain"
    assert "T2_STAGE_SYNC" not in os.environ and stage_sync_requested() is True
    # the overlap default comes from the preset when the environment is silent
    assert "T2_ATTN_IN_OVERLAP" not in os.environ
    assert attn_in_overlap_requested(base) is True and attn_in_overlap_requested(preset_by_name("rtx4090-low")) is False
    assert attn_in_overlap_requested(None) is False

    args = parser.parse_args(["--bs-cache-mode", "middle", "--no-stage-sync"])
    changed = apply_runtime_knobs(args, base)
    assert changed.bootstrap.cache_mode == "middle"
    assert changed.bootstrap.level_budget == base.bootstrap.level_budget
    assert changed.bootstrap.program_depth == base.bootstrap.program_depth
    # only the bootstrap cache mode moved
    assert changed == type(base).__call__(**{**changed.__dict__}) and {
        k for k, v in changed.__dict__.items() if v != base.__dict__[k]
    } == {"bootstrap"}
    assert os.environ["T2_STAGE_SYNC"] == "0" and stage_sync_requested() is False

    with pytest.raises(SystemExit):
        parser.parse_args(["--bs-cache-mode", "fast"])

    # the overlap switches only touch the environment; the environment then wins over any preset
    monkeypatch.delenv("T2_ATTN_IN_OVERLAP", raising=False)
    assert apply_runtime_knobs(parser.parse_args(["--no-attn-in-overlap"]), base) == base
    assert os.environ["T2_ATTN_IN_OVERLAP"] == "0" and attn_in_overlap_requested(base) is False
    apply_runtime_knobs(parser.parse_args(["--attn-in-overlap"]), base)
    assert os.environ["T2_ATTN_IN_OVERLAP"] == "1" and attn_in_overlap_requested(preset_by_name("rtx4090-low")) is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--attn-in-overlap", "--no-attn-in-overlap"])
