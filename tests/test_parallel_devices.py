from __future__ import annotations

from types import SimpleNamespace

import pytest

import llama3fhe.parallel.devices as devices_module
from llama3fhe.parallel.devices import device_scope, pinned_device_index


def test_device_scope_switches_and_restores(monkeypatch) -> None:
    calls: list[int] = []
    state = {"current": 0}

    def set_device(index):
        calls.append(int(index))
        state["current"] = int(index)

    fake = SimpleNamespace(
        cuda=SimpleNamespace(current_device=lambda: state["current"], set_device=set_device)
    )
    monkeypatch.setattr(devices_module, "_easyfhe", lambda: fake)
    with device_scope(3):
        assert state["current"] == 3
    assert state["current"] == 0
    assert calls == [3, 0]
    with device_scope(0):
        pass
    assert calls == [3, 0]
    with pytest.raises(ValueError):
        with device_scope(-1):
            pass


def test_pinned_device_index_reads_single_visible_device(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pinned_device_index() is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    assert pinned_device_index() == 5
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert pinned_device_index() is None
