from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from archive.experiments import pairfold128_cache_memory_gate as gate
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.pairfold_kv_cache import (
    PairFoldLayerCache,
    PairFoldTokenWrite,
)


class _FakeCipher:
    def __init__(self, registry: list[_FakeCipher]) -> None:
        self.release_count = 0
        registry.append(self)

    def release(self) -> None:
        if self.release_count:
            raise AssertionError("cipher released more than once")
        self.release_count += 1


class _FakeFHE:
    def __init__(self, registry: list[_FakeCipher]) -> None:
        self.registry = registry
        self.add_count = 0

    def homo_add(self, left, right, context):  # noqa: ANN001
        assert left.release_count == 0
        assert right.release_count == 0
        assert context is not None
        self.add_count += 1
        return _FakeCipher(self.registry)


class _FakeContext:
    def __init__(self, registry: list[_FakeCipher]) -> None:
        self.context = object()
        self.fhe = _FakeFHE(registry)


def test_expected_inventory_pins_both_page_boundaries() -> None:
    assert gate.expected_cache_inventory(0) == {
        "tokens": 0,
        "pages": 0,
        "ciphertexts": 0,
        "page_occupancies": [],
    }
    assert gate.expected_cache_inventory(128)["page_occupancies"] == [128]
    assert gate.expected_cache_inventory(129)["page_occupancies"] == [128, 1]
    assert gate.expected_cache_inventory(256)["page_occupancies"] == [128, 128]
    assert gate.expected_cache_inventory(257) == {
        "tokens": 257,
        "pages": 3,
        "ciphertexts": 12,
        "page_occupancies": [128, 128, 1],
    }


def test_append_sequence_exercises_transactional_tail_and_releases_all() -> None:
    registry: list[_FakeCipher] = []
    context = _FakeContext(registry)
    cache = PairFoldLayerCache()

    def token_factory(ordinal: int, column: int) -> PairFoldTokenWrite:
        assert column == ordinal % 128
        handles = [_FakeCipher(registry) for _ in range(4)]
        return PairFoldTokenWrite(
            key_groups=tuple(handles[:2]),
            value_groups=tuple(handles[2:]),
            column=column,
        )

    observations = gate.append_token_sequence(
        cache,
        crypto_context=context,
        token_factory=token_factory,
    )

    assert observations == [
        {
            **gate.expected_cache_inventory(128),
            "append_page_index": 0,
            "append_column": 127,
        },
        {
            **gate.expected_cache_inventory(129),
            "append_page_index": 1,
            "append_column": 0,
        },
        {
            **gate.expected_cache_inventory(256),
            "append_page_index": 1,
            "append_column": 127,
        },
        {
            **gate.expected_cache_inventory(257),
            "append_page_index": 2,
            "append_column": 0,
        },
    ]
    # 254 partial-page appends, four independent K/K/V/V replacements each.
    assert context.fhe.add_count == 254 * 4
    assert sum(cipher.release_count == 0 for cipher in registry) == 12

    cache.release()
    assert all(cipher.release_count == 1 for cipher in registry)


def test_return_gate_checks_device_live_and_reserved_bytes() -> None:
    start = {
        "device_used_bytes": 1_000,
        "allocator_allocated_bytes": 200,
        "allocator_reserved_bytes": 400,
    }
    passing = {
        "device_used_bytes": 1_064,
        "allocator_allocated_bytes": 264,
        "allocator_reserved_bytes": 464,
    }
    failing = {**passing, "allocator_reserved_bytes": 465}

    passed = gate.evaluate_return_gate(start, passing, tolerance_bytes=64)
    failed = gate.evaluate_return_gate(start, failing, tolerance_bytes=64)

    assert passed["passed"] is True
    assert all(row["passed"] for row in passed["checks"].values())
    assert failed["passed"] is False
    assert failed["checks"]["allocator_reserved_bytes"]["delta_bytes"] == 65


def test_real_payload_helper_uses_only_requested_pairfold_column() -> None:
    layout = PairFold128Layout()
    payloads = gate._folded_token_payloads(layout, ordinal=129, column=1)
    expected_slots = gate._token_slots(layout, 1)

    assert payloads.shape == (4, layout.slots)
    assert np.all(np.count_nonzero(payloads, axis=1) == expected_slots.size)
    assert np.all(payloads[:2, expected_slots].imag < 0.0)
    assert np.all(payloads[2:, expected_slots].imag > 0.0)
    inactive = np.ones((layout.slots,), dtype=bool)
    inactive[expected_slots] = False
    assert not np.any(payloads[:, inactive])


def test_cli_defaults_and_json_writer_are_gate_specific(tmp_path: Path) -> None:
    parsed = gate._build_parser().parse_args([])
    assert parsed.device == "cuda"
    assert parsed.kv_limbs == 12
    assert parsed.tolerance_mib == 64.0
    assert parsed.output_json.name == "pairfold128_cache_257_memory_gate.json"

    output = tmp_path / "nested" / "report.json"
    payload = {"schema": "test", "bytes": 123, "passed": True}
    gate._write_report(payload, output)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
