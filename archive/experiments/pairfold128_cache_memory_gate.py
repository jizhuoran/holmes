#!/usr/bin/env python3
from __future__ import annotations

"""Focused PairFold128 257-token cache ownership and CUDA-memory gate.

This experiment intentionally does not evaluate a Transformer layer.  It
encrypts four real EasyFHE ciphertexts per Decode token and feeds those token
writes through :class:`PairFoldLayerCache.append`, including all transactional
tail replacements.  The four interesting boundaries are recorded before the
three-page cache is explicitly released and the CUDA allocator is trimmed.

Run it as a module so the repository package is resolved consistently::

    PYTHONPATH=. env_fhe/bin/python -m \
      archive.experiments.pairfold128_cache_memory_gate \
      --output-json artifacts/pairfold128_cache_257_memory_gate.json
"""

import argparse
import gc
import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import easyfhe
import numpy as np

from llama3fhe.backend import (
    create_sim_context,
    cuda_memory_snapshot,
    release_if_supported,
)
from llama3fhe.config import SimulatorConfig
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.pairfold_kv_cache import (
    PAIRFOLD_PAGE_SIZE,
    PairFoldLayerCache,
    PairFoldTokenWrite,
)

GATE_TOKENS = 257
BOUNDARY_TOKENS = (128, 129, 256, 257)
DEFAULT_TOLERANCE_MIB = 64
DEFAULT_KV_LIMBS = 12
_MIB = 1024**2
_RETURN_METRICS = (
    "device_used_bytes",
    "allocator_allocated_bytes",
    "allocator_reserved_bytes",
)


def expected_cache_inventory(tokens: int) -> dict[str, object]:
    """Return the no-eviction PairFold page shape for ``tokens``."""

    tokens = int(tokens)
    if tokens < 0:
        raise ValueError("tokens must be non-negative.")
    if tokens == 0:
        occupancies: list[int] = []
    else:
        full_pages, tail = divmod(tokens, PAIRFOLD_PAGE_SIZE)
        occupancies = [PAIRFOLD_PAGE_SIZE] * full_pages
        if tail:
            occupancies.append(tail)
    pages = len(occupancies)
    return {
        "tokens": tokens,
        "pages": pages,
        "ciphertexts": 4 * pages,
        "page_occupancies": occupancies,
    }


def cache_inventory(cache: PairFoldLayerCache) -> dict[str, object]:
    """Return a JSON-safe inventory while also exercising ownership checks."""

    handles = cache.handles
    inventory = {
        "tokens": int(cache.content_tokens),
        "pages": len(cache.pages),
        "ciphertexts": len(handles),
        "page_occupancies": [int(page.occupied) for page in cache.pages],
    }
    expected = expected_cache_inventory(cache.content_tokens)
    if inventory != expected:
        raise AssertionError(
            f"PairFold cache inventory mismatch: {inventory} != {expected}."
        )
    return inventory


def append_token_sequence(
    cache: PairFoldLayerCache,
    *,
    crypto_context,
    token_factory: Callable[[int, int], PairFoldTokenWrite],
    total_tokens: int = GATE_TOKENS,
    checkpoints: Iterable[int] = BOUNDARY_TOKENS,
) -> list[dict[str, object]]:
    """Append token owners and capture selected page-boundary inventories.

    ``token_factory`` receives the zero-based token ordinal and required page
    column.  If append fails before consuming a token, this helper releases
    that still-live owner; the caller remains responsible for ``cache``.
    """

    total_tokens = int(total_tokens)
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive.")
    wanted = {int(value) for value in checkpoints}
    if any(value <= 0 or value > total_tokens for value in wanted):
        raise ValueError("checkpoints must lie inside the append sequence.")

    observations: list[dict[str, object]] = []
    for ordinal in range(total_tokens):
        column = ordinal % PAIRFOLD_PAGE_SIZE
        token = token_factory(ordinal, column)
        try:
            page_index, committed_column = cache.append(
                token,
                crypto_context=crypto_context,
            )
        except BaseException:
            if not token.is_consumed:
                token.release()
            raise
        token_count = ordinal + 1
        if token_count in wanted:
            inventory = cache_inventory(cache)
            expected_page, expected_column = divmod(
                token_count - 1, PAIRFOLD_PAGE_SIZE
            )
            if (page_index, committed_column) != (
                expected_page,
                expected_column,
            ):
                raise AssertionError(
                    "append location mismatch at token "
                    f"{token_count}: {(page_index, committed_column)} != "
                    f"{(expected_page, expected_column)}."
                )
            observations.append(
                {
                    **inventory,
                    "append_page_index": int(page_index),
                    "append_column": int(committed_column),
                }
            )
    return observations


def evaluate_return_gate(
    start: Mapping[str, object],
    after_trim: Mapping[str, object],
    *,
    tolerance_bytes: int,
) -> dict[str, object]:
    """Check that live and reserved CUDA bytes return near the warm baseline."""

    tolerance_bytes = int(tolerance_bytes)
    if tolerance_bytes < 0:
        raise ValueError("tolerance_bytes must be non-negative.")
    checks: dict[str, object] = {}
    for metric in _RETURN_METRICS:
        if metric not in start or metric not in after_trim:
            raise ValueError(f"memory snapshots are missing {metric!r}.")
        baseline = int(start[metric])
        observed = int(after_trim[metric])
        limit = baseline + tolerance_bytes
        checks[metric] = {
            "start_bytes": baseline,
            "after_trim_bytes": observed,
            "delta_bytes": observed - baseline,
            "limit_bytes": limit,
            "passed": observed <= limit,
        }
    return {
        "tolerance_bytes": tolerance_bytes,
        "tolerance_mib": tolerance_bytes / _MIB,
        "checks": checks,
        "passed": all(bool(row["passed"]) for row in checks.values()),
    }


def _snapshot_with_mib(device: str, *, phase: str) -> dict[str, object]:
    snapshot = cuda_memory_snapshot(device)
    if not bool(snapshot.get("available", False)):
        raise RuntimeError(f"CUDA memory snapshot unavailable for {device!r}.")
    result: dict[str, object] = {"phase": str(phase), **snapshot}
    result["mib"] = {
        name.removesuffix("_bytes"): float(value) / _MIB
        for name, value in snapshot.items()
        if name.endswith("_bytes") and isinstance(value, int)
    }
    return result


def _force_allocator_trim(device: str) -> None:
    """Synchronize and force ``empty_cache`` regardless of production policy."""

    module = easyfhe.get_device_module(str(device))
    if str(device).startswith("cuda"):
        # ``empty_cache`` acts on the current CUDA device, so bind the exact
        # index instead of silently trimming cuda:0 for a cuda:N gate.
        with module.device(str(device)):
            module.synchronize(str(device))
            module.empty_cache()
            module.synchronize(str(device))
        return
    module.synchronize()
    module.empty_cache()
    module.synchronize()


def _synchronize_exact_device(device: str) -> None:
    """Synchronize the selected CUDA index rather than the ambient default."""

    module = easyfhe.get_device_module(str(device))
    with module.device(str(device)):
        module.synchronize(str(device))


def _reset_allocator_peaks(device: str) -> None:
    import torch

    torch.cuda.reset_peak_memory_stats(torch.device(str(device)))


def _token_slots(layout: PairFold128Layout, column: int) -> np.ndarray:
    """Vector of all valid folded slots for one page column and one KV group."""

    column = int(column)
    if not 0 <= column < PAIRFOLD_PAGE_SIZE:
        raise ValueError(f"column must lie in [0, 127], got {column}.")
    return np.fromiter(
        (
            layout.page_slot(pair, feature, column, lane)
            for pair in range(layout.pair_blocks)
            for feature in range(layout.fold_dim)
            for lane in range(layout.head_lanes)
        ),
        dtype=np.int64,
        count=layout.pair_blocks * layout.fold_dim * layout.head_lanes,
    )


def _folded_token_payloads(
    layout: PairFold128Layout,
    *,
    ordinal: int,
    column: int,
) -> np.ndarray:
    """Create four small nonzero K/K/V/V payloads at one page column."""

    slots = _token_slots(layout, column)
    payloads = np.zeros((4, int(layout.slots)), dtype=np.complex128)
    token_scale = 1.0 + (int(ordinal) % PAIRFOLD_PAGE_SIZE) / PAIRFOLD_PAGE_SIZE
    for cipher_index in range(4):
        real = 1e-5 * token_scale * (cipher_index + 1)
        imaginary = 0.5e-5 * token_scale * (cipher_index + 1)
        if cipher_index < int(layout.key_value_groups):
            imaginary = -imaginary
        payloads[cipher_index, slots] = complex(real, imaginary)
    return payloads


def _encrypted_token_factory(
    *,
    layout: PairFold128Layout,
    crypto_context,
    level: int,
) -> Callable[[int, int], PairFoldTokenWrite]:
    def create(ordinal: int, column: int) -> PairFoldTokenWrite:
        payloads = _folded_token_payloads(
            layout,
            ordinal=ordinal,
            column=column,
        )
        outputs: list[object] = []
        try:
            for payload in payloads:
                outputs.append(
                    crypto_context.encrypt(
                        payload,
                        level=int(level),
                        slots=int(layout.slots),
                    )
                )
            token = PairFoldTokenWrite(
                key_groups=tuple(outputs[: layout.key_value_groups]),
                value_groups=tuple(outputs[layout.key_value_groups :]),
                column=column,
            )
            outputs = []
            return token
        finally:
            for cipher in outputs:
                release_if_supported(cipher)

    return create


def _cipher_metadata(cache: PairFoldLayerCache) -> dict[str, object]:
    handles = cache.handles
    return {
        "python_types": sorted(
            {f"{type(handle).__module__}.{type(handle).__qualname__}" for handle in handles}
        ),
        "cur_limbs": sorted({int(handle.state.cur_limbs) for handle in handles}),
        "scale_degrees": sorted(
            {int(handle.state.scale_degree) for handle in handles}
        ),
        "extended_flags": sorted({bool(handle.is_ext) for handle in handles}),
    }


def _write_report(report: Mapping[str, Any], output_json: Path) -> None:
    text = json.dumps(report, indent=2, ensure_ascii=False)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


def run_gate(args: argparse.Namespace) -> dict[str, object]:
    """Run the real-CUDA gate and return its complete JSON payload."""

    device = str(args.device)
    if not device.startswith("cuda"):
        raise ValueError("the PairFold memory gate requires a CUDA device.")
    kv_limbs = int(args.kv_limbs)
    if kv_limbs <= 1:
        raise ValueError("kv_limbs must be greater than one.")
    tolerance_bytes = int(round(float(args.tolerance_mib) * _MIB))
    if tolerance_bytes < 0:
        raise ValueError("tolerance_mib must be non-negative.")

    simulator = SimulatorConfig(
        logN=int(args.log_n),
        maxLevelsRemaining=int(args.max_levels),
        dnum=int(args.dnum),
        dcrtBits=int(args.dcrt_bits),
        firstMod=int(args.first_mod),
        device=device,
    )
    simulator.validate()
    layout = PairFold128Layout()

    context_start = time.perf_counter()
    context = create_sim_context(
        simulator,
        input_dim=int(layout.carrier_dimension),
        block_out=int(layout.native_token_lanes),
        rotations=(),
        include_default_rotations=False,
    )
    context_seconds = time.perf_counter() - context_start
    level = int(context.context.L) - kv_limbs
    if level < 0:
        raise ValueError(
            f"kv_limbs={kv_limbs} exceeds generated context L="
            f"{int(context.context.L)}."
        )
    token_factory = _encrypted_token_factory(
        layout=layout,
        crypto_context=context,
        level=level,
    )

    # Materialize encrypt/add kernels and allocator pools before defining the
    # baseline.  This warm-up has the same ownership path but no page boundary.
    warmup_cache = PairFoldLayerCache()
    try:
        append_token_sequence(
            warmup_cache,
            crypto_context=context,
            token_factory=token_factory,
            total_tokens=2,
            checkpoints=(),
        )
        _synchronize_exact_device(context.device)
    finally:
        warmup_cache.release()
        del warmup_cache
    gc.collect()
    _force_allocator_trim(context.device)
    _reset_allocator_peaks(context.device)
    start_memory = _snapshot_with_mib(context.device, phase="start")

    cache: PairFoldLayerCache | None = PairFoldLayerCache()
    boundaries: list[dict[str, object]] = []
    final_inventory: dict[str, object] | None = None
    cipher_metadata: dict[str, object] | None = None
    append_started = time.perf_counter()
    try:
        boundaries = append_token_sequence(
            cache,
            crypto_context=context,
            token_factory=token_factory,
        )
        _synchronize_exact_device(context.device)
        append_seconds = time.perf_counter() - append_started
        peak_memory = _snapshot_with_mib(context.device, phase="peak")
        final_inventory = cache_inventory(cache)
        cipher_metadata = _cipher_metadata(cache)
        if final_inventory != expected_cache_inventory(GATE_TOKENS):
            raise AssertionError("257-token PairFold inventory is not 3 pages/12 CT.")
        if cipher_metadata["cur_limbs"] != [kv_limbs]:
            raise AssertionError(
                "resident cache limb count differs from requested kv_limbs: "
                f"{cipher_metadata['cur_limbs']} != {[kv_limbs]}."
            )

        release_started = time.perf_counter()
        cache.release()
        cache = None
        gc.collect()
        _synchronize_exact_device(context.device)
        after_release_memory = _snapshot_with_mib(
            context.device, phase="after_release"
        )
        release_seconds = time.perf_counter() - release_started

        trim_started = time.perf_counter()
        _force_allocator_trim(context.device)
        after_trim_memory = _snapshot_with_mib(context.device, phase="after_trim")
        trim_seconds = time.perf_counter() - trim_started
    finally:
        if cache is not None:
            cache.release()
            cache = None
            gc.collect()
            _synchronize_exact_device(context.device)
            _force_allocator_trim(context.device)

    return_gate = evaluate_return_gate(
        start_memory,
        after_trim_memory,
        tolerance_bytes=tolerance_bytes,
    )
    return {
        "schema": "pairfold128_cache_257_memory_gate_v1",
        "status": "passed" if return_gate["passed"] else "failed",
        "packing": "pairfold128",
        "backend": "EasyFHE CKKS",
        "device": context.device,
        "parameters": {
            "tokens": GATE_TOKENS,
            "page_size": PAIRFOLD_PAGE_SIZE,
            "max_page_eviction": False,
            "kv_limbs": kv_limbs,
            "encryption_level": level,
            "log_n": int(args.log_n),
            "max_levels": int(args.max_levels),
            "dnum": int(args.dnum),
            "dcrt_bits": int(args.dcrt_bits),
            "first_mod": int(args.first_mod),
        },
        "context": {
            "L": int(context.context.L),
            "keygen_wall_seconds": float(context.keygen_wall_time_s),
            "construction_wall_seconds": float(context_seconds),
        },
        "warmup": {
            "tokens": 2,
            "includes_encrypt": True,
            "includes_transactional_homo_add": True,
            "released_and_trimmed_before_start": True,
        },
        "boundaries": boundaries,
        "final_cache": {
            **dict(final_inventory or {}),
            "cipher_metadata": cipher_metadata,
        },
        "memory": {
            "start": start_memory,
            "peak": peak_memory,
            "after_release": after_release_memory,
            "after_trim": after_trim_memory,
        },
        "return_gate": return_gate,
        "timing_seconds": {
            "append_257": float(append_seconds),
            "release_gc_sync": float(release_seconds),
            "allocator_trim": float(trim_seconds),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-n", type=int, default=16)
    parser.add_argument("--max-levels", type=int, default=12)
    parser.add_argument("--dnum", type=int, default=3)
    parser.add_argument("--dcrt-bits", type=int, default=59)
    parser.add_argument("--first-mod", type=int, default=60)
    parser.add_argument("--kv-limbs", type=int, default=DEFAULT_KV_LIMBS)
    parser.add_argument(
        "--tolerance-mib",
        type=float,
        default=float(DEFAULT_TOLERANCE_MIB),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/pairfold128_cache_257_memory_gate.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_gate(args)
    _write_report(report, args.output_json)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
