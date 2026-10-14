#!/usr/bin/env python3
"""T2 single transformer layer (L10) on an N-card host: the full per-rank station chain, end to end.

The whole layer dataflow under the T2 schedule (one process per rank, every
BS inside the owning card).  The 16 FM ciphertexts are split into
``--world-size`` ranks of ``16 / world_size`` adjacent ciphertexts (one T2
*slot* = two ciphertexts = one PairFM source shard = one W_O carrier,
:mod:`llama3fhe.parallel.prefill.ownership`):

* ``--world-size 8`` (default): one slot per rank -- one rank per card on an
  8-card host (8x4090, 8xA100, 8xRTX PRO 6000);
* ``--world-size 4``: two slots (four FM ciphertexts) per rank -- one rank
  per card on a 4-card host, same code path, linear weights encoded once per
  card;
* ``--ranks-per-device 2`` (with the default world 8): the older 4-card way,
  two one-slot ~40 GiB ranks per 96 GB card;
* ``--emulate-single-device``: all ranks as threads of one process on one card.

Cross-rank traffic goes through the transport selected by ``--transport``
(``host`` = D2H -> NumPy -> shared memory -> H2D, or ``pinned`` = D2H straight
into a pinned shared-memory arena -> H2D; the EasyFHE torch shim has neither
NCCL nor CUDA IPC and the 4090s have no P2P, so both are host-staged).

    rmsnorm_in -> qkv -> attention_input_exchange -> attention_core_station
    -> pairfm_exchange -> w_o_station -> residual -> rmsnorm_post -> mlp -> residual

Per rank (all ranks run concurrently, one CKKS context per card):

1. ``build_worker_state`` with the T0 tiers + ``linear`` + the four
   ``attn_input_*`` sub-tiers + ``attn_qk`` / ``attn_pv`` / ``softmax`` +
   ``pairfm_output``.
2. Encrypt this rank's FM ciphertexts of one shared random hidden state
   (random Llama-3-8B shaped weights, ``WEIGHT_SCALE``); the rest is the
   station chain of ``llama3fhe.parallel.prefill.stations`` with the leases each
   station needs (``linear`` for QKV / W_O / MLP, the attention sub-tiers
   inside the exchange and core stations, ``pairfm_output`` inside
   ``pairfm_exchange``).
3. Every station records wall, device peak / after, resident cache, the key
   traffic of its stage (tracer), the BS count and the exit limbs / levels;
   the two exchanges also record their communication bytes and stage split.
4. ``--stop-after <station>`` ends the chain early (staged bring-up);
   ``--verify`` decrypts the station outputs and reports max / mean |value|
   and finiteness (no plaintext oracle yet: random weights, first goal is the
   dataflow).
5. ``--repeat N`` runs the chain N times on fresh encryptions of the same
   input; the first pass pays the one-off costs (``linear`` tier upload, BS
   warm-up), the last pass is the steady state reported in ``stations``.
   Every ``refresh`` of the rank's ``BootstrapOperator`` is timed through a
   proxy (``bootstrap_seconds`` / ``bootstrap_records`` per station), so a
   station's wall splits into BS / key H2D / evict / transport wait / compute.
   ``--barrier-each`` puts a barrier in front of every station so that the
   per-station walls are not polluted by rank skew (the layer wall then
   includes the barrier waits; run without it for the true pipeline time).
6. ``--real-weights <fixture.npz>`` replaces the random weights and hidden state
   with one layer of the QuaRot checkpoint plus the plaintext hidden state that
   the poly model (``poly_prefill_layer_numpy`` over the previous layers of the
   canonical prefill128 fixture) produces for it; ``--verify`` then compares every
   decrypted FM checkpoint (rmsnorm_in, residual_attn, rmsnorm_post, layer output)
   with the same numpy oracle on this rank's token rows (``oracle`` per station,
   ``layer_output_error`` per rank).  Build the fixture with
   ``bench/make_layer_fixture.py`` on a host that holds the checkpoint.
7. ``gather_to(0)`` of the per-rank summary; rank 0 writes ``summary.json``.

Usage (from the repository root, idle cards)::

    # 8 cards, one rank each (8x4090 low-memory preset)
    python bench/t2_layer.py --preset a100-high --output artifacts/t2_layer/run1
    python bench/t2_layer.py --stop-after w_o --verify --output artifacts/t2_layer/run1_wo
    python bench/t2_layer.py --repeat 3 --barrier-each --output artifacts/t2_layer/run4_profile
    python bench/t2_layer.py --real-weights fixtures/layer10_fixture.npz --verify --output artifacts/t2_layer/run5_real
    # 8 x RTX PRO 6000 (96 GB, everything resident)
    python bench/t2_layer.py --preset a100-high --repeat 3 --barrier-each \
        --real-weights fixtures/layer10_fixture.npz --verify --output artifacts/t2_layer/pro6000_run1
    # 4 x RTX PRO 6000: one rank per card owning two slots (four FM ciphertexts)
    python bench/t2_layer.py --preset a100-high --world-size 4 --repeat 3 --barrier-each \
        --real-weights fixtures/layer10_fixture.npz --verify --output artifacts/t2_layer/pro6000_4card_w4
    # 4 x RTX PRO 6000, older way: eight one-slot ranks, two per card (== --devices 0 0 1 1 2 2 3 3)
    python bench/t2_layer.py --preset a100-high --ranks-per-device 2 --repeat 3 --barrier-each \
        --real-weights fixtures/layer10_fixture.npz --verify --output artifacts/t2_layer/pro6000_4card
    # one card pretending to be eight ranks
    python bench/t2_layer.py --preset h100-emu --emulate-single-device --repeat 3 \
        --real-weights fixtures/layer10_fixture.npz --verify --output artifacts/t2_emu/run1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llama3fhe.parallel.launch import expand_devices, spawn_workers  # noqa: E402
from llama3fhe.parallel.presets import preset_by_name  # noqa: E402
from llama3fhe.parallel.prefill.transport import DEFAULT_TRANSPORT, TRANSPORT_BACKENDS  # noqa: E402

#: Backends a spawned worker process can build from ``(rank, endpoints)`` alone.
PROCESS_TRANSPORTS = tuple(kind for kind in TRANSPORT_BACKENDS if kind != "inprocess")

SEED = 26092802
HIDDEN_SCALE = 0.3
COEFF_LAYER = 10
WEIGHT_SCALE = 0.02
INTERMEDIATE = 14336
TIERS = ("bootstrap", "rmsnorm", "linear", "attn_input_lane", "attn_input_q", "attn_input_k", "attn_input_v")
CORE_TIERS = ("attn_qk", "attn_pv", "softmax")
OUTPUT_TIERS = ("pairfm_output",)
STATIONS = ("rmsnorm_in", "qkv", "attn_input", "attn_core", "pairfm", "w_o", "residual_attn",
            "rmsnorm_post", "mlp", "residual_mlp")


# ------------------------------------------------- RMSNorm / Softmax knobs (preset default, environment override)
#: ``T2_*`` values that force the production table even when the preset names another one.
_PRODUCTION_TABLE = ("default", "none", "production", "off")


def _knob(env_name: str, preset_value) -> str:
    """The environment wins over the preset (``preset_value`` may be ``None``)."""

    value = os.environ.get(env_name)
    if value is not None and value.strip():
        return value.strip()
    return "" if preset_value is None else str(preset_value)


def _alt_table_path(env_name: str, preset_value):
    """``(path or None, spec)`` of the table to patch in: env ``T2_*`` > preset > production."""

    from llama3fhe.parallel.presets import polynomial_table_path

    spec = _knob(env_name, preset_value)
    if not spec or spec.lower() in _PRODUCTION_TABLE:
        return None, spec
    return polynomial_table_path(spec), spec


def rmsnorm_coeffs_table(default_path, preset=None):
    """Load the RMSNorm Chebyshev table, optionally patched from the preset / environment.

    ``preset.rmsnorm_coeffs`` (``rtx6000pro-high``:
    ``rmsnorm_poly_coeffs_reltol5e-3_noinc.json``) or ``T2_RMS_COEFFS=<json>``
    swaps the whole table (a bare file name resolves under
    ``assets/polynomials/``; ``T2_RMS_COEFFS=default`` forces the production
    table); ``T2_RMS_COEFFS_LAYERS="2,4,5,6"`` limits the swap to those layers
    (both the ``attn`` and ``ffn`` branch).  Returns ``(table, description)``
    where the description lands in the summary so a record is never ambiguous
    about which rsqrt degrees it ran.
    """

    from llama3fhe.approx.rmsnorm import load_rmsnorm_poly_coeffs

    table = load_rmsnorm_poly_coeffs(default_path)
    alt, _ = _alt_table_path("T2_RMS_COEFFS", getattr(preset, "rmsnorm_coeffs", None))
    if alt is None:
        return table, {"source": str(default_path), "patched_layers": []}
    alt_table = load_rmsnorm_poly_coeffs(alt)
    layers_env = os.environ.get("T2_RMS_COEFFS_LAYERS", "all").strip()
    if layers_env.lower() in ("", "all"):
        return alt_table, {"source": str(alt), "patched_layers": "all"}
    layers = sorted(parse_layer_set(layers_env))
    for branch in ("attn", "ffn"):
        for idx in layers:
            table["branches"][branch][str(idx)] = alt_table["branches"][branch][str(idx)]
    return table, {"source": str(default_path), "alt": str(alt), "patched_layers": layers}


def silu_coeffs_table(default_path, preset=None):
    """Load the SiLU Chebyshev table, optionally swapped from the preset / environment.

    ``preset.silu_coeffs`` or ``T2_SILU_COEFFS=<json>`` (bare name resolves under
    ``assets/polynomials/``; ``default`` = production) replaces the per-layer
    table; ``T2_SILU_COEFFS_LAYERS="29,30,31"`` limits the swap to those layers.
    The degree decides the MLP entry limbs (``mlp_input_limbs``), so the swap
    changes both the SiLU polynomial cost and the gate/up projection limbs.
    Returns ``(per_layer_table, description)``.
    """

    from llama3fhe.approx.silu import load_silu_poly_coeffs

    table = load_silu_poly_coeffs(default_path)["per_layer"]
    alt, _ = _alt_table_path("T2_SILU_COEFFS", getattr(preset, "silu_coeffs", None))
    if alt is None:
        return table, {"source": str(default_path), "patched_layers": []}
    alt_table = load_silu_poly_coeffs(alt)["per_layer"]
    layers_env = os.environ.get("T2_SILU_COEFFS_LAYERS", "all").strip()
    if layers_env.lower() in ("", "all"):
        return alt_table, {"source": str(alt), "patched_layers": "all"}
    layers = sorted(parse_layer_set(layers_env))
    for idx in layers:
        table[str(idx)] = alt_table[str(idx)]
    return table, {"source": str(default_path), "alt": str(alt), "patched_layers": layers}


def parse_layer_set(spec: str, *, num_layers: int = 32) -> set[int]:
    """``"all"`` / ``"3-31"`` / ``"0,1,2"`` / ``"0-2,14,31"`` -> set of layer indices."""

    spec = spec.strip().lower()
    if spec in ("", "all"):
        return set(range(num_layers))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


# ------------------------------------------------- Softmax A/B knobs (environment)
_SOFTMAX_EXP_FIELDS = ("coefficients", "degree", "fit_interval", "max_error", "max_error_target",
                       "ps_k", "ps_m", "fit_method", "method", "uniform_max_abs")


def softmax_coeffs_table(default_path, preset=None):
    """Load the Softmax Alg2 table, optionally patched from the preset / environment.

    ``preset.softmax_coeffs`` (``rtx6000pro-high``:
    ``softmax_poly_coeffs_joint_tol_v1.json``) or ``T2_SOFTMAX_COEFFS=<json>``
    names the artifact to patch in (a bare file name resolves under
    ``assets/polynomials/``; ``T2_SOFTMAX_COEFFS=default`` forces the
    production table).  ``preset.softmax_coeffs_parts`` /
    ``T2_SOFTMAX_PARTS`` (``exp,j1,j2``) limits the swap to those parts:
    ``exp`` = the per-layer exp Chebyshev series (only the polynomial fields, the
    ``U_hat`` / ``stats_min`` / ``k`` map stays the default), ``j1`` / ``j2`` =
    the per-layer ``invsqrt_alg2`` rsqrt series (whole ``by_layer`` entry),
    ``k`` = the per-layer Alg2 round count (``k_by_layer`` / ``layers[L].k``;
    needed by tables from ``bench/probes/softmax_k_ladder_ppl.py --export``).
    ``T2_SOFTMAX_LAYERS`` (default ``all``, e.g. ``3-31``) limits it to layers.
    Returns ``(artifact, description)``; the description goes into the summary.
    """

    from llama3fhe.approx.softmax import load_softmax_poly_coeffs

    table = load_softmax_poly_coeffs(default_path)
    alt, _ = _alt_table_path("T2_SOFTMAX_COEFFS", getattr(preset, "softmax_coeffs", None))
    if alt is None:
        return table, {"source": str(default_path), "patched_layers": []}
    alt_table = load_softmax_poly_coeffs(alt)
    parts_spec = _knob("T2_SOFTMAX_PARTS", getattr(preset, "softmax_coeffs_parts", None)) or "exp,j1,j2"
    parts = {p.strip().lower() for p in parts_spec.split(",") if p.strip()}
    unknown = parts - {"exp", "j1", "j2", "k"}
    if unknown:
        raise ValueError(f"T2_SOFTMAX_PARTS: unknown parts {sorted(unknown)} (allowed exp,j1,j2,k)")
    layers = sorted(parse_layer_set(os.environ.get("T2_SOFTMAX_LAYERS", "all")))
    for idx in layers:
        key = str(idx)
        if "exp" in parts and key in alt_table.get("layers", {}):
            src = alt_table["layers"][key]
            dst = table["layers"][key]
            for field in _SOFTMAX_EXP_FIELDS:
                if field in src:
                    dst[field] = src[field]
                elif field in dst:
                    del dst[field]
        for part, stage in (("j1", "j1_rough"), ("j2", "j2_precise")):
            if part not in parts:
                continue
            src = alt_table["invsqrt_alg2"][stage].get("by_layer", {})
            if key in src:
                table["invsqrt_alg2"][stage].setdefault("by_layer", {})[key] = src[key]
        if "k" in parts:
            alt_k = (alt_table.get("k_by_layer") or {}).get(key, alt_table.get("layers", {}).get(key, {}).get("k"))
            if alt_k is not None:
                table.setdefault("k_by_layer", {})[key] = int(alt_k)
                if key in table.get("layers", {}):
                    table["layers"][key]["k"] = int(alt_k)
    return table, {"source": str(default_path), "alt": str(alt), "parts": sorted(parts),
                   "patched_layers": layers if len(layers) < 32 else "all"}


def softmax_table_degrees(table) -> dict:
    """Per-layer ``(k, exp degree, j1 degree, j2 degree)`` of a Softmax table (for the summary)."""

    out = {}
    for key, row in table.get("layers", {}).items():
        j1 = table["invsqrt_alg2"]["j1_rough"].get("by_layer", {}).get(key, table["invsqrt_alg2"]["j1_rough"])
        j2 = table["invsqrt_alg2"]["j2_precise"].get("by_layer", {}).get(key, table["invsqrt_alg2"]["j2_precise"])
        k = (table.get("k_by_layer") or {}).get(key, row.get("k", table.get("k")))
        out[key] = {"k": int(k), "exp": int(row["degree"]), "j1": int(j1["degree"]), "j2": int(j2["degree"])}
    return out


def rmsnorm_exit_kwargs(layer_idx: int | None = None, coeffs: dict | None = None, preset=None) -> dict:
    """``rmsnorm_station`` keyword arguments for one layer.

    The mode comes from ``T2_RMS_EXIT`` in the environment, else
    ``preset.rmsnorm_exit_mode`` (``rtx6000pro-high``: ``slim`` on layers
    ``3-31``), else ``pair``.

    ``pair``: the production flow -- exit pair-BS on the normalised
    stream (2 calls per rank in world 4, 1 in world 8) and the adaptive slim
    refresh only when the rsqrt polynomial leaves < 4 limbs (layer 2 ffn).
    ``slim``: no exit pair-BS; instead the slim mean-square row is always
    bootstrapped after the rsqrt polynomial (1 call per rank regardless of
    how many ciphertexts the rank owns) and the normalised stream inherits the
    fresh level through the extract/scale multiply: 18 - 1 (extract) - 1
    (scale) = 16 limbs >= QKV 13 / MLP <= 15.  ``T2_RMS_SLIM_MIN_LIMBS``
    overrides the 16.

    ``T2_RMS_SLIM_LAYERS`` (else ``preset.rmsnorm_slim_exit_layers``, else
    ``all``; e.g. ``3-31``) restricts the slim
    exit to those layers -- the others keep the pair exit (the rsqrt row of
    layers 0-2 is 50-200x larger than the normalised stream, which hurts the
    slim BS).  ``T2_RMS_SLIM_ITERS`` (1|2) is the Meta-BTS pass count of the
    slim BS and ``T2_RMS_SLIM_PRESCALE`` (0 < f <= 1, or ``auto`` = the power
    of two that brings ``rsqrt(fit_lo)`` under 1 for the given ``coeffs``)
    scales the rsqrt row before that BS (undone in the extract plaintext;
    costs one limb *before* the BS, none after).  Both apply to the slim
    refresh in ``pair`` mode too.
    """

    mode = _rmsnorm_exit_mode(preset)
    kwargs: dict = {"exit_refresh": True}
    iters = int(os.environ.get("T2_RMS_SLIM_ITERS", "1"))
    if iters != 1:
        kwargs["slim_refresh_iterations"] = iters
    prescale_env = os.environ.get("T2_RMS_SLIM_PRESCALE", "1").strip().lower()
    if prescale_env == "auto":
        # power of two with f * max(rsqrt) <= 1 over the fit interval: rsqrt(lo + eps) is the row maximum
        prescale = 1.0
        if coeffs is not None:
            lo = float(coeffs["fit_interval"][0]) + float(coeffs.get("eps", 1e-5))
            prescale = 2.0 ** -math.ceil(max(0.0, math.log2(lo ** -0.5)))
    else:
        prescale = float(prescale_env)
    if prescale != 1.0:
        kwargs["slim_refresh_pre_scale"] = prescale
    if mode == "slim":
        slim_layers = parse_layer_set(_rmsnorm_slim_layers(preset))
        if layer_idx is None or int(layer_idx) in slim_layers:
            kwargs["exit_refresh"] = False
            kwargs["slim_refresh_min_limbs"] = int(os.environ.get("T2_RMS_SLIM_MIN_LIMBS", "16"))
    period_rows = getattr(preset, "rmsnorm_slim_period_rows", None)
    if period_rows is not None:
        kwargs["slim_period_rows"] = int(period_rows)
    return kwargs


def _rmsnorm_exit_mode(preset=None) -> str:
    mode = (_knob("T2_RMS_EXIT", getattr(preset, "rmsnorm_exit_mode", None)) or "pair").lower()
    if mode not in ("pair", "slim"):
        raise ValueError(f"T2_RMS_EXIT / preset.rmsnorm_exit_mode must be 'pair' or 'slim', got {mode!r}")
    return mode


def _rmsnorm_slim_layers(preset=None) -> str:
    return _knob("T2_RMS_SLIM_LAYERS", getattr(preset, "rmsnorm_slim_exit_layers", None)) or "all"


def rmsnorm_exit_description(preset=None) -> dict:
    """What ``rmsnorm_exit_kwargs`` will do, for the summary."""

    mode = _rmsnorm_exit_mode(preset)
    desc = {"mode": mode, "slim_iters": int(os.environ.get("T2_RMS_SLIM_ITERS", "1")),
            "slim_prescale": os.environ.get("T2_RMS_SLIM_PRESCALE", "1")}
    if mode == "slim":
        desc["slim_layers"] = sorted(parse_layer_set(_rmsnorm_slim_layers(preset)))
        desc["slim_min_limbs"] = int(os.environ.get("T2_RMS_SLIM_MIN_LIMBS", "16"))
    desc["slim_period_rows"] = getattr(preset, "rmsnorm_slim_period_rows", None)
    return desc


BS_CACHE_MODES = ("middle", "plain", "none")


def add_runtime_knob_args(parser: argparse.ArgumentParser) -> None:
    """CLI knobs shared by ``t2_layer.py`` and ``t2_model.py`` (A/B switches that leave the preset defaults alone)."""

    parser.add_argument("--bs-cache-mode", choices=BS_CACHE_MODES, default=None,
                        help="constant-cache mode of the rank's bootstrap program (preset default: plain on the "
                             "high tier, middle on the 4090).  ``plain`` keeps every CtoS/StoC/EvalMod constant "
                             "resident as a ready plaintext; ``middle`` caches the half-way representation; ``none`` recomputes.")
    parser.add_argument("--no-stage-sync", action="store_true",
                        help="T2_STAGE_SYNC=0 for the workers: skip the device synchronisations that only delimit "
                             "the operators' sub-stage timers (per linear chunk, per attention/RMSNorm stage, "
                             "around every bootstrap).  Station walls stay exact (the station wrapper still "
                             "synchronises its stream); the sub-stage split and the per-BS seconds become host "
                             "enqueue times and must not be read as device times.")
    overlap = parser.add_mutually_exclusive_group()
    overlap.add_argument("--attn-in-overlap", action="store_true",
                         help="T2_ATTN_IN_OVERLAP=1 for the workers: attention-input exchange in split-phase order "
                              "(lane all-gather issued before the V partials, partial all-to-all issued before Q/K; "
                              "llama3fhe.parallel.prefill.stations.attention_input_exchange).  Same ciphertexts, the "
                              "transfer and the rank skew hide behind the rank's own GPU work.  Preset default: on "
                              "for the high tier, off for the 4090.")
    overlap.add_argument("--no-attn-in-overlap", action="store_true",
                         help="T2_ATTN_IN_OVERLAP=0 for the workers: the blocking exchange order (A/B baseline).")
    parser.add_argument("--rms-slim-period", type=int, default=None, metavar="ROWS",
                        help="preset.rmsnorm_slim_period_rows: periodic slim pack of the RMSNorm mean squares "
                             "(power of two rows, e.g. 16) -> the slim refresh runs through a sparse-slot "
                             "bootstrap program (log_slots = log2(ROWS * tokens_per_cipher)) and the extract "
                             "broadcast needs log2(ROWS) rotations.  0 = legacy row pack / dense program.")
    smx = parser.add_mutually_exclusive_group()
    smx.add_argument("--softmax-slim-periodic", dest="softmax_slim_periodic", action="store_true", default=None,
                     help="preset.softmax_slim_periodic: periodic slim pack of the Softmax Alg2 row sums (pair i on "
                          "every Delta row == i mod P, P = Q pairs per rank) -> the 1-2 slim refreshes per Alg2 round "
                          "run through a sparse-slot bootstrap program (log_slots = log2(P * 2 * seq_len): 9 at W=8).")
    smx.add_argument("--no-softmax-slim-periodic", dest="softmax_slim_periodic", action="store_false",
                     help="legacy one-row-per-pair slim pack, dense slim refreshes (A/B baseline).")
    parser.add_argument("--level-budget", default=None, metavar="C2S,S2C",
                        help="preset.bootstrap.level_budget, e.g. 3,3 (preset default 5,5).  Fewer CtoS/StoC levels "
                             "= fewer BSGS passes per bootstrap (Slot-Coeff runs (3,3)); the program depth drops by "
                             "the same amount, so pair it with --depth to keep the post-bootstrap levels "
                             "(depth - 10 - c2s - s2c, 17 on the high tier).")
    parser.add_argument("--depth", type=int, default=None,
                        help="preset.depth (context levels; high tier 37).  --level-budget 3,3 --depth 33 keeps the "
                             "17 post-bootstrap levels with a 4-limb shorter tower (cheaper raise / CtoS / keys).")
    parser.add_argument("--dnum", type=int, default=None,
                        help="preset.dnum (key-switch digits; preset default 5, Slot-Coeff runs 3).  The Sylph BSGS "
                             "kernels behind HOLMES_BATCH_BS=1 need at most 3 digits at the raised modulus, i.e. dnum 3.")


def apply_runtime_knobs(args: argparse.Namespace, preset):
    """Apply :func:`add_runtime_knob_args` to ``preset`` / the worker environment; returns the preset."""

    from dataclasses import replace

    if getattr(args, "bs_cache_mode", None) is not None:
        preset = replace(preset, bootstrap=replace(preset.bootstrap, cache_mode=str(args.bs_cache_mode)))
    if getattr(args, "rms_slim_period", None) is not None:
        rows = int(args.rms_slim_period)
        preset = replace(preset, rmsnorm_slim_period_rows=None if rows <= 0 else rows)
    if getattr(args, "softmax_slim_periodic", None) is not None:
        preset = replace(preset, softmax_slim_periodic=bool(args.softmax_slim_periodic))
    if getattr(args, "level_budget", None) is not None:
        budget = tuple(int(v) for v in str(args.level_budget).split(","))
        if len(budget) != 2:
            raise SystemExit("--level-budget expects two integers, e.g. 3,3")
        preset = replace(preset, bootstrap=replace(preset.bootstrap, level_budget=budget))
    if getattr(args, "depth", None) is not None:
        preset = replace(preset, depth=int(args.depth))
    if getattr(args, "dnum", None) is not None:
        preset = replace(preset, dnum=int(args.dnum))
    if any(getattr(args, k, None) is not None for k in ("level_budget", "depth", "dnum")):
        print(f"[knobs] depth {preset.depth} dnum {preset.dnum} level_budget {preset.bootstrap.level_budget} "
              f"-> post-bootstrap levels {preset.bootstrap_post_levels}", flush=True)
    if getattr(args, "no_stage_sync", False):
        os.environ["T2_STAGE_SYNC"] = "0"
    if getattr(args, "attn_in_overlap", False):
        os.environ["T2_ATTN_IN_OVERLAP"] = "1"
    elif getattr(args, "no_attn_in_overlap", False):
        os.environ["T2_ATTN_IN_OVERLAP"] = "0"
    return preset


def stage_sync_requested() -> bool:
    """What the spawned workers will see for ``T2_STAGE_SYNC`` (set by the CLI or inherited)."""

    return os.environ.get("T2_STAGE_SYNC", "1") != "0"


def attn_in_overlap_requested(preset=None) -> bool:
    """Whether the attention-input exchange runs in split-phase order: ``T2_ATTN_IN_OVERLAP`` (set by the CLI
    or inherited) wins, else ``preset.attention_input_overlap`` (high tier on, 4090 off), else off."""

    from llama3fhe.parallel.prefill.stations import attention_input_overlap_enabled

    return attention_input_overlap_enabled(default=bool(getattr(preset, "attention_input_overlap", False)))


def runtime_knob_summary(preset) -> dict:
    """The runtime-knob fields every ``summary.json`` carries (same keys in t2_layer and t2_model)."""

    return {
        "bs_cache_mode": str(preset.bootstrap.cache_mode),
        "stage_sync": stage_sync_requested(),
        "attn_in_overlap": attn_in_overlap_requested(preset),
        "rms_slim_period_rows": getattr(preset, "rmsnorm_slim_period_rows", None),
        "softmax_slim_periodic": bool(getattr(preset, "softmax_slim_periodic", False)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="a100-high")
    add_runtime_knob_args(parser)
    parser.add_argument("--input-level", type=int, default=35,
                        help="levels consumed by the FM inputs entering rmsnorm_in (35 -> 3 limbs)")
    parser.add_argument("--hidden-scale", type=float, default=HIDDEN_SCALE)
    parser.add_argument("--coeff-layer", type=int, default=COEFF_LAYER)
    parser.add_argument("--qkv-limbs", type=int, default=None,
                        help="QKV entry modulus (default stations.QKV_INPUT_LIMBS; 0 = no drop)")
    parser.add_argument("--mlp-limbs", type=int, default=None,
                        help="MLP entry modulus (default stations.mlp_input_limbs(silu_entry); 0 = no drop)")
    parser.add_argument("--stop-after", choices=STATIONS, default=None,
                        help="end the chain after this station (staged bring-up)")
    parser.add_argument("--prefetch-keys", action="store_true",
                        help="prefetch each sub-tier at lease start instead of loading on first use")
    parser.add_argument("--world-size", type=int, default=None,
                        help="number of ranks over the 16-ciphertext layout (default preset.world_size = 8: one T2 "
                             "slot = two FM ciphertexts per rank; 4: two slots = four ciphertexts per rank, one "
                             "process per card on a 4-card host).  Must divide 16 with an even quotient; the "
                             "preset's mlp_feature_shard_k is clamped to a divisor of the new world size unless "
                             "--mlp-shard-k is given")
    parser.add_argument("--devices", type=int, nargs="*", default=None,
                        help="physical CUDA indices for ranks 0..world_size-1; repeats put several ranks on one card "
                             "(then also pass the matching --ranks-per-device); ignored with --emulate-single-device")
    parser.add_argument("--ranks-per-device", type=int, default=1,
                        help="ranks sharing each physical card (2 on a 4-card 96 GB host: cards 0..3 each run two "
                             "resident ~40 GiB ranks); default devices = each of world_size/N cards repeated N times")
    parser.add_argument("--transport", choices=PROCESS_TRANSPORTS, default=None,
                        help="cross-rank transport backend for the worker processes (T2_TRANSPORT; default: the "
                             "preset's, 'pinned' on the 4090 = D2H -> pinned /dev/shm arena -> H2D, otherwise "
                             f"{DEFAULT_TRANSPORT} = D2H -> NumPy -> /dev/shm -> H2D; 'inprocess' is what "
                             "--emulate-single-device uses internally)")
    parser.add_argument("--verify", action="store_true",
                        help="decrypt every station output on every rank and report magnitudes")
    parser.add_argument("--real-weights", default=None,
                        help="layer fixture .npz (bench/make_layer_fixture.py): real weights + poly-model hidden state; "
                             "--verify then compares against the numpy oracle")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the station chain this many times; the last pass is the steady state")
    parser.add_argument("--barrier-each", action="store_true",
                        help="barrier before every station (per-station walls without rank skew)")
    parser.add_argument("--output", default=None,
                        help="directory for summary.json (default artifacts/t2_layer_<timestamp>)")
    parser.add_argument("--max-plaintext-rows", type=int, default=None,
                        help="override preset.linear_max_plaintext_rows for QKV / W_O / MLP (2048 rtx6000pro-high at "
                             "one slot per card, 512 otherwise on the high tier, 256 4090); applied after --world-size "
                             "/ --ranks-per-device, so it also overrides their multi-slot fallback to 512")
    parser.add_argument("--prefetch-mode", choices=("lazy", "sync", "async"), default=None,
                        help="override preset.key_prefetch_mode; async = side-stream upload of the next tier while the "
                             "current station computes (first iteration learns the limbs, so use --repeat >= 2)")
    parser.add_argument("--prefetch-margin-gib", type=float, default=None,
                        help="override preset.key_prefetch_margin_gib (free device GiB an async issue must leave)")
    parser.add_argument("--mlp-shard-k", type=int, default=None,
                        help="override preset.mlp_feature_shard_k (1 = token shard only; 2 = pairs of ranks split "
                             "the intermediate pages, Megatron-style; must divide the world size)")
    parser.add_argument("--transform-hoist", choices=("ext_normal", "normal"), default=None,
                        help="override preset.transform_hoist_strategy (LLAMA_FHE_TRANSFORM_HOIST for the workers: "
                             "ext_normal = extended-basis mask plaintexts, normal = plain plaintexts + moddown)")
    parser.add_argument("--evict-sync", choices=("device", "stream"), default=None,
                        help="override preset.key_evict_sync: device = EasyFHE clear_cuda_rotation_cache (full sync + "
                             "empty_cache), stream = compute-stream sync only (does not join an async prefetch copy)")
    parser.add_argument("--emulate-single-device", action="store_true",
                        help="play all world_size ranks on the one visible card: one process, one shared context "
                             "(all-rank key union resident), one lock-stepped thread per rank "
                             "(llama3fhe.parallel.emulation); per-rank station compute = wall - baton wait, the "
                             "emulated layer time is the per-station max over ranks (requires key_policy=resident)")
    parser.add_argument("--comm-gbps", type=float, default=None,
                        help="with --emulate-single-device: model the communication of every station as "
                             "bytes_sent_max_over_ranks / (this many GB/s) and add it to the emulated layer time")
    return parser.parse_args()


def worker_tiers(active) -> tuple[str, ...]:
    """Key tiers the station chain ``active`` needs (same for every rank)."""

    tiers = TIERS
    if "attn_core" in active:
        tiers = tiers + CORE_TIERS
    if "pairfm" in active:
        tiers = tiers + OUTPUT_TIERS
    return tiers


def _lease_rows(records):
    rows = []
    for r in records:
        row = {"tier": getattr(r, "tier", None), "prefetch_keys": getattr(r, "prefetched_keys", None),
               "prefetch_s": getattr(r, "prefetch_seconds", None), "evict_s": getattr(r, "evict_seconds", None),
               "evicted_entries": getattr(r, "evicted_entries", None),
               "retained_entries": getattr(r, "retained_entries", None)}
        if getattr(r, "async_keys", 0) or getattr(r, "async_issued_tier", None):
            row.update({
                "async_keys": getattr(r, "async_keys", 0), "async_gib": round(float(getattr(r, "async_gib", 0.0)), 4),
                "async_wait_s": round(float(getattr(r, "async_wait_seconds", 0.0)), 4),
                "async_issued_tier": getattr(r, "async_issued_tier", None),
                "async_issued_keys": getattr(r, "async_issued_keys", 0),
                "async_issued_gib": round(float(getattr(r, "async_issued_gib", 0.0)), 4),
                "async_issue_s": round(float(getattr(r, "async_issue_seconds", 0.0)), 4),
                "async_skipped": getattr(r, "async_skipped", None),
            })
        rows.append(row)
    return rows


def load_layer_fixture(path):
    """Real weights + poly-model hidden state written by ``bench/make_layer_fixture.py``."""

    import numpy as np

    with np.load(path) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def layer_oracle(hidden, fixture, *, coeffs_attn, coeffs_ffn, silu_entry, softmax_config, attention_layout,
                 layer_idx, rope_theta):
    """The plaintext checkpoints the encrypted stations must reproduce (same maths as ``poly_prefill_layer_numpy``)."""

    import numpy as np

    from llama3fhe.approx.layer import (
        _attention_out_from_probs,
        _content_causal_valid_mask,
        _gqa_unscaled_scores,
        _project_qkv,
        _rmsnorm_poly,
        _rope_qkv,
        _silu_poly,
        softmax_alg2_poly_scores_numpy,
    )
    from llama3fhe.graph import LayerWeights

    weights = LayerWeights(
        input_layernorm=fixture["input_layernorm"], q_proj=fixture["q_proj"], k_proj=fixture["k_proj"],
        v_proj=fixture["v_proj"], o_proj=fixture["o_proj"], post_attention_layernorm=fixture["post_attention_layernorm"],
        gate_proj=fixture["gate_proj"], up_proj=fixture["up_proj"], down_proj=fixture["down_proj"],
    )
    residual = np.asarray(hidden, dtype=np.float32)
    normed_in = _rmsnorm_poly(residual, coeffs_attn, None)
    attn_in = _rmsnorm_poly(residual, coeffs_attn, weights.input_layernorm)
    query, key, value = _project_qkv(attn_in, weights)
    query, key = _rope_qkv(query, key, layout=attention_layout, rope_theta=float(rope_theta), start_pos=0)
    scores, _, _ = _gqa_unscaled_scores(query, key, layout=attention_layout)
    valid = _content_causal_valid_mask(int(residual.shape[0]), token_valid_mask=None)
    probabilities, _report = softmax_alg2_poly_scores_numpy(
        scores, config=softmax_config, layer_idx=int(layer_idx), head_dim=int(attention_layout.head_dim), valid=valid)
    attn_out = _attention_out_from_probs(probabilities, value, layout=attention_layout)
    w_o_out = attn_out @ np.asarray(weights.o_proj, dtype=np.float32).T
    residual_attn = residual + w_o_out
    normed_post = _rmsnorm_poly(residual_attn, coeffs_ffn, None)
    mlp_in = _rmsnorm_poly(residual_attn, coeffs_ffn, weights.post_attention_layernorm)
    gate = mlp_in @ np.asarray(weights.gate_proj, dtype=np.float32).T
    up = mlp_in @ np.asarray(weights.up_proj, dtype=np.float32).T
    mlp_out = (_silu_poly(gate, silu_entry) * up) @ np.asarray(weights.down_proj, dtype=np.float32).T
    layer_out = residual_attn + mlp_out
    return {
        "rmsnorm_in": normed_in, "attn_in": attn_in, "query": query, "key": key, "value": value,
        "attn_out": attn_out, "w_o": w_o_out, "residual_attn": residual_attn, "rmsnorm_post": normed_post,
        "mlp": mlp_out, "layer_output": layer_out, "gate_max_abs": float(np.max(np.abs(gate))),
        "scores_max_abs": float(np.max(np.abs(scores))),
    }


def _decrypt_complex(context, ciphers):
    """Decrypt ciphers keeping the imaginary slots (verification only), [n, slots] complex128."""

    from llama3fhe.operators.attention.codec import decrypt_complex_rows

    return decrypt_complex_rows(tuple(ciphers), crypto_context=context)


def attention_input_oracle(oracle, attention_layout, *, query_items, key_items, value_items, context):
    """Own Q pairs / K pair / V pairs against the monolithic adapter's plaintext packing.

    Q: ``pack_query_c_feature_folded`` (0.5*(x + i*x_paired), RoPE applied);
    K: ``pack_key_delta_feature_folded``; V: the finish step splits the complex
    carrier into ``2*Re`` / ``2*Im``; V pairs use the ``pack_value_delta`` scale (1x).
    """

    import numpy as np

    out = {}
    q_pack = attention_layout.pack_query_c_feature_folded(oracle["query"])
    k_pack = attention_layout.pack_key_delta_feature_folded(oracle["key"])
    v_pack = attention_layout.pack_value_delta(oracle["value"])
    for name, items, packed, scale in (("query", query_items, q_pack, 1.0), ("key", key_items, k_pack, 1.0),
                                       ("value", value_items, v_pack, 1.0)):
        indices = [int(i) for i, _ in items]
        got = _decrypt_complex(context, [c for _, c in items])
        if name == "value":
            got = got.real
        out[name] = _compare(got, packed[indices], scale=scale)
        out[name]["pairs"] = indices
    return out


def attention_core_oracle(oracle, attention_layout, *, output_items, context):
    """Own ``O_C`` pairs against ``pack_output_c(attn_out)`` (real C-form, one pair per two heads)."""

    import numpy as np

    indices = [int(i) for i, _ in output_items]
    packed = attention_layout.pack_output_c(oracle["attn_out"])
    got = np.stack([np.asarray(context.decrypt(c), dtype=np.complex128).real for _, c in output_items])
    result = _compare(got, packed[indices])
    result["pairs"] = indices
    # imaginary leakage (should be ~0 for a real C-form output)
    imag = _decrypt_complex(context, [c for _, c in output_items]).imag
    result["imag_max_abs"] = float(np.max(np.abs(imag)))
    return result


def pairfm_oracle(oracle, layout, attention_layout, *, carriers, carrier_indices, context):
    """Own W_O carriers against ``distributed_pairfm_carriers_numpy(pack_output_c(attn_out))[carrier_indices]``.

    Returns the worst (max-abs-error) carrier's comparison plus a per-carrier list.
    """

    import numpy as np

    from llama3fhe.operators.attention.distributed.distributed_output import distributed_pairfm_carriers_numpy

    packed = attention_layout.pack_output_c(oracle["attn_out"])
    expected = distributed_pairfm_carriers_numpy(packed, feature_layout=layout, attention_layout=attention_layout)
    got = _decrypt_complex(context, list(carriers))
    per_carrier = []
    for value, index in zip(got, carrier_indices, strict=True):
        entry = _compare(value, expected[int(index)])
        entry["carrier_index"] = int(index)
        per_carrier.append(entry)
    worst = max(per_carrier, key=lambda e: float(e.get("max_abs_err", 0.0)))
    result = dict(worst)
    result["carrier_indices"] = [int(v) for v in carrier_indices]
    result["per_carrier"] = per_carrier
    return result


def _decrypt_rows(context, layout, ownership, ciphers):
    """Decrypt this rank's FM ciphertexts and return its token rows [token_count, hidden]."""

    import numpy as np

    canvas = np.zeros((int(layout.cipher_count), int(layout.slots)), dtype=np.float64)
    for slot, index in zip(ownership.cipher_indices, ciphers):
        canvas[int(slot)] = np.asarray(context.decrypt(index), dtype=np.complex128).real
    return layout.unpack(canvas)[int(ownership.token_start):int(ownership.token_stop)]


def _compare(got, expected, *, scale=1.0):
    """Error of ``got`` against ``expected * scale`` (max / mean abs, relative to the oracle's max)."""

    import numpy as np

    expected = np.asarray(expected, dtype=np.float64) * float(scale)
    got = np.asarray(got, dtype=np.float64)
    diff = got - expected
    ref = float(np.max(np.abs(expected)))
    return {
        "max_abs_err": float(np.max(np.abs(diff))), "mean_abs_err": float(np.mean(np.abs(diff))),
        "rms_err": float(np.sqrt(np.mean(diff ** 2))), "oracle_max_abs": ref,
        "oracle_rms": float(np.sqrt(np.mean(expected ** 2))),
        "rel_max_err": float(np.max(np.abs(diff)) / max(ref, 1e-30)),
        "got_max_abs": float(np.max(np.abs(got))), "finite": bool(np.all(np.isfinite(got))),
        "scale": float(scale),
    }


def _magnitudes(context, named_ciphers):
    import numpy as np

    out = {}
    for name, cipher in named_ciphers:
        values = np.asarray(context.decrypt(cipher), dtype=np.complex128)
        out[str(name)] = {
            "max_abs": float(np.max(np.abs(values))),
            "mean_abs": float(np.mean(np.abs(values))),
            "finite": bool(np.all(np.isfinite(values))),
            "limbs": int(cipher.state.cur_limbs),
        }
    return out


class _BootstrapTimer:
    """Proxy around the rank's ``BootstrapOperator`` that times every ``refresh``.

    Stations receive this object as ``bootstrap_operator``; everything except
    ``refresh`` is forwarded, ``refresh`` synchronises and records
    ``(seconds, input limbs, iterations)`` so a station's BS seconds can be
    separated from its compute.
    """

    def __init__(self, inner, *, shared: bool = False):
        self._inner = inner
        self.records: list[tuple[float, int, int, int, str | None]] = []
        # Under the single-card emulation one ``BootstrapOperator`` serves
        # every rank thread, so the proxy counts its own ``refresh`` calls
        # instead of reading the operator's process-wide counter.
        self._shared = bool(shared)
        self._calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def calls(self) -> int:
        return self._calls if self._shared else int(self._inner.calls)

    def refresh(self, cipher, **kwargs):
        import easyfhe

        from llama3fhe.backend import stage_sync_enabled

        # Under T2_STAGE_SYNC=0 the record is the host enqueue time (the
        # operator skips its trailing sync as well); station walls stay exact.
        sync = stage_sync_enabled()
        if sync:
            easyfhe.cuda.synchronize()
        started = time.perf_counter()
        result = self._inner.refresh(cipher, **kwargs)
        if sync:
            easyfhe.cuda.synchronize()
        self._calls += 1
        from llama3fhe.operators.nonlinear.bootstrap import current_bootstrap_tag

        self.records.append((time.perf_counter() - started, int(cipher.state.cur_limbs),
                             int(kwargs.get("iterations") or self._inner.config.iterations),
                             int(result.state.cur_limbs), current_bootstrap_tag()))
        return result

    def refresh_many(self, ciphers, **kwargs):
        import easyfhe

        from llama3fhe.backend import stage_sync_enabled
        from llama3fhe.operators.nonlinear.bootstrap import current_bootstrap_tag

        ciphers = list(ciphers)
        sync = stage_sync_enabled()
        if sync:
            easyfhe.cuda.synchronize()
        started = time.perf_counter()
        results = list(self._inner.refresh_many(ciphers, **kwargs))
        if sync:
            easyfhe.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self._calls += len(results)
        iterations = int(kwargs.get("iterations") or self._inner.config.iterations)
        tag = current_bootstrap_tag()
        # one record per ciphertext, the batch wall shared evenly, so per-station BS seconds still add up
        for cipher, result in zip(ciphers, results):
            self.records.append((elapsed / max(1, len(results)), int(cipher.state.cur_limbs), iterations,
                                 int(result.state.cur_limbs), tag))
        return results

    def take(self) -> list[tuple[float, int, int, int, str | None]]:
        taken, self.records = self.records, []
        return taken


# ----------------------------------------------------------------- worker
def worker_entry(rank: int, preset, endpoints, *, input_level: int, verify: bool, prefetch_keys: bool,
                 qkv_limbs: int | None, mlp_limbs: int | None, stop_after: str | None, repeat: int = 1,
                 barrier_each: bool = False, hidden_scale: float = HIDDEN_SCALE, coeff_layer: int = COEFF_LAYER,
                 real_weights: str | None = None, emulation=None):
    """One rank's station chain.

    ``emulation`` (an :class:`llama3fhe.parallel.emulation.EmulatedRank`) replaces
    the per-process context build: the rank runs as a thread against the shared
    :class:`~llama3fhe.parallel.worker.WorkerState` and the in-process transport
    it carries; ``endpoints`` is ignored and the shared state is released by the
    caller, not here.
    """
    import numpy as np

    import easyfhe

    from llama3fhe.application.args import DEFAULT_RMS, DEFAULT_SILU, DEFAULT_SOFTMAX
    from llama3fhe.approx.attention import rope_cos_sin
    from llama3fhe.approx.rmsnorm import get_layer_coeffs
    from llama3fhe.approx.silu import load_silu_poly_coeffs
    from llama3fhe.approx.softmax import SoftmaxPolyConfig
    from llama3fhe.backend import release_all
    from llama3fhe.graph import _linear_config
    from llama3fhe.layouts.linear import LinearCarrierLayout
    from llama3fhe.operators.attention.config import AttentionOperatorConfig
    from llama3fhe.parallel.memory import native_cache_inventory
    from llama3fhe.parallel.prefill.stations import (
        QKV_INPUT_LIMBS,
        attention_core_operator_config,
        attention_core_station,
        attention_input_exchange,
        mlp_input_limbs,
        mlp_station,
        pairfm_exchange,
        qkv_station,
        residual_station,
        rmsnorm_station,
        w_o_station,
    )
    from llama3fhe.backend import plaintext_cache_snapshot
    from llama3fhe.parallel.prefill.transport import build_transport
    from llama3fhe.parallel.worker import build_worker_state, host_rss_gib, load_layer_config

    stop_index = len(STATIONS) if stop_after is None else STATIONS.index(str(stop_after)) + 1
    active = STATIONS[:stop_index]
    tiers = worker_tiers(active)

    emulated = emulation is not None
    if emulated:
        from dataclasses import replace as _replace

        from llama3fhe.parallel.memory import MemoryProbe

        transport = emulation.transport
        state = _replace(emulation.shared_state, rank=int(rank), transport=transport,
                         memory=MemoryProbe(rank=int(rank)))
        state.memory.reset_peak()
    else:
        transport = build_transport(rank, endpoints)
        state = build_worker_state(rank, preset, transport=transport, tiers=tiers, trace_keys=True)
    probe, context, layout, attention_layout = state.memory, state.context, state.feature_layout, state.attention_layout
    ownership, world, native, life, tracer = state.ownership, state.world, state.native, state.keys, state.tracer
    bootstrap = _BootstrapTimer(state.bootstrap_operator, shared=emulated)
    # Tracer stages are keyed by name; under the emulation eight ranks share one tracer.
    stage_prefix = f"r{rank}." if emulated else ""
    config = load_layer_config(preset)
    schedule = preset.layer_schedule()
    qkv_config = _linear_config(schedule, schedule.qkv_max_plaintext_rows, schedule.qkv_hoist_strategy)
    mlp_config = _linear_config(schedule, schedule.mlp_max_plaintext_rows, schedule.mlp_hoist_strategy)
    resident = state.resident_report()
    key_plan = state.key_plan.summary(alpha=int(native.alpha), special_limbs=int(native.K))

    # -- inputs: this rank's adjacent FM pair of one shared hidden state + shared weights
    #    (random by default; --real-weights loads one checkpoint layer + the poly model's hidden state)
    rms_all, rms_table_info = rmsnorm_coeffs_table(DEFAULT_RMS, preset)
    coeffs_attn = get_layer_coeffs(rms_all, "attn", int(coeff_layer))
    coeffs_ffn = get_layer_coeffs(rms_all, "ffn", int(coeff_layer))
    rms_exit = rmsnorm_exit_kwargs(int(coeff_layer), coeffs_attn, preset)
    rms_exit_post = rmsnorm_exit_kwargs(int(coeff_layer), coeffs_ffn, preset)
    silu_entry = load_silu_poly_coeffs(DEFAULT_SILU)["per_layer"][str(int(coeff_layer))]["tiers"]["standard"]
    hidden_dim = int(layout.hidden_dim)
    kv_dim = int(attention_layout.key_value_heads) * int(attention_layout.head_dim)
    fixture = None
    if real_weights:
        fixture = load_layer_fixture(real_weights)
        hidden = np.asarray(fixture["hidden_in"], dtype=np.float32)
        if hidden.shape != (int(layout.seq_len), hidden_dim):
            raise ValueError(f"fixture hidden_in has shape {hidden.shape}, layout wants {(layout.seq_len, hidden_dim)}")
        gamma_in = np.asarray(fixture["input_layernorm"], dtype=np.float32)
        gamma_post = np.asarray(fixture["post_attention_layernorm"], dtype=np.float32)
        q_w, k_w, v_w, o_w = (np.asarray(fixture[k], dtype=np.float32) for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
        gate_w, up_w, down_w = (np.asarray(fixture[k], dtype=np.float32) for k in ("gate_proj", "up_proj", "down_proj"))
    else:
        rng = np.random.default_rng(SEED)
        hidden = rng.normal(scale=float(hidden_scale), size=(int(layout.seq_len), hidden_dim)).astype(np.float32)
        gamma_in = gamma_post = np.ones(hidden_dim, dtype=np.float32)
        q_w = rng.normal(scale=WEIGHT_SCALE, size=(hidden_dim, hidden_dim)).astype(np.float32)
        k_w = rng.normal(scale=WEIGHT_SCALE, size=(kv_dim, hidden_dim)).astype(np.float32)
        v_w = rng.normal(scale=WEIGHT_SCALE, size=(kv_dim, hidden_dim)).astype(np.float32)
        o_w = rng.normal(scale=WEIGHT_SCALE, size=(hidden_dim, hidden_dim)).astype(np.float32)
        gate_w = rng.normal(scale=WEIGHT_SCALE, size=(INTERMEDIATE, hidden_dim)).astype(np.float32)
        up_w = rng.normal(scale=WEIGHT_SCALE, size=(INTERMEDIATE, hidden_dim)).astype(np.float32)
        down_w = rng.normal(scale=WEIGHT_SCALE, size=(hidden_dim, INTERMEDIATE)).astype(np.float32)
    packed = layout.pack(hidden)

    fuse_silu = bool(schedule.fuse_silu_domain_map)
    mlp_entry = (mlp_input_limbs(silu_entry, fuse_silu_domain_map=fuse_silu)
                 if mlp_limbs is None else (int(mlp_limbs) or None))
    qkv_entry = QKV_INPUT_LIMBS if qkv_limbs is None else (int(qkv_limbs) or None)

    carrier_layout = LinearCarrierLayout(seq_len=int(layout.seq_len), dimension=int(layout.hidden_dim),
                                         slots=int(layout.slots))
    cos, sin = rope_cos_sin(seq_len=int(attention_layout.seq_len), head_dim=int(attention_layout.head_dim),
                            theta=float(AttentionOperatorConfig().rope_theta), start_pos=0)
    operator_config = AttentionOperatorConfig(
        sparse_max_baby_offsets=int(schedule.sparse_max_baby_offsets),
        shear_baby_steps=int(schedule.shear_baby_steps),
        pv_complex_delta_folding=bool(schedule.pv_complex_delta_folding),
        pv_prefold_after_softmax=bool(schedule.pv_prefold_after_softmax),
        batched_sparse_input_bsgs=bool(schedule.batched_sparse_attention_input),
        qk_complex_feature_folding=True,
        qk_key_features_pre_folded=True,
        qk_query_features_pre_scaled=True,
        qk_query_features_pre_folded=True,
        # preset decides (rtx6000pro-high / a100-high / h100-emu: 1 pass, 4090: 2);
        # T2_PAIRFM_BS_ITERATIONS in the environment overrides for A/B.
        pairfm_refresh_bootstrap_iterations=int(os.environ.get(
            "T2_PAIRFM_BS_ITERATIONS", str(int(preset.pairfm_refresh_bootstrap_iterations)))),
        softmax_slim_periodic=bool(getattr(preset, "softmax_slim_periodic", False)),
    )
    core_config = attention_core_operator_config(schedule, layer_idx=int(coeff_layer), base=operator_config)
    softmax_table, softmax_table_info = softmax_coeffs_table(DEFAULT_SOFTMAX, preset)
    softmax_config = SoftmaxPolyConfig.from_artifact(softmax_table)
    w_o_config = _linear_config(schedule, schedule.output_max_plaintext_rows, schedule.output_hoist_strategy)
    oracle = None
    oracle_seconds = 0.0
    if fixture is not None and verify:
        started = time.perf_counter()
        oracle = layer_oracle(
            hidden, fixture, coeffs_attn=coeffs_attn, coeffs_ffn=coeffs_ffn, silu_entry=silu_entry,
            softmax_config=softmax_config, attention_layout=attention_layout, layer_idx=int(coeff_layer),
            rope_theta=float(AttentionOperatorConfig().rope_theta))
        oracle_seconds = time.perf_counter() - started
        rows = slice(int(ownership.token_start), int(ownership.token_stop))
        reference_out = np.asarray(fixture["hidden_out"], dtype=np.float32)
        oracle_self_check = _compare(oracle["layer_output"], reference_out)

    def done(name: str) -> bool:
        return name not in active

    def _stage_dict(values) -> dict[str, float]:
        return {str(k): round(float(v), 4) for k, v in dict(values or {}).items()}

    def run_layer(iteration: int) -> tuple[dict[str, dict], dict[str, dict], float]:
        """One full pass of the station chain; returns (stations, magnitudes, wall)."""

        stations: dict[str, dict] = {}
        magnitudes: dict[str, dict] = {}
        order = [20]
        prefix = f"t2l.r{rank}.i{iteration}"

        def timed(name: str, tier: str | None, fn, *, barrier: bool = False):
            if barrier or barrier_each:
                transport.barrier(f"layer/{iteration}/before/{name}")
            probe.reset_peak()
            easyfhe.cuda.synchronize()
            calls_before = bootstrap.calls
            bootstrap.take()
            transport_before = transport.stats.to_dict()
            plain_before = plaintext_cache_snapshot(context)
            started = time.perf_counter()
            with tracer.stage(f"{stage_prefix}{name}_{iteration}"):
                if tier is None:
                    value = fn()
                else:
                    with life.lease(tier, prefetch=bool(prefetch_keys)) as record:
                        value = fn()
                    value = (value, record)
            easyfhe.cuda.synchronize()
            wall = time.perf_counter() - started
            lease_record = None
            if tier is not None:
                value, lease_record = value
            row = probe.snap(f"{order[0]} {name}_{iteration}", {"wall_s": round(wall, 4)})
            order[0] += 1
            stats = tracer.stages[f"{stage_prefix}{name}_{iteration}"].as_row()
            stats["miss_log"] = [list(v) for v in tracer.stages[f"{stage_prefix}{name}_{iteration}"].miss_log[:256]]
            transport_after = transport.stats.to_dict()
            plain_after = plaintext_cache_snapshot(context)
            bs_records = bootstrap.take()
            stations[name] = {
                "wall_s": wall,
                "tier": tier,
                "bootstrap_calls": bootstrap.calls - calls_before,
                "bootstrap_seconds": round(sum(r[0] for r in bs_records), 4),
                "bootstrap_records": [{"s": round(r[0], 4), "in_limbs": r[1], "iterations": r[2], "out_limbs": r[3], "tag": r[4]} for r in bs_records],
                "peak_alloc_gib": row.get("peak_alloc_gib"),
                "alloc_after_gib": row.get("alloc_gib"),
                "reserved_after_gib": row.get("reserved_gib"),
                "alloc_retries": row.get("alloc_retries"),
                "cache_after_gib": native_cache_inventory(native)["total_gib"],
                "host_rss_gib": round(host_rss_gib(), 2),
                "keys": {k: stats.get(k) for k in ("requests", "misses", "gib_loaded", "load_seconds",
                                                     "max_limbs", "distinct_rotations")},
                "transport_delta": {k: round(transport_after[k] - transport_before[k], 4)
                                    for k in ("recv_wait_seconds", "send_seconds", "bytes_sent", "bytes_received")},
                "leases": _lease_rows([lease_record]) if lease_record is not None else [],
                # session plaintext cache (public masks): per-station deltas + running bytes
                "plain_cache": {**{k: int(plain_after[k] - plain_before[k]) for k in ("hits", "misses", "skips")},
                                "entries": int(plain_after["entries"]), "gib": round(plain_after["bytes"] / 2**30, 3)},
            }
            return value

        def record_exit(name: str, ciphers, label: str | None = None):
            ciphers = tuple(ciphers)
            stations[name]["output_limbs"] = [int(c.state.cur_limbs) for c in ciphers]
            stations[name]["output_levels"] = [int(context.level_for_cipher(c)) for c in ciphers]
            if verify:
                magnitudes.update(_magnitudes(context, [(f"{label or name}/{i}", c) for i, c in enumerate(ciphers)]))

        inputs = tuple(
            context.encrypt(packed[index], level=int(input_level), slots=int(layout.slots))
            for index in ownership.cipher_indices
        )
        easyfhe.cuda.synchronize()
        probe.snap(f"10 inputs_{iteration}", {"input_limbs": int(inputs[0].state.cur_limbs)})
        transport.barrier(f"layer/{iteration}/start")
        layer_started = time.perf_counter()

        keep: dict[str, object] = {}
        try:
            # -- 1. RMSNorm (entry) on T0 keys; keeps the refreshed residual branch
            rms = timed("rmsnorm_in", None, lambda: rmsnorm_station(
                inputs, gamma_in, coeffs=coeffs_attn, ownership=ownership, layout=layout, context=context,
                bootstrap_operator=bootstrap, keep_residual=True, **rms_exit,
                function_prefix=f"{prefix}.rms_in"))
            keep["rms"] = rms
            record_exit("rmsnorm_in", rms.ciphers)
            if oracle is not None:
                # cipher = norm(x) * (gamma / deferred_output_scale) * value_scale  (gamma deferred to QKV)
                deferred = np.asarray(rms.deferred_output_scale, dtype=np.float64).reshape(1, -1)
                stations["rmsnorm_in"]["oracle"] = _compare(
                    _decrypt_rows(context, layout, ownership, rms.ciphers),
                    oracle["rmsnorm_in"][rows] * (gamma_in.astype(np.float64).reshape(1, -1) / deferred),
                    scale=float(rms.value_scale))
                stations["rmsnorm_in"]["oracle_residual"] = _compare(
                    _decrypt_rows(context, layout, ownership, rms.residual_inputs), hidden[rows])
            stations["rmsnorm_in"]["residual_limbs"] = [int(c.state.cur_limbs) for c in rms.residual_inputs]
            stations["rmsnorm_in"]["stage_seconds"] = _stage_dict(rms.stage_seconds)
            stations["rmsnorm_in"]["inner_stage_seconds"] = _stage_dict(rms.level_trace.get("rmsnorm_stage_seconds"))
            release_all(inputs)
            inputs = ()
            if done("qkv"):
                raise StopIteration

            # -- 2. QKV token shard under the linear tier
            qkv = timed("qkv", "linear", lambda: qkv_station(
                rms.ciphers, hidden, q_w, k_w, v_w, ownership=ownership, layout=layout, attention_layout=attention_layout,
                context=context, config=config, operator_config=qkv_config,
                input_column_scale=rms.linear_input_column_scale(),
                input_limbs=qkv_entry, function_prefix=f"{prefix}.qkv"))
            keep["qkv"] = qkv
            record_exit("qkv", qkv.carrier_ciphers)
            stations["qkv"]["input_limbs"] = {"before_drop": list(rms.output_limbs), "entry": qkv_entry}
            stations["qkv"]["stage_seconds"] = _stage_dict(qkv.stage_seconds)
            _release_ciphers(rms.ciphers)
            rms.ciphers = ()
            if done("attn_input"):
                raise StopIteration

            # -- 3. attention-input exchange (lane -> V partial -> all_gather + all_to_all -> finish -> Q -> K)
            attn_in = timed("attn_input", None, lambda: attention_input_exchange(
                qkv.carrier_ciphers, ownership=ownership, world=world, transport=transport,
                carrier_layout=carrier_layout, attention_layout=attention_layout, context=context, cos=cos, sin=sin,
                model_config=config, operator_config=operator_config, keys=life, prefetch_keys=bool(prefetch_keys),
                overlap=attn_in_overlap_requested(preset),
                function_prefix=f"{prefix}.attn_in", tag=f"layer/{iteration}/attn_in"), barrier=True)
            keep["attn_in"] = attn_in
            stations["attn_input"].update({
                "stage_seconds": _stage_dict(attn_in.stage_seconds),
                "comm_gib": {k: round(int(v) / float(2**30), 4) for k, v in attn_in.comm_bytes.items()},
                "leases": _lease_rows(attn_in.lease_records),
                "scheduler_stats": {k: float(v) for k, v in attn_in.scheduler_stats.items()},
                "output_limbs": {
                    "query": [int(c.state.cur_limbs) for _, c in attn_in.query_items],
                    "key": [int(c.state.cur_limbs) for _, c in attn_in.key_items],
                    "value": [int(c.state.cur_limbs) for _, c in attn_in.value_items],
                },
            })
            if oracle is not None:
                stations["attn_input"]["oracle"] = attention_input_oracle(
                    oracle, attention_layout, query_items=attn_in.query_items, key_items=attn_in.key_items,
                    value_items=attn_in.value_items, context=context)
            qkv.release()
            if done("attn_core"):
                raise StopIteration

            # -- 4. attention core (QK -> softmax -> PV) on the own Q pairs x own K/V pair
            core = timed("attn_core", None, lambda: attention_core_station(
                attn_in.query_items, attn_in.key_items, attn_in.value_items, ownership=ownership,
                attention_layout=attention_layout, context=context, softmax_config=softmax_config,
                operator_config=core_config, bootstrap_operator=bootstrap, layer_idx=int(coeff_layer),
                keys=life, prefetch_keys=bool(prefetch_keys), function_prefix=f"{prefix}.core"), barrier=True)
            keep["core"] = core
            record_exit("attn_core", [c for _, c in core.output_items], label="attn_core_o_c")
            stations["attn_core"].update({
                "stage_seconds": _stage_dict(core.stage_seconds),
                "level_trace": {k: list(v) for k, v in core.level_trace.items()},
                "output_pairs": [int(i) for i, _ in core.output_items],
                "leases": _lease_rows(core.lease_records),
            })
            if oracle is not None:
                stations["attn_core"]["oracle"] = attention_core_oracle(
                    oracle, attention_layout, output_items=core.output_items, context=context)
            attn_in.release()
            if done("pairfm"):
                raise StopIteration

            # -- 5. PairFM: own O_C -> 8 partials (per slot, pre-added) -> all-to-all -> own W_O carriers
            #       (pairfm_output lease inside)
            pf = timed("pairfm", None, lambda: pairfm_exchange(
                core.output_items, ownership=ownership, world=world, transport=transport, feature_layout=layout,
                attention_layout=attention_layout, context=context, model_config=config, operator_config=core_config,
                bootstrap_operator=bootstrap, keys=life, prefetch_keys=bool(prefetch_keys),
                function_prefix=f"{prefix}.pairfm", tag=f"layer/{iteration}/pairfm"), barrier=True)
            keep["pf"] = pf
            record_exit("pairfm", pf.ciphers, label="w_o_carrier")
            stations["pairfm"].update({
                "stage_seconds": _stage_dict(pf.stage_seconds),
                "source_stage_seconds": _stage_dict(pf.source_stage_seconds),
                "source_slot_seconds": [round(float(v), 4) for v in pf.source_slot_seconds],
                "comm_gib": {k: round(int(v) / float(2**30), 4) for k, v in pf.comm_bytes.items()},
                "input_limbs": list(pf.input_limbs),
                "partial_limbs": int(pf.partial_limbs),
                "carrier_indices": [int(v) for v in pf.carrier_indices],
                "leases": _lease_rows(pf.lease_records),
            })
            if oracle is not None:
                stations["pairfm"]["oracle"] = pairfm_oracle(
                    oracle, layout, attention_layout, carriers=pf.carriers, carrier_indices=pf.carrier_indices,
                    context=context)
            core.release()
            if done("w_o"):
                raise StopIteration

            # -- 6. W_O: own carriers -> own FM shards (linear tier, no communication)
            w_o_profile: dict[str, float] = {}
            wo = timed("w_o", "linear", lambda: w_o_station(
                pf.carriers, o_w, ownership=ownership, layout=layout, attention_layout=attention_layout,
                context=context, operator_config=w_o_config, profile=w_o_profile))
            keep["wo"] = tuple(wo)
            record_exit("w_o", wo)
            if oracle is not None:
                stations["w_o"]["oracle"] = _compare(_decrypt_rows(context, layout, ownership, wo), oracle["w_o"][rows])
            stations["w_o"]["stage_seconds"] = _stage_dict({k: v for k, v in w_o_profile.items() if "." not in k})
            pf.release()
            if done("residual_attn"):
                raise StopIteration

            # -- 7. residual: refreshed layer input + W_O shards
            res1 = timed("residual_attn", None, lambda: residual_station(
                rms.residual_inputs, wo, ownership=ownership, layout=layout, context=context))
            keep["res1"] = res1
            record_exit("residual_attn", res1.ciphers)
            if oracle is not None:
                stations["residual_attn"]["oracle"] = _compare(
                    _decrypt_rows(context, layout, ownership, res1.ciphers), oracle["residual_attn"][rows])
            stations["residual_attn"]["modulus_aligned_pairs"] = int(res1.modulus_aligned_pairs)
            stations["residual_attn"]["input_limbs"] = {
                "residual": [int(c.state.cur_limbs) for c in rms.residual_inputs],
                "w_o": [int(c.state.cur_limbs) for c in wo],
            }
            rms.release()
            _release_ciphers(keep.pop("wo"))
            if done("rmsnorm_post"):
                raise StopIteration

            # -- 8. post-attention RMSNorm (T0 keys); keeps the refreshed residual branch
            rms2 = timed("rmsnorm_post", None, lambda: rmsnorm_station(
                res1.ciphers, gamma_post, coeffs=coeffs_ffn, ownership=ownership, layout=layout, context=context,
                bootstrap_operator=bootstrap, keep_residual=True, **rms_exit_post,
                function_prefix=f"{prefix}.rms_post"))
            keep["rms2"] = rms2
            record_exit("rmsnorm_post", rms2.ciphers)
            if oracle is not None:
                deferred = np.asarray(rms2.deferred_output_scale, dtype=np.float64).reshape(1, -1)
                stations["rmsnorm_post"]["oracle"] = _compare(
                    _decrypt_rows(context, layout, ownership, rms2.ciphers),
                    oracle["rmsnorm_post"][rows] * (gamma_post.astype(np.float64).reshape(1, -1) / deferred),
                    scale=float(rms2.value_scale))
                stations["rmsnorm_post"]["oracle_residual"] = _compare(
                    _decrypt_rows(context, layout, ownership, rms2.residual_inputs), oracle["residual_attn"][rows])
            stations["rmsnorm_post"]["residual_limbs"] = [int(c.state.cur_limbs) for c in rms2.residual_inputs]
            stations["rmsnorm_post"]["stage_seconds"] = _stage_dict(rms2.stage_seconds)
            stations["rmsnorm_post"]["inner_stage_seconds"] = _stage_dict(rms2.level_trace.get("rmsnorm_stage_seconds"))
            res1.release()
            if done("mlp"):
                raise StopIteration

            # -- 9. MLP token shard (gate/up -> SiLU -> down) under the linear tier
            mlp = timed("mlp", "linear", lambda: mlp_station(
                rms2.ciphers, hidden, gate_w, up_w, down_w, silu_entry=silu_entry, ownership=ownership, layout=layout,
                context=context, config=config, operator_config=mlp_config,
                input_column_scale=rms2.linear_input_column_scale(),
                fuse_silu_domain_map=fuse_silu, input_limbs=mlp_entry,
                feature_shard_k=int(preset.mlp_feature_shard_k), transport=transport,
                function_prefix=f"{prefix}.mlp"))
            keep["mlp"] = mlp
            record_exit("mlp", mlp.ciphers)
            if oracle is not None:
                stations["mlp"]["oracle"] = _compare(_decrypt_rows(context, layout, ownership, mlp.ciphers), oracle["mlp"][rows])
            stations["mlp"]["input_limbs"] = {"before_drop": list(rms2.output_limbs), "entry": mlp_entry}
            stations["mlp"]["stage_seconds"] = _stage_dict(mlp.stage_seconds)
            stations["mlp"]["feature_shard_k"] = int(preset.mlp_feature_shard_k)
            if getattr(mlp, "comm_bytes", None):
                stations["mlp"]["comm_bytes"] = {k: int(v) for k, v in mlp.comm_bytes.items()}
                stations["mlp"]["shard_group"] = list(mlp.group)
                stations["mlp"]["shard_pages"] = list(mlp.pages)
            _release_ciphers(rms2.ciphers)
            rms2.ciphers = ()
            if done("residual_mlp"):
                raise StopIteration

            # -- 10. residual: refreshed post-norm input + MLP shards -> next layer's FM pair
            res2 = timed("residual_mlp", None, lambda: residual_station(
                rms2.residual_inputs, mlp.ciphers, ownership=ownership, layout=layout, context=context))
            keep["res2"] = res2
            record_exit("residual_mlp", res2.ciphers, label="layer_output")
            if oracle is not None:
                stations["residual_mlp"]["oracle"] = _compare(
                    _decrypt_rows(context, layout, ownership, res2.ciphers), oracle["layer_output"][rows])
            stations["residual_mlp"]["modulus_aligned_pairs"] = int(res2.modulus_aligned_pairs)
            stations["residual_mlp"]["input_limbs"] = {
                "residual": [int(c.state.cur_limbs) for c in rms2.residual_inputs],
                "mlp": [int(c.state.cur_limbs) for c in mlp.ciphers],
            }
            rms2.release()
            mlp.release()
        except StopIteration:
            pass
        finally:
            for value in keep.values():
                release = getattr(value, "release", None)
                if callable(release):
                    release()
                elif isinstance(value, (tuple, list)):
                    _release_ciphers(value)
            if inputs:
                release_all(inputs)
        easyfhe.cuda.synchronize()
        layer_wall = time.perf_counter() - layer_started
        return stations, magnitudes, layer_wall

    iterations = []
    for iteration in range(max(1, int(repeat))):
        life.prefetch_log = []
        stations, magnitudes, layer_wall = run_layer(iteration)
        prefetch_log = list(life.prefetch_log)
        iterations.append({
            "iteration": int(iteration),
            "layer_wall_s": layer_wall,
            "station_wall_sum_s": sum(v["wall_s"] for v in stations.values()),
            "bootstrap_calls": sum(v["bootstrap_calls"] for v in stations.values()),
            "bootstrap_seconds": round(sum(v["bootstrap_seconds"] for v in stations.values()), 4),
            "stations": stations,
            "magnitudes": magnitudes if verify else None,
            "prefetch": {
                "mode": life.prefetch_mode,
                "profile_keys": {tier: len(seen) for tier, seen in life.profile.items()},
                "issued": [e for e in prefetch_log if e.get("issued") is True],
                "skipped": [e for e in prefetch_log if e.get("issued") is False],
                "awaited": [e for e in prefetch_log if e.get("awaited")],
                "async_wait_s": round(sum(float(e.get("wait_s", 0.0)) for e in prefetch_log if e.get("awaited")), 4),
                "async_issued_gib": round(sum(float(e.get("gib", 0.0)) for e in prefetch_log if e.get("issued") is True), 4),
            },
        })
    probe.snap("90 end", {"resident_keys_gib": native_cache_inventory(native)["total_gib"]})

    summary = {
        "rank": int(rank),
        "emulated": bool(emulated),
        "device_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "ownership": ownership.to_dict(),
        "tiers": list(tiers),
        "active_stations": list(active),
        "barrier_each": bool(barrier_each),
        "setup_seconds": state.setup_seconds,
        "resident": {**resident, "pinned_keys": state.resident_report()["pinned_keys"]},
        "key_plan": key_plan,
        "rmsnorm_knobs": {"coeffs": rms_table_info, "exit": rms_exit, "exit_policy": rmsnorm_exit_description(preset)},
        "softmax_knobs": {"coeffs": softmax_table_info,
                          "degrees": softmax_table_degrees(softmax_table).get(str(int(coeff_layer)))},
        "iterations": iterations,
        # last iteration = steady state (resident tiers loaded, leases warm on the pinned path)
        "stations": iterations[-1]["stations"],
        "magnitudes": iterations[-1]["magnitudes"],
        "bootstrap_calls_total": bootstrap.calls,
        "real_weights": None if fixture is None else str(real_weights),
        "oracle": None if oracle is None else {
            "seconds": round(oracle_seconds, 2),
            "self_check_vs_fixture_hidden_out": oracle_self_check,
            "scores_max_abs": oracle["scores_max_abs"], "gate_max_abs": oracle["gate_max_abs"],
            "hidden_in_max_abs": float(np.max(np.abs(hidden))), "hidden_in_rms": float(np.sqrt(np.mean(hidden ** 2))),
            "layer_output_error": iterations[-1]["stations"].get("residual_mlp", {}).get("oracle"),
        },
        "transport": {
            **transport.stats.to_dict(),
            "backend": type(transport).__name__,
            # pinned backend: staged/restored/overflow/deduped counters of the arena
            **({"arena": dict(transport.arena.stats)} if hasattr(transport, "arena") else {}),
        },
        "timeline": probe.timeline,
    }
    gathered = transport.gather_to(0, "layer/summary", summary)
    transport.barrier("layer/final")
    transport.close()
    if not emulated:
        state.release()
    return gathered if rank == 0 else None


def _release_ciphers(ciphers) -> None:
    from llama3fhe.backend import release_all

    ciphers = tuple(c for c in ciphers if c is not None)
    if ciphers:
        release_all(ciphers)


# ------------------------------------------------------------------- host
def _attribution(entry: dict) -> dict:
    """Split one station's wall into BS / key H2D / evict / transport wait / the rest (compute)."""

    wall = float(entry["wall_s"])
    bs_s = float(entry.get("bootstrap_seconds") or 0.0)
    keys_s = float(entry["keys"].get("load_seconds") or 0.0)
    evict_s = sum(float(r.get("evict_s") or 0.0) for r in entry.get("leases") or [])
    wait_s = float((entry.get("transport_delta") or {}).get("recv_wait_seconds") or 0.0)
    return {"wall_s": wall, "bs_s": bs_s, "keys_s": keys_s, "evict_s": evict_s, "wait_s": wait_s,
            "compute_s": max(0.0, wall - bs_s - keys_s - evict_s - wait_s)}


def _oracle_entries(oracle: dict):
    """A station's oracle is either one error dict or a dict of named error dicts (attn_input: query/key/value)."""

    if "rel_max_err" in oracle:
        return [(None, oracle)]
    return [(str(k), v) for k, v in oracle.items() if isinstance(v, dict) and "rel_max_err" in v]


def _table_row(summary: dict) -> dict:
    stations = summary["stations"]
    row = {"rank": summary["rank"], "resident_keys_gib": round(summary["resident"]["resident_keys_gib"], 2),
           "bs_total": summary["bootstrap_calls_total"],
           "iter_layer_wall_s": [round(it["layer_wall_s"], 3) for it in summary.get("iterations", [])]}
    for name in STATIONS:
        entry = stations.get(name)
        if entry is None:
            continue
        split = _attribution(entry)
        cell = {"wall_s": round(entry["wall_s"], 3), "bs": entry["bootstrap_calls"],
                "bs_s": round(split["bs_s"], 3), "compute_s": round(split["compute_s"], 3),
                "peak_gib": round(entry.get("peak_alloc_gib") or 0.0, 2)}
        if entry["keys"].get("gib_loaded"):
            cell["keys_gib"] = round(entry["keys"]["gib_loaded"], 2)
            cell["keys_s"] = round(split["keys_s"], 3)
        if split["evict_s"]:
            cell["evict_s"] = round(split["evict_s"], 3)
        if split["wait_s"]:
            cell["wait_s"] = round(split["wait_s"], 3)
        if entry["keys"].get("misses"):
            cell["misses"] = entry["keys"]["misses"]
        limbs = entry.get("output_limbs")
        if limbs is not None:
            cell["out_limbs"] = limbs
        if entry.get("oracle"):
            for label, err in _oracle_entries(entry["oracle"]):
                suffix = "" if label is None else f".{label}"
                cell[f"rel_err{suffix}"] = round(err["rel_max_err"], 6)
                cell[f"max_err{suffix}"] = round(err["max_abs_err"], 6)
        row[name] = cell
    row["layer_wall_s"] = round(sum(stations[n]["wall_s"] for n in STATIONS if n in stations), 3)
    if summary.get("oracle") and summary["oracle"].get("layer_output_error"):
        err = summary["oracle"]["layer_output_error"]
        row["layer_output"] = {k: round(err[k], 6) for k in ("max_abs_err", "mean_abs_err", "rms_err", "rel_max_err")}
        row["layer_output"]["oracle_max_abs"] = round(err["oracle_max_abs"], 4)
        row["layer_output"]["finite"] = err["finite"]
    return row


def _median(values):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])


def _attribution_table(per_rank) -> dict:
    """Cross-rank median of every station's wall split plus the layer totals (steady state)."""

    out: dict[str, dict] = {}
    totals = {"wall_s": 0.0, "bs_s": 0.0, "keys_s": 0.0, "evict_s": 0.0, "wait_s": 0.0, "compute_s": 0.0}
    for name in STATIONS:
        splits = [_attribution(s["stations"][name]) for s in per_rank if name in s["stations"]]
        if not splits:
            continue
        cell = {key: round(_median(sp[key] for sp in splits), 3) for key in totals}
        cell["wall_max_s"] = round(max(sp["wall_s"] for sp in splits), 3)
        cell["bs"] = per_rank[0]["stations"][name]["bootstrap_calls"]
        out[name] = cell
        for key in totals:
            totals[key] += cell[key]
    out["layer"] = {key: round(value, 3) for key, value in totals.items()}
    wall = totals["wall_s"] or 1.0
    out["layer"]["share"] = {key: round(totals[key] / wall, 3) for key in ("bs_s", "keys_s", "evict_s", "wait_s", "compute_s")}
    return out


def _emulated_table(per_rank, *, comm_gbps: float | None = None) -> dict:
    """Per station: each rank's share alone on the card (``wall - recv_wait``), max / median / sum over ranks.

    ``layer_emulated_s`` = sum over stations of the slowest rank's share (lock-step,
    communication free); ``layer_comm_model_s`` adds ``bytes_sent`` (max over
    ranks) / ``comm_gbps`` per station when a bandwidth is given;
    ``layer_all_ranks_s`` = sum over ranks and stations (the T2 code path run
    serially on this card, with its redundant per-rank encodes).
    """

    out: dict[str, dict] = {}
    layer_max = layer_sum = layer_comm = 0.0
    for name in STATIONS:
        rows = [s["stations"][name] for s in per_rank if name in s["stations"]]
        ranks = [int(s["rank"]) for s in per_rank if name in s["stations"]]
        if not rows:
            continue
        shares = [max(0.0, float(r["wall_s"]) - float((r.get("transport_delta") or {}).get("recv_wait_seconds") or 0.0))
                  for r in rows]
        bs = [float(r.get("bootstrap_seconds") or 0.0) for r in rows]
        bytes_sent = [int((r.get("transport_delta") or {}).get("bytes_sent") or 0) for r in rows]
        comm = (max(bytes_sent) / (float(comm_gbps) * 1e9)) if comm_gbps else 0.0
        cell = {
            "compute_max_s": round(max(shares), 3), "compute_median_s": round(_median(shares), 3),
            "compute_sum_s": round(sum(shares), 3),
            "compute_argmax_rank": int(ranks[shares.index(max(shares))]),
            "bs_max_s": round(max(bs), 3), "bs_calls": int(rows[0].get("bootstrap_calls") or 0),
            "bytes_sent_max": max(bytes_sent), "comm_model_s": round(comm, 4),
            "peak_alloc_gib_max": round(max(float(r.get("peak_alloc_gib") or 0.0) for r in rows), 2),
        }
        out[name] = cell
        layer_max += max(shares)
        layer_sum += sum(shares)
        layer_comm += comm
    out["layer"] = {
        "layer_emulated_s": round(layer_max, 3),
        "layer_comm_model_s": round(layer_comm, 3),
        "layer_emulated_with_comm_s": round(layer_max + layer_comm, 3),
        "layer_all_ranks_s": round(layer_sum, 3),
        "comm_gbps": comm_gbps,
    }
    return out


def _consistency(per_rank) -> dict:
    """Every rank must have walked the same stations with the same BS count and exit limbs."""

    reference = per_rank[0]["stations"]
    mismatches = {}
    for summary in per_rank[1:]:
        for name, entry in summary["stations"].items():
            base = reference.get(name)
            if base is None:
                mismatches[f"{name}@r{summary['rank']}"] = "missing on rank 0"
                continue
            for key in ("bootstrap_calls", "output_limbs"):
                if entry.get(key) != base.get(key):
                    mismatches[f"{name}/{key}@r{summary['rank']}"] = [base.get(key), entry.get(key)]
    return {"ok": not mismatches, "mismatches": mismatches}


def _run_emulated(preset, kwargs: dict, *, stop_after: str | None):
    """Build one shared context (all-rank key union resident) and run the ranks as lock-stepped threads."""

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("LLAMA_FHE_PLAIN_CACHE_GB", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("LLAMA_FHE_TRANSFORM_HOIST", str(preset.transform_hoist_strategy))
    os.environ["T2_WORLD_SIZE"] = str(int(preset.world_size))
    if str(preset.key_policy) != "resident":
        raise SystemExit("--emulate-single-device shares one key cache between the rank threads: "
                         "use a preset with key_policy='resident' (a100-high / h100-emu)")

    import easyfhe

    from llama3fhe.parallel.emulation import all_rank_key_union, run_emulated_world
    from llama3fhe.parallel.memory import native_cache_inventory
    from llama3fhe.parallel.worker import build_worker_state, production_layouts

    stop_index = len(STATIONS) if stop_after is None else STATIONS.index(str(stop_after)) + 1
    tiers = worker_tiers(STATIONS[:stop_index])
    feature_layout, attention_layout = production_layouts()
    started = time.perf_counter()
    union = all_rank_key_union(preset, feature_layout, attention_layout, tiers=tiers)
    shared = build_worker_state(0, preset, tiers=tiers, trace_keys=True, extra_rotations=union)
    easyfhe.cuda.synchronize()
    setup = {
        "seconds": round(time.perf_counter() - started, 2),
        "worker_setup_seconds": dict(shared.setup_seconds),
        "key_union": {k: len(v) for k, v in union.items()},
        "key_plan": shared.key_plan.summary(alpha=int(shared.native.alpha), special_limbs=int(shared.native.K)),
        "resident_keys_gib": native_cache_inventory(shared.native)["total_gib"],
        "resident": shared.resident_report(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    print(json.dumps({"emulation_setup": {k: v for k, v in setup.items() if k != "resident"}}, default=str), flush=True)
    try:
        outcomes = run_emulated_world(preset, worker_entry, shared_state=shared, kwargs=kwargs)
    finally:
        shared.release()
    setup["baton_wait_seconds"] = [round(o.baton_wait_seconds, 3) for o in outcomes]
    setup["thread_wall_seconds"] = [round(o.wall_seconds, 2) for o in outcomes]
    return outcomes, setup


def _check_world(preset) -> None:
    """The station chain owns whole T2 slots (two FM ciphertexts each) per rank.

    ``world_size x ciphers_per_rank`` must cover the 16-ciphertext production
    layout with an even, positive ``ciphers_per_rank`` (8 ranks x 2, 4 x 4,
    2 x 8); the PairFM algebra needs the 8 source slots to coincide with the
    8 W_O carriers, which :func:`~llama3fhe.parallel.prefill.ownership.ownership_from_layouts`
    checks again per rank.
    """

    from llama3fhe.parallel.worker import production_layouts

    feature_layout, _ = production_layouts()
    cipher_count = int(feature_layout.cipher_count)
    per_rank = int(preset.ciphers_per_rank)
    if int(preset.cipher_count) != cipher_count or per_rank <= 0 or per_rank % 2:
        raise SystemExit(
            f"world_size {preset.world_size} x ciphers_per_rank {per_rank} does not cover the "
            f"{cipher_count}-ciphertext layout in whole T2 slots (2 ciphertexts): use --world-size "
            f"{cipher_count // 2} (one slot per rank) or {cipher_count // 4} (two slots per rank), or the "
            f"default world with --ranks-per-device on a host with fewer cards"
        )


def main() -> int:
    args = parse_args()
    preset = preset_by_name(args.preset)
    from dataclasses import replace

    if args.prefetch_mode is not None:
        preset = replace(preset, key_prefetch_mode=str(args.prefetch_mode))
    if args.prefetch_margin_gib is not None:
        preset = replace(preset, key_prefetch_margin_gib=float(args.prefetch_margin_gib))
    if args.evict_sync is not None:
        preset = replace(preset, key_evict_sync=str(args.evict_sync))
    if args.mlp_shard_k is not None:
        preset = replace(preset, mlp_feature_shard_k=int(args.mlp_shard_k))
    if args.transform_hoist is not None:
        preset = replace(preset, transform_hoist_strategy=str(args.transform_hoist))
        os.environ["LLAMA_FHE_TRANSFORM_HOIST"] = str(args.transform_hoist)
    preset = apply_runtime_knobs(args, preset)
    if args.world_size is not None:
        try:
            preset = preset.with_world_size(
                int(args.world_size),
                mlp_feature_shard_k=None if args.mlp_shard_k is None else int(args.mlp_shard_k),
            )
        except ValueError as exc:
            raise SystemExit(f"--world-size {args.world_size}: {exc}") from exc
    _check_world(preset)
    if args.transport is not None:
        os.environ["T2_TRANSPORT"] = str(args.transport)
    if not args.emulate_single_device:
        # Fail on a bad rank -> card map before any worker touches a GPU.
        expand_devices(preset.world_size, args.devices, ranks_per_device=int(args.ranks_per_device))
        if int(args.ranks_per_device) > 1:
            preset = preset.for_colocation(int(args.ranks_per_device))
    if args.max_plaintext_rows is not None:
        # After with_world_size / for_colocation: an explicit value also overrides their multi-slot 512 fallback.
        preset = replace(preset, linear_max_plaintext_rows=int(args.max_plaintext_rows))
    if preset.key_prefetch_mode == "async" and args.prefetch_keys:
        raise SystemExit("--prefetch-keys (synchronous tier upload) and --prefetch-mode async are exclusive")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "artifacts" / f"t2_layer_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "input_level": int(args.input_level),
        "verify": bool(args.verify),
        "prefetch_keys": bool(args.prefetch_keys),
        "qkv_limbs": args.qkv_limbs,
        "mlp_limbs": args.mlp_limbs,
        "stop_after": args.stop_after,
        "hidden_scale": float(args.hidden_scale),
        "coeff_layer": int(args.coeff_layer),
        "repeat": int(args.repeat),
        "barrier_each": bool(args.barrier_each),
        "real_weights": None if args.real_weights is None else str(Path(args.real_weights).resolve()),
    }
    started = time.perf_counter()
    setup: dict = {}
    if args.emulate_single_device:
        outcomes, setup = _run_emulated(preset, kwargs, stop_after=args.stop_after)
    else:
        outcomes = spawn_workers(preset, worker_entry, kwargs=kwargs, devices=args.devices,
                                 ranks_per_device=int(args.ranks_per_device), transport=args.transport)
    total = time.perf_counter() - started
    per_rank = outcomes[0].result
    assert per_rank is not None and len(per_rank) == preset.world_size

    table = [_table_row(summary) for summary in per_rank]
    consistency = _consistency(per_rank)
    attribution = _attribution_table(per_rank)
    emulated_table = _emulated_table(per_rank, comm_gbps=args.comm_gbps) if args.emulate_single_device else None
    result = {
        "preset": preset.name,
        "emulated_single_device": bool(args.emulate_single_device),
        "emulation": None if emulated_table is None else {**emulated_table, "setup": setup},
        "linear_max_plaintext_rows": int(preset.linear_max_plaintext_rows),
        "key_prefetch_mode": str(preset.key_prefetch_mode),
        "key_prefetch_margin_gib": float(preset.key_prefetch_margin_gib),
        "key_evict_sync": str(preset.key_evict_sync),
        "pin_leased_keys": bool(preset.pin_leased_keys),
        "mlp_feature_shard_k": int(preset.mlp_feature_shard_k),
        "transform_hoist_strategy": os.environ.get("LLAMA_FHE_TRANSFORM_HOIST", str(preset.transform_hoist_strategy)),
        **runtime_knob_summary(preset),
        "timestamp": stamp,
        "total_wall_s": round(total, 2),
        "worker_wall_s": [round(o.wall_seconds, 2) for o in outcomes],
        "args": vars(args),
        "consistency": consistency,
        "attribution": attribution,
        "table": table,
        "ranks": per_rank,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"preset": preset.name, "total_wall_s": result["total_wall_s"],
                      "stations": per_rank[0]["active_stations"], "consistency_ok": consistency["ok"],
                      "output": str(out_dir)}))
    for row in table:
        print(json.dumps(row))
    print(json.dumps({"attribution_median": attribution}))
    if emulated_table is not None:
        print(json.dumps({"emulated": emulated_table}))
    if per_rank[0].get("oracle"):
        worst = {}
        for name in STATIONS:
            by_label: dict[str, list] = {}
            for s in per_rank:
                oracle = s["stations"].get(name, {}).get("oracle")
                if oracle:
                    for label, err in _oracle_entries(oracle):
                        by_label.setdefault(name if label is None else f"{name}.{label}", []).append(err)
            for label, errs in by_label.items():
                worst[label] = {"rel_max_err": round(max(e["rel_max_err"] for e in errs), 6),
                                "max_abs_err": round(max(e["max_abs_err"] for e in errs), 6),
                                "oracle_max_abs": round(max(e["oracle_max_abs"] for e in errs), 4),
                                "finite": all(e["finite"] for e in errs)}
        print(json.dumps({"oracle_worst_over_ranks": worst,
                          "oracle_self_check": per_rank[0]["oracle"]["self_check_vs_fixture_hidden_out"]["max_abs_err"]}))
    if not consistency["ok"]:
        print(json.dumps({"mismatches": consistency["mismatches"]}))
    return 0 if consistency["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
