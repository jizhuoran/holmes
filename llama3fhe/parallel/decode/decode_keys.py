"""Rotation-key plan of the multi-card *decode* step (T2-Decode).

The decode graph rotates by a different set than the prefill station chain
(token lane 1 instead of 8; PairFold KV routing / QK / PV instead of the Delta
attention), so :func:`~.keys.build_key_plan` -- which asks the *prefill*
planners -- is the wrong starting point.  :func:`build_decode_key_plan` builds
the :class:`~.keys.KeyPlan` straight from the decode planners
(:func:`llama3fhe.graph.decode_layer_rotations` split by consumer):

===========  =====  ==================================================  =========================
tier         level  rotations                                           limb cap
===========  =====  ==================================================  =========================
bootstrap    0      Meta-BTS (preset ``bootstrap``)                     none (full chain)
rmsnorm      0      DecodeFM feature reduction (same set as prefill)    none
linear       1      PairFold QKV + folded W_O + MLP BSGS diagonals      ``limb_caps["linear"]``
attn_core    2      PairFold KV/Q routing, QK, Alg2 Softmax, PV         ``attention_limb_cap``
===========  =====  ==================================================  =========================

An optional ``attn_pv`` tier raises only the probability-to-PV route keys for
single-card schedules whose post-Softmax probability has more than 13 limbs.
"""

from __future__ import annotations

from typing import Mapping

from ..prefill.keys import KeyPlan, KeyTier, normalize_rotation

DECODE_TIERS = ("bootstrap", "rmsnorm", "linear", "attn_core")


def decode_rotation_families(
    schedule,
    *,
    feature_layout,
    pairfold_layout=None,
    log_n: int = 16,
) -> dict[str, tuple[int, ...]]:
    """``{family: normalised rotations}`` of the pairfold128 decode layer.

    Families: ``rmsnorm``, ``qkv``, ``w_o``, ``mlp``, ``route`` (KV / Q
    extraction from the carrier), ``attention`` (QK, Softmax, PV, folded-O).
    """

    from llama3fhe.graph import _decode_linear_config, _linear_config
    from llama3fhe.layouts.pairfold128 import PairFold128Layout
    from llama3fhe.operators.attention.pairfold.pairfold_decode import (
        pairfold_decode_rotations,
    )
    from llama3fhe.operators.attention.pairfold.pairfold_route import (
        pairfold_route_rotations,
    )
    from llama3fhe.operators.linear.pairfold.pairfold_output import (
        pairfold_w_o_rotations,
    )
    from llama3fhe.operators.linear.pairfold.pairfold_qkv import pairfold_qkv_rotations
    from llama3fhe.operators.linear.prefill.mlp import mlp_rotations
    from llama3fhe.operators.norm.rmsnorm import rmsnorm_rotations

    pairfold_layout = PairFold128Layout() if pairfold_layout is None else pairfold_layout
    ring_dim = 1 << int(log_n)

    def norm(values) -> tuple[int, ...]:
        return tuple(sorted({normalize_rotation(int(v), ring_dim=ring_dim) for v in values if int(v) != 0}))

    return {
        "rmsnorm": norm(rmsnorm_rotations(feature_layout)),
        "qkv": norm(
            pairfold_qkv_rotations(
                pairfold_layout,
                operator_config=_decode_linear_config(
                    schedule, schedule.qkv_max_plaintext_rows, schedule.qkv_hoist_strategy
                ),
            )
        ),
        "w_o": norm(
            pairfold_w_o_rotations(
                pairfold_layout,
                operator_config=_linear_config(
                    schedule, schedule.output_max_plaintext_rows, schedule.output_hoist_strategy
                ),
            )
        ),
        "mlp": norm(
            mlp_rotations(
                feature_layout,
                operator_config=_decode_linear_config(
                    schedule, schedule.mlp_max_plaintext_rows, schedule.mlp_hoist_strategy
                ),
            )
        ),
        "route": norm(
            pairfold_route_rotations(
                pairfold_layout, token_columns=None, max_baby_offsets=int(schedule.sparse_max_baby_offsets)
            )
        ),
        "attention": norm(pairfold_decode_rotations(pairfold_layout)),
    }


def build_decode_key_plan(
    preset,
    *,
    schedule,
    feature_layout,
    attention_layout=None,
    pairfold_layout=None,
    limb_caps: Mapping[str, int] | None = None,
    attention_limb_cap: int | None = None,
    probability_route_limb_cap: int | None = None,
    policy: str | None = None,
    resident_extra=None,
    linear_limb_floor: int | None = None,
) -> KeyPlan:
    """The decode :class:`~.keys.KeyPlan` of ``preset`` (see the module docstring).

    ``limb_caps`` defaults to the preset's caps (only ``linear`` is used);
    ``linear_limb_floor`` raises the ``linear`` cap to at least that many limbs
    (:func:`~.decode_stations.decode_linear_limb_floor` derives the requirement
    across the selected layers); ``attention_limb_cap`` caps the ``attn_core`` tier (``None`` = full
    chain). ``probability_route_limb_cap`` raises the cap for the PV probability
    route after Softmax; omitting it preserves the preset's route cap.
    ``policy`` / ``resident_extra`` default to the preset's.
    """

    from ..worker import resident_rotations

    caps = dict(preset.limb_caps) if limb_caps is None else {str(k): int(v) for k, v in dict(limb_caps).items()}
    if linear_limb_floor is not None and caps.get("linear") is not None:
        caps["linear"] = max(int(caps["linear"]), int(linear_limb_floor))
    families = decode_rotation_families(
        schedule, feature_layout=feature_layout, pairfold_layout=pairfold_layout, log_n=int(preset.log_n)
    )
    ring_dim = 1 << int(preset.log_n)
    bootstrap = tuple(
        sorted(
            {
                normalize_rotation(int(v), ring_dim=ring_dim)
                for v in resident_rotations(preset, feature_layout, attention_layout, families=("bootstrap",)).bootstrap
                if int(v) != 0
            }
        )
    )
    linear = tuple(sorted(set(families["qkv"]) | set(families["w_o"]) | set(families["mlp"])))
    attention = tuple(sorted(set(families["route"]) | set(families["attention"])))
    tiers = [
        KeyTier("bootstrap", 0, bootstrap, None),
        KeyTier("rmsnorm", 0, families["rmsnorm"], None),
        KeyTier("linear", 1, linear, caps.get("linear")),
        KeyTier("attn_core", 2, attention, None if attention_limb_cap is None else int(attention_limb_cap)),
    ]
    if probability_route_limb_cap is not None:
        from llama3fhe.layouts.pairfold128 import PairFold128Layout
        from llama3fhe.operators.attention.pairfold.pairfold_decode import (
            pv_probability_route_rotations,
        )

        pairfold_layout = PairFold128Layout() if pairfold_layout is None else pairfold_layout
        tiers.append(
            KeyTier(
                "attn_pv",
                2,
                tuple(
                    normalize_rotation(int(v), ring_dim=ring_dim)
                    for v in pv_probability_route_rotations(pairfold_layout)
                ),
                int(probability_route_limb_cap),
            )
        )
    return KeyPlan(
        tiers=tuple(tiers),
        policy=str(preset.key_policy if policy is None else policy),
        ring_dim=ring_dim,
        full_limbs=int(preset.depth) + 1,
        resident_extra=tuple(preset.key_resident_extra) if resident_extra is None else tuple(resident_extra),
    )


__all__ = ["DECODE_TIERS", "build_decode_key_plan", "decode_rotation_families"]
