"""Two plaintext Prefix keys in one Delta Softmax main stream.

The three Prefix scores that do not fit beside causal Content scores stay in
otherwise unused donor slots through Alg2.  Only the row-sum and lambda side
branches route those slots to their true query positions.  QK still receives
plaintext Prefix K; this module only handles encrypted score ciphertexts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from llama3fhe.backend import profile_sync_device as sync_device
from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout

from ....approx.chebyshev import chebyshev_ps_mul_depth
from ..config import AttentionOperatorConfig
from ..decode.decode import _mul_packed_rows
from ..decode.decode_softmax import _align_cipher_pair
from ..prefill.softmax import (
    SLIM_MID_ALG2_LAYERS,
    SLIM_POST_ALG2_LAYERS,
    SLIM_POST_ALG2_MIN_ROUNDS,
    SLIM_POST_EXP_LAYERS,
    delta_causal_mask,
    extra_softmax_checkpoint_enabled,
)


def fused_prefix_rotations(layout: AttentionPairLayout) -> tuple[int, ...]:
    """Additional rotations for the two-key donor geometry."""

    seq = int(layout.seq_len)
    if seq < 5:
        raise ValueError("fused Prefix requires at least five Delta rows.")
    return tuple(sorted({-4 * seq - 4, -2 * seq - 4, -2 * seq,
                         2 * seq, 2 * seq + 4, 4 * seq + 4}))


def _release(ciphers) -> None:
    for cipher in tuple(ciphers or ()):
        release_if_supported(cipher)


def _rotate(ciphers, amount: int, *, crypto_context) -> tuple[object, ...]:
    outputs: list[object] = []
    try:
        for cipher in ciphers:
            outputs.append(
                crypto_context.fhe.homo_rotate(
                    cipher, int(amount), crypto_context.context
                )
            )
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        _release(outputs)


def _add_pairs(left, right, *, crypto_context) -> tuple[object, ...]:
    outputs: list[object] = []
    try:
        for a, b in zip(left, right, strict=True):
            aa, bb, own_a, own_b = _align_cipher_pair(
                a, b, crypto_context=crypto_context
            )
            try:
                outputs.append(
                    crypto_context.fhe.homo_add(
                        aa, bb, crypto_context.context
                    )
                )
            finally:
                if own_a:
                    release_if_supported(aa)
                if own_b:
                    release_if_supported(bb)
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        _release(outputs)


def _masked(ciphers, mask: np.ndarray, *, name: str, crypto_context):
    return _mul_packed_rows(
        tuple(ciphers), mask, name=name, crypto_context=crypto_context
    )


@dataclass(frozen=True)
class FusedPrefixDeltaPlan:
    layout: AttentionPairLayout
    content_mask: np.ndarray
    prefix_source_regular: np.ndarray
    prefix_source_donor: np.ndarray
    regular_mask: np.ndarray
    donor_mask: np.ndarray
    active_mask: np.ndarray

    @classmethod
    def build(
        cls,
        layout: AttentionPairLayout,
        *,
        prefix_k: int,
        token_valid_mask: np.ndarray | None = None,
    ) -> FusedPrefixDeltaPlan:
        if int(prefix_k) != 2:
            raise ValueError("fused Prefix currently supports exactly two keys.")
        seq = int(layout.seq_len)
        if seq < 5:
            raise ValueError("fused Prefix requires at least five Delta rows.")
        query_valid = (
            np.ones(seq, dtype=bool)
            if token_valid_mask is None
            else np.asarray(token_valid_mask, dtype=bool).reshape(-1)
        )
        if query_valid.shape != (seq,):
            raise ValueError(
                f"token_valid_mask must have shape {(seq,)}, got {query_valid.shape}."
            )
        source_regular = np.zeros(int(layout.slots), dtype=np.float64)
        source_donor = np.zeros_like(source_regular)
        overflow = {(0, seq - 1), (1, seq - 2), (1, seq - 1)}
        for prefix_j in range(2):
            for query in range(seq):
                if not query_valid[query]:
                    continue
                source = (
                    source_donor
                    if (prefix_j, query) in overflow
                    else source_regular
                )
                for lane in (0, 1):
                    source[layout.c_slot(prefix_j, query, lane)] = 1.0
        # Every payload copy must carry the same public mask.
        payload = int(layout.payload_slots)
        for mask in (source_regular, source_donor):
            mask[payload:] = np.tile(mask[:payload], int(layout.payload_repetitions) - 1)
        regular = np.roll(source_regular, 2 * seq)
        donor = np.roll(source_donor, 4 * seq + 4)
        content = delta_causal_mask(
            layout, token_valid_mask=token_valid_mask
        ).astype(np.float64)
        if np.any((regular + donor + content) > 1.0):
            raise RuntimeError("fused Prefix score slots collide with Content.")
        active = content + regular + donor
        pairs = int(layout.query_pair_count)
        rows = lambda mask: np.broadcast_to(
            mask, (pairs, int(layout.slots))
        ).copy()
        return cls(
            layout=layout,
            content_mask=rows(content),
            prefix_source_regular=rows(source_regular),
            prefix_source_donor=rows(source_donor),
            regular_mask=rows(regular),
            donor_mask=rows(donor),
            active_mask=active,
        )

    def merge_scores(
        self, prefix_scores, content_scores, *, multiplier: float, crypto_context
    ) -> tuple[object, ...]:
        """Return a publicly scaled Delta stream with Prefix donor cells."""

        pairs = int(self.layout.query_pair_count)
        if len(prefix_scores) != pairs or len(content_scores) != pairs:
            raise ValueError(
                f"fused Prefix needs {pairs} Prefix and Content score ciphers."
            )
        seq = int(self.layout.seq_len)
        content = regular = donor = shifted_regular = shifted_donor = ()
        prefix = merged = ()
        try:
            content = _masked(
                content_scores, self.content_mask * multiplier,
                name="prefix.fused.content_score", crypto_context=crypto_context,
            )
            regular = _masked(
                prefix_scores, self.prefix_source_regular * multiplier,
                name="prefix.fused.regular_score", crypto_context=crypto_context,
            )
            donor = _masked(
                prefix_scores, self.prefix_source_donor * multiplier,
                name="prefix.fused.donor_score", crypto_context=crypto_context,
            )
            shifted_regular = _rotate(regular, -2 * seq, crypto_context=crypto_context)
            shifted_donor = _rotate(donor, -4 * seq - 4, crypto_context=crypto_context)
            prefix = _add_pairs(shifted_regular, shifted_donor, crypto_context=crypto_context)
            merged = _add_pairs(content, prefix, crypto_context=crypto_context)
            result = merged
            merged = ()
            return result
        finally:
            for ciphers in (content, regular, donor, shifted_regular,
                            shifted_donor, prefix, merged):
                _release(ciphers)

    def row_sum_input(self, squared, *, crypto_context) -> tuple[object, ...]:
        """Move donor squares to true query slots on the Z side branch."""

        regular = donor = restored = result = ()
        try:
            regular = _masked(
                squared, 1.0 - self.donor_mask,
                name="prefix.fused.z.regular", crypto_context=crypto_context,
            )
            donor = _masked(
                squared, self.donor_mask,
                name="prefix.fused.z.donor", crypto_context=crypto_context,
            )
            restored = _rotate(
                donor, 2 * int(self.layout.seq_len) + 4,
                crypto_context=crypto_context,
            )
            result = _add_pairs(regular, restored, crypto_context=crypto_context)
            output = result
            result = ()
            return output
        finally:
            for ciphers in (regular, donor, restored, result):
                _release(ciphers)

    def lambda_for_main(self, lambda_pairs, *, crypto_context) -> tuple[object, ...]:
        """Give donor slots the lambda of their logical query rows."""

        regular = shifted = donor = result = ()
        try:
            regular = _masked(
                lambda_pairs, 1.0 - self.donor_mask,
                name="prefix.fused.lambda.regular", crypto_context=crypto_context,
            )
            shifted = _rotate(
                lambda_pairs, -2 * int(self.layout.seq_len) - 4,
                crypto_context=crypto_context,
            )
            donor = _masked(
                shifted, self.donor_mask,
                name="prefix.fused.lambda.donor", crypto_context=crypto_context,
            )
            result = _add_pairs(regular, donor, crypto_context=crypto_context)
            output = result
            result = ()
            return output
        finally:
            for ciphers in (regular, shifted, donor, result):
                _release(ciphers)

    def split_probabilities(self, probabilities, *, crypto_context):
        """Restore PC C-form Prefix rows and isolate CC Delta probabilities."""

        seq = int(self.layout.seq_len)
        content = regular = donor = restored_regular = restored_donor = prefix = ()
        try:
            content = _masked(
                probabilities, self.content_mask,
                name="prefix.fused.output.content", crypto_context=crypto_context,
            )
            regular = _masked(
                probabilities, self.regular_mask,
                name="prefix.fused.output.regular", crypto_context=crypto_context,
            )
            donor = _masked(
                probabilities, self.donor_mask,
                name="prefix.fused.output.donor", crypto_context=crypto_context,
            )
            restored_regular = _rotate(regular, 2 * seq, crypto_context=crypto_context)
            restored_donor = _rotate(donor, 4 * seq + 4, crypto_context=crypto_context)
            prefix = _add_pairs(restored_regular, restored_donor, crypto_context=crypto_context)
            result = (prefix, content)
            prefix = content = ()
            return result
        finally:
            for ciphers in (content, regular, donor, restored_regular,
                            restored_donor, prefix):
                _release(ciphers)


def _minimum_no_refresh_input_limbs(
    *, exp_depth: int, rounds: int, checkpoint_after_round: int | None = None
) -> int:
    """Budget to the first main refresh, or through PV to PairFM.

    Merge and exp domain map cost one level each; exp costs ``exp_depth``;
    the post-exp causal mask and each square/update pair cost 1 and 2 levels.
    A scheduled main refresh needs six input limbs.  Without one, the final
    C/Delta split and PV cost three more levels before PairFM's six-limb
    floor.  Lambda refreshes independently and does not replenish main.
    """

    checkpoint_input_floor = pairfm_input_floor = 6
    score_merge = domain_map = post_exp_mask = output_split = 1
    pv_depth = 2
    if checkpoint_after_round is not None:
        if not 0 <= int(checkpoint_after_round) <= int(rounds):
            raise ValueError("checkpoint_after_round must lie within Alg2.")
        return (
            checkpoint_input_floor + score_merge + domain_map + int(exp_depth)
            + post_exp_mask + 2 * int(checkpoint_after_round)
        )
    return (
        pairfm_input_floor
        + pv_depth
        + output_split
        + 2 * int(rounds)
        + post_exp_mask
        + int(exp_depth)
        + domain_map
        + score_merge
    )


def softmax_alg2_prefix_fused_fhe(
    prefix_score_ciphers,
    content_score_ciphers,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    prefix_k: int,
    layer_idx: int = 0,
    rounds: int | None = None,
    score_scale: float | None = None,
    content_form: str = "delta",
    token_valid_mask: np.ndarray | None = None,
    bootstrap_operator=None,
    refresh_before_exp: bool | None = None,
    refresh_after_exp: bool | None = None,
    operator_config=None,
    stage_seconds: dict[str, float] | None = None,
):
    """Opt-in fused Prefix path; leaves the input score tuples owned by caller."""

    from ..prefill.softmax import softmax_alg2_fhe_delta

    if content_form != "delta":
        raise ValueError("fused Prefix requires content_form='delta'.")
    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    if refresh_before_exp is None:
        refresh_before_exp = operator_config.refresh_softmax_exp_inputs
    if refresh_after_exp is None:
        refresh_after_exp = operator_config.refresh_softmax_exp_outputs
    layer_idx = int(layer_idx)
    k = int(config.k_for(layer_idx) if rounds is None else rounds)
    exp_depth = chebyshev_ps_mul_depth(
        config.layer_exp[layer_idx]["coefficients"]
    )
    checkpoint_after_round = None
    if refresh_after_exp or extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_EXP_BOOTSTRAP", layer_idx,
        SLIM_POST_EXP_LAYERS, bootstrap_operator,
    ):
        checkpoint_after_round = 0
    elif extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_MID_ALG2_BOOTSTRAP", layer_idx,
        SLIM_MID_ALG2_LAYERS, bootstrap_operator,
    ) and (k + 1) // 2 < k:
        checkpoint_after_round = (k + 1) // 2
    elif extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", layer_idx,
        SLIM_POST_ALG2_LAYERS, bootstrap_operator,
        rounds=k, min_rounds=SLIM_POST_ALG2_MIN_ROUNDS,
    ):
        checkpoint_after_round = k
    needed_without_refresh = _minimum_no_refresh_input_limbs(
        exp_depth=exp_depth, rounds=k,
        checkpoint_after_round=checkpoint_after_round,
    )
    input_limbs = min(
        int(cipher.state.cur_limbs)
        for cipher in tuple(prefix_score_ciphers) + tuple(content_score_ciphers)
    )
    upgraded_pre_exp = bool(
        not refresh_before_exp and input_limbs < needed_without_refresh
    )
    if upgraded_pre_exp:
        if bootstrap_operator is None or refresh_after_exp:
            raise ValueError(
                "fused Prefix score stream needs a pre-exp bootstrap: "
                f"input limbs={input_limbs}, estimated required={needed_without_refresh}."
            )
        refresh_before_exp = True
        operator_config = replace(
            operator_config,
            refresh_softmax_exp_inputs=True,
            pair_softmax_pre_exp_bootstrap=True,
        )
        print(
            "[prefix-fused] "
            f"layer={layer_idx} input_limbs={input_limbs} "
            f"required_without_refresh={needed_without_refresh} "
            "upgraded_to_paired_pre_exp=True",
            flush=True,
        )
    plan = FusedPrefixDeltaPlan.build(
        layout, prefix_k=prefix_k, token_valid_mask=token_valid_mask
    )
    score_multiplier = (
        (1.0 / np.sqrt(float(layout.head_dim)))
        if score_scale is None
        else float(score_scale)
    ) / float(2**k)
    if operator_config.pair_softmax_pre_exp_bootstrap:
        fit_lo, fit_hi = (
            float(value) for value in config.layer_exp[layer_idx]["fit_interval"]
        )
        score_multiplier *= 0.5 * (2.0 / (fit_hi - fit_lo))
    timings = {} if stage_seconds is None else stage_seconds
    merged = ()
    result = None
    try:
        started = time.perf_counter()
        merged = plan.merge_scores(
            prefix_score_ciphers, content_score_ciphers,
            multiplier=score_multiplier,
            crypto_context=crypto_context,
        )
        sync_device(crypto_context.device)
        timings["fused.merge"] = time.perf_counter() - started
        result = softmax_alg2_fhe_delta(
            merged,
            layout=layout,
            layer_idx=layer_idx,
            config=config,
            crypto_context=crypto_context,
            operator_config=operator_config,
            score_scale=score_scale,
            bootstrap_operator=bootstrap_operator,
            refresh_before_exp=refresh_before_exp,
            refresh_after_exp=refresh_after_exp,
            token_valid_mask=token_valid_mask,
            rounds=rounds,
            fused_prefix_plan=plan,
        )
        started = time.perf_counter()
        outputs = plan.split_probabilities(
            result.probability_ciphers, crypto_context=crypto_context
        )
        sync_device(crypto_context.device)
        timings["fused.split"] = time.perf_counter() - started
        timings.update(result.stage_seconds)
        return outputs
    finally:
        _release(merged)
        if result is not None:
            result.release()
