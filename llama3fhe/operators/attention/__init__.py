"""Attention subsystem: QKV carriers in, W_O-ready carriers out.

The package owns RoPE/layout preparation, Delta QK, Delta-native Softmax,
Delta PV, and the PairFM output adapter. QKV and W_O remain linear operators
outside this boundary.
"""

from ...approx.attention import (
    DeltaAttentionResult,
    apply_rope_heads,
    attention_gqa_numpy,
    pv_delta_numpy,
    qk_delta_numpy,
    rope_cos_sin,
    run_delta_attention_numpy,
    softmax_delta_numpy,
)
from .config import AttentionOperatorConfig
from .decode.decode import (
    DecodeAttentionFHEResult,
    apply_decode_softmax_public_map_fhe,
    correct_prefill_emitted_key_c_fhe,
    correct_prefill_emitted_key_c_numpy,
    decode_attention_alg2_numpy,
    decode_attention_fhe,
    decode_attention_numpy,
    decode_attention_rotations,
    decode_broadcast_token0_numpy,
    decode_broadcast_tokens_fhe,
    decode_output_c_rotations,
    decode_reduce_tokens_fhe,
    folded_key_c_to_real_c_fhe,
    output_c_to_decode_fm_fhe,
    output_c_to_decode_fm_numpy,
    prefill_kv_cache_numpy,
    pv_c_fhe,
    pv_c_numpy,
    qk_c_fhe,
    qk_c_numpy,
    softmax_alg2_row_numpy,
    softmax_row_numpy,
)
from .decode.decode_softmax import (
    DecodeAlg2Poly,
    apply_decode_alg2_public_map_fhe,
    apply_decode_alg2_public_map_numpy,
    softmax_alg2_poly_numpy,
    softmax_alg2_row_fhe,
)
from .prefill.input import (
    ComplexTokenAttentionComplexity,
    ComplexTokenAttentionResult,
    complex_token_attention_complexity,
    complex_token_attention_rotations,
    prepare_attention_inputs_fhe,
)
from .pairfold.pairfold128 import (
    folded_w_o_pairfold128_numpy,
    pv_pairfold128_numpy,
    qk_pairfold128_numpy,
)
from .pairfold.pairfold_decode import (
    PairFoldDecodeComplexity,
    add_pairfold_output_contributions_fhe,
    pairfold_decode_complexity,
    pairfold_decode_rotations,
    pv_pairfold_pages_fhe,
    pv_pairfold_prefix_fhe,
    pv_pairfold_rotations,
    qk_pairfold_pages_fhe,
    qk_pairfold_prefix_fhe,
    qk_pairfold_rotations,
    validate_pairfold_score_compatibility,
)

__all__ = [name for name in globals() if not name.startswith("_")]
