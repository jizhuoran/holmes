"""One persistent-feature-major encrypted transformer-layer graph.

The long-lived tensor remains real feature-major across both residual paths::

    FM -> RMSNorm -> FM QKV -> Q_C/K_Delta/V_Delta
       -> Alg2 Softmax/PV -> O_C -> PairFM -> FM W_O -> residual
       -> paired Meta-BTS -> FM RMSNorm -> FM SwiGLU -> residual -> FM

PairFM is private to the attention output epilogue. The complete graph uses a
ciphertext-only Alg2/Meta-BTS schedule and never hides decrypt/re-encrypt
refreshes behind a smaller nominal depth.

Park Slot/Coeff one-layer orchestration is ``graph_park.run_park_layer``.
It does not replace this feature-major graph or production ``env_fhe``.
Slot↔Coeff still uses a labeled decrypt stub at layer width.
Small-scale FmtSwitch/RingSwitch is ``operators.linear.fmt_switch``.


Layout (2026-10 refactor): ``common`` (dataclasses, audits, rotation planning),
``prefill`` (:func:`evaluate`), ``decode`` (:func:`evaluate_decode`).  This package
re-exports the former ``llama3fhe/graph.py`` namespace; monkeypatch the submodule
that *calls* a helper (``llama3fhe.graph.prefill`` / ``.decode``), not this package.
"""

from __future__ import annotations

from .common import (
    LayerApproximations,
    LayerResult,
    LayerSchedule,
    LayerWeights,
    _attention_exclusive_cache_keep_rotations,
    _cipher_levels,
    _cipher_runtime_state,
    _cipher_stream_state_audit,
    _evict_attention_exclusive_cuda_cache,
    _decode_linear_config,
    _linear_config,
    _pairfold_prefill_cache_from_raw_c,
    _residual_state_audit,
    _select_residual_identity,
    operation_audit,
    required_rotations,
)
from .decode import (
    decode_layer_rotations,
    decode_sparse_bootstrap_log_slots,
    evaluate_decode,
    inspect_decode_layer_input,
)
from .prefill import evaluate

__all__ = [
    "LayerResult",
    "LayerSchedule",
    "decode_layer_rotations",
    "decode_sparse_bootstrap_log_slots",
    "evaluate",
    "evaluate_decode",
    "inspect_decode_layer_input",
    "operation_audit",
    "required_rotations",
]
