"""Layer evaluation settings shared by the application, graph, and static audits.

LayerSchedule carries decisions that remain constant across a run. Ciphertexts,
weights, layouts, and per-layer values are passed separately. The CLI resolves
steady_state_attention_refresh from the layer count.
"""

from __future__ import annotations

from dataclasses import dataclass

from .operators.linear import LLAMA3_LINEAR_SCHEDULES


@dataclass(frozen=True)
class LayerSchedule:
    """All run-constant evaluation decisions, with production defaults."""

    # -- refresh placement -------------------------------------------------
    #: Refresh QK scores before Softmax the way steady-state layers do.
    steady_state_attention_refresh: bool = False
    #: Tri-state: None lets the layer graph decide from the refresh context.
    refresh_softmax_exp_inputs: bool | None = None
    refresh_softmax_exp_outputs: bool = False
    #: Preserve the residual captured before the fused bootstrap.
    #: Borrow the pre-bootstrap residual instead of materializing a
    #: refreshed copy.
    preserve_prebootstrap_residual: bool = True
    pair_residual_bootstrap: bool = True

    # -- softmax -----------------------------------------------------------
    softmax_main_deep_bootstrap: bool = False
    pair_softmax_pre_exp_bootstrap: bool = True
    reuse_softmax_round_square: bool = True
    direct_softmax_slim_inactive_fill: bool = False
    fuse_softmax_slim_chebyshev_map: bool = True
    #: One native pass for the main Softmax checkpoint; None would defer
    #: to the bootstrap program's own two.
    softmax_main_bootstrap_iterations: int | None = 1
    softmax_lambda_bootstrap_iterations: int | None = 1
    match_softmax_single_pass_bootstrap_level: bool = True
    softmax_bootstrap_iterations: int = 2
    softmax_bootstrap_precision: int = 8

    # -- bootstrap budgets -------------------------------------------------
    bootstrap_post_levels: int = 19
    bootstrap_log_slots: int = 15
    bootstrap_level_budget: tuple[int, int] = (4, 4)
    residual_bootstrap_level_drop: int = 0
    residual_bootstrap_iterations: int = 1
    input_residual_bootstrap_iterations: int | None = None
    post_attention_residual_bootstrap_iterations: int | None = None
    residual_bootstrap_precision: int = 14
    #: Sparse log_slots for the RMSN2 rsqrt slim refresh only. None keeps
    #: the full-slot program. RMSN1 recovery stays on the main program.
    rmsnorm2_slim_log_slots: int | None = None

    # -- rmsnorm / mlp fusions ---------------------------------------------
    adaptive_rmsnorm_slim_refresh: bool = True
    fuse_rmsnorm_output_scale: bool = True
    fuse_silu_domain_map: bool = True
    mlp_input_level_drop: int = 0

    # -- attention ---------------------------------------------------------
    batched_sparse_attention_input: bool = True
    #: Absolute Layer0 alignment target; a multi-layer run must clear it to
    #: None because a bootstrap has raised the modulus chain in between.
    qk_output_level: int | None = 17
    attention_rotation_chunk_size: int = 8
    pv_complex_delta_folding: bool = True
    pv_prefold_after_softmax: bool = False
    pairfm_bootstrap_mode: str = "auto"
    pairfm_post_refresh_input_limbs: int | None = 6
    #: Drop emitted / appended K_C and V_C to this limb count.  None keeps
    #: adapter-exit limbs.  Compact QK + degree-8 exp + post-exp BS needs
    #: about 12 without a pre-exp score bootstrap; 4/6/8 starve that path.
    kv_limbs: int | None = None
    #: PairFold V_C limb target when different from K_C. None uses kv_limbs.
    kv_value_limbs: int | None = None

    # -- linear kernels ----------------------------------------------------
    linear_baby_steps: int = LLAMA3_LINEAR_SCHEDULES.qkv.baby_steps
    linear_anchor_step: int = LLAMA3_LINEAR_SCHEDULES.qkv.baby_anchor_step
    qkv_hoist_strategy: str = LLAMA3_LINEAR_SCHEDULES.qkv.hoist_strategy
    output_hoist_strategy: str = LLAMA3_LINEAR_SCHEDULES.w_o.hoist_strategy
    mlp_hoist_strategy: str = LLAMA3_LINEAR_SCHEDULES.gate_up.hoist_strategy
    qkv_max_plaintext_rows: int = LLAMA3_LINEAR_SCHEDULES.qkv.max_plaintext_rows
    output_max_plaintext_rows: int = (
        LLAMA3_LINEAR_SCHEDULES.w_o.max_plaintext_rows
    )
    mlp_max_plaintext_rows: int = (
        LLAMA3_LINEAR_SCHEDULES.gate_up.max_plaintext_rows
    )
    sparse_max_baby_offsets: int = 32
    shear_baby_steps: int = 16

    # -- decode shifted-lane (SL) packing ---------------------------------
    #: Run the four decode linears (QKV, folded W_O, gate/up, down) in the
    #: shifted-lane packing (:mod:`llama3fhe.layouts.shifted_lane`): token
    #: lane ``l`` of the DecodeFM ciphertext carries the input rotated by
    #: ``hidden_dim / 8 * l`` channels, so one 4096-wide block needs 512
    #: instead of 4096 plaintext diagonals and the output is lane-folded.
    #: Costs one extra level at the gate/up exit and at the down / W_O exit
    #: (lane-0 mask); the residual stream, both RMSNorms and the PairFold
    #: routes stay in the lane-0 form.  Prefill is untouched.
    decode_shifted_lanes: bool = False
    decode_sl_baby_steps: int = 32

    # -- decode sparse bootstrap -------------------------------------------
    #: Periodic slim pack of both Decode RMSNorms
    #: (:func:`llama3fhe.operators.norm.rmsnorm.bootstrap_rmsnorm_fhe`
    #: ``slim_period_rows``).  The DecodeFM stream is one cipher, so ``1`` is
    #: the natural period: the slim message repeats every
    #: ``tokens_per_cipher`` (8) slots, the extract broadcast needs zero
    #: rotations and the rsqrt refresh runs through the ``log_slots=3`` sparse
    #: program when :attr:`bootstrap_sparse_log_slots` carries it.  ``None``
    #: keeps the dense row pack (Prefill never reads this field).
    decode_rmsnorm_slim_period_rows: int | None = None
    #: Extra sparse-slot bootstrap programs prepared beside the dense one
    #: (:class:`~llama3fhe.operators.nonlinear.bootstrap.BootstrapConfig`
    #: ``sparse_log_slots``) and whose rotations join the keygen plan.  The
    #: Decode graph wants ``(3, 12)``: the period-1 slim RMSNorm row and the
    #: merged-page Softmax ``total`` / ``filled`` / ``lambda`` rows (period
    #: ``decode_page_merge_stride`` = 4096 slots); see
    #: :func:`llama3fhe.graph.decode.decode_sparse_bootstrap_log_slots` /
    #: :func:`llama3fhe.model.with_decode_sparse_bootstrap`.  Empty by
    #: default so Prefill-only runs pay no extra keys.
    bootstrap_sparse_log_slots: tuple[int, ...] = ()

    # -- memory / profiling ------------------------------------------------
    #: Evict completed QK/PV CUDA rotation and precompute material at their
    #: synchronized last-use boundaries while retaining every downstream key.
    #: Opt-in because the extra synchronizations trade throughput for headroom.
    evict_attention_exclusive_cuda_cache: bool = False
    evict_rotation_cuda_cache_before_softmax_bootstrap: bool = False
    evict_rotation_cuda_cache_before_output_projection: bool = False
    evict_rotation_cuda_cache_before_mlp: bool = False
    evict_rotation_cuda_cache_after_layer: bool = False
    profile_device_memory: bool = False


__all__ = ["LayerSchedule"]
