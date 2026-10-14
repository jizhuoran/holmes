"""Batched pair bootstrap for Holmes on Slot-Coeff's batched EasyFHE pieces (``HOLMES_BATCH_BS=1``).



This module installs :func:`bootstrap_batch` as ``BootstrapOperator.batch_backend`` so ``refresh_many`` (used by
``stations.pair_bootstrap`` and ``prefix_softmax._refresh_partition``) runs the whole group through one pass:

  raise (per ciphertext, EasyFHE ``raise_ciphertext``)  ->  stack  ->  ``bsgs.coeffs_to_slots`` (batched BSGS MAC)
  ->  conjugate / real / imag  ->  ``evalmod.eval_mod`` x 2 (batched)  ->  recombine, ``scale_after_approx``
  ->  ``bsgs.slots_to_coeffs``  ->  ``scale_to_original_message``  ->  normalize / align to ``program.output_state``
  ->  unstack.

Every step is the one EasyFHE's ``bootstrap_modraise_first`` runs per ciphertext (same constants, same plan, same
residues per Sylph's ``bsgs`` / ``evalmod`` docstrings); only the batching differs.  Requires the Sylph kernels
(``bench/cmp/holmes_sylph_kernels.install()`` first, i.e. ``HOLMES_SYLPH_KERNELS=1``).  ``HOLMES_BATCH_BS_MAX`` (default 16)
caps the batch; larger groups run in chunks.  ``HOLMES_BATCH_BS=0`` leaves the operator alone (A/B baseline: the same
``refresh_many`` call then loops over ``refresh``).
"""
from __future__ import annotations

import os
import sys


def enabled() -> bool:
    return os.environ.get("HOLMES_BATCH_BS", "0") == "1"


def max_batch() -> int:
    return max(1, int(os.environ.get("HOLMES_BATCH_BS_MAX", "16")))


def _stack(cts):
    import easyfhe as torch

    if len(cts) == 1:
        return cts[0]
    cv = [torch.stack([c.cv[i] if c.cv[i].dim() == 2 else c.cv[i][0] for c in cts], 0) for i in range(len(cts[0].cv))]
    return cts[0].cipher_like(cv, batch_size=len(cts))


def _unstack(ct):
    if int(ct.batch_size) == 1:
        return [ct]
    return [ct.cipher_like([x[i] for x in ct.cv], batch_size=1) for i in range(int(ct.batch_size))]


def _monomial_inplace(ct, degree, ctx):
    from easyfhe.fhe.ops import kernels as F

    for comp in ct.cv:
        F.cv_mul_by_monomial(comp, ct.state.cur_limbs, int(degree), ctx)
    return ct


def _eval_mod(part, ctx, consts, plan, evalmod):
    if evalmod.enabled():
        return evalmod.eval_mod(part, ctx, consts, plan)
    from easyfhe.bs.openfhe.runtime import approx as bs_approx

    # SYLPH_EVALMOD=0 ablation: EasyFHE's EvalMod takes one ciphertext
    return _stack([bs_approx.eval_bootstrap_approx_mod(item, ctx, consts, plan) for item in _unstack(part)])


def bootstrap_batch(ciphers, ctx, program):
    """``[easyfhe.bs.openfhe.bootstrap(c, ctx, program) for c in ciphers]`` as one batched pass."""

    from easyfhe.fhe.ops import alignment, arithmetic
    from easyfhe.bs.openfhe.runtime.raising import raise_ciphertext
    from easyfhe.bs.openfhe.runtime.scaling import scale_after_approx, scale_to_original_message
    from sylph_easy.easyfhe_ext import bsgs, evalmod
    from sylph_easy.easyfhe_ext.rotate import rotate as fast_rotate

    ciphers = list(ciphers)
    cap = max_batch()
    if len(ciphers) > cap:
        out = []
        for start in range(0, len(ciphers), cap):
            out.extend(bootstrap_batch(ciphers[start:start + cap], ctx, program))
        return out

    consts, plan = program.constants, program._runtime_plan
    raise_to = int(program.raise_to_limbs)
    slots = int(ciphers[0].slots)
    if slots != int(ctx.M) // 4 or int(plan.slots) != slots:
        # sparse-slot programs keep EasyFHE's per-ciphertext path (replicate / restore steps)
        import easyfhe.bs.openfhe as bs
        return [bs.bootstrap(c, ctx, program) for c in ciphers]

    raised = [raise_ciphertext(c, ctx, consts, raise_to) for c in ciphers]
    if any(r.state != raised[0].state for r in raised):
        import easyfhe.bs.openfhe as bs
        return [bs.bootstrap(c, ctx, program) for c in ciphers]
    stack = _stack(raised)

    # CoeffsToSlots on the batch
    encoded = bsgs.coeffs_to_slots(stack, ctx, consts, plan)
    # EvalMod (full-slot variant of easyfhe ... runtime.evalmod.eval_mod_full) on the batch
    conjugate = fast_rotate(encoded, 2 * int(ctx.N) - 1, ctx)
    imag = arithmetic.homo_sub(encoded, conjugate, ctx)
    real = arithmetic.homo_add(encoded, conjugate, ctx)
    imag = _monomial_inplace(imag, 3 * int(ctx.M) // 4, ctx)
    real = alignment.normalize_scale(real, ctx)
    imag = alignment.normalize_scale(imag, ctx)
    real = _eval_mod(real, ctx, consts, plan, evalmod)
    imag = _eval_mod(imag, ctx, consts, plan, evalmod)
    imag = _monomial_inplace(imag, int(ctx.M) // 4, ctx)
    encoded = arithmetic.homo_add(real, imag, ctx)
    encoded = scale_after_approx(encoded, ctx, consts, raise_to_limbs=raise_to)
    encoded = alignment.normalize_scale(encoded, ctx)
    # SlotsToCoeffs on the batch, then EasyFHE's output alignment
    decoded = bsgs.slots_to_coeffs(encoded, ctx, consts, plan)
    decoded = scale_to_original_message(decoded, ctx, consts)
    decoded = decoded.cipher_like(decoded.cv, slots=slots)
    decoded = alignment.normalize_scale(decoded, ctx)
    if decoded.state != program.output_state:
        decoded = alignment.align_to(decoded, program.output_state, ctx)
    return _unstack(decoded)


def install() -> dict:
    """Register :func:`bootstrap_batch` on ``BootstrapOperator`` (needs the Sylph kernels installed)."""

    root = os.environ.get("SYLPH_ROOT", "./third_party/sylph-easy")
    if root not in sys.path:
        sys.path.insert(0, root)
    from sylph_easy.easyfhe_ext import bsgs, evalmod, keyswitch

    if not keyswitch.enabled():
        raise RuntimeError("HOLMES_BATCH_BS needs the Sylph key-switch kernels (HOLMES_SYLPH_KERNELS=1)")
    from llama3fhe.operators.nonlinear.bootstrap import BootstrapOperator

    BootstrapOperator.batch_backend = staticmethod(bootstrap_batch)
    return {"batch_bs": True, "max_batch": max_batch(), "bsgs": bsgs.enabled(), "evalmod": evalmod.enabled()}


def uninstall() -> None:
    from llama3fhe.operators.nonlinear.bootstrap import BootstrapOperator

    BootstrapOperator.batch_backend = None
