"""Static (theoretical) ciphertext-level operation counts per transformer stage for both schemes.

Units match ``opcount`` API labels: ``rot`` = ciphertext rotations (incl. conjugations), ``ctxct`` = ciphertext
products with relinearisation, ``ptxct`` = ciphertext x plaintext products, ``bootstrap`` / ``s2c`` /
``halfboot`` = ciphertexts refreshed / converted.  Polynomial evaluations (rsqrt, exp, SiLU, EvalMod) are
reported as ``poly`` with their degrees in ``notes`` and are *not* expanded into ctxct: compare those from the
measured ``api.ctxct`` of the ``inverse_sqrt`` / ``softmax`` / ``silu`` regions.

Holmes: the repository's own static audits (``*_complexity``) at their default audit parameters (128 tokens,
hidden 4096, 16 feature-major ciphertexts, QKV carrier 3072, baby 64).  Where the runtime deviates from the
audit (e.g. slim RMSNorm broadcasts over 16 rows = 4 doubling steps instead of 12), the measured number is
below the audit and the deviation is the thing to explain.
Slot-Coeff: closed-form counts from the Sylph layout (16 slot cts of 512 features x 64 tokens, K = N / 128 =
512 coefficients per output block, Rhombus split by ``best_split``).
"""
from __future__ import annotations

import math

HIDDEN, INTER, TOKENS, N = 4096, 14336, 128, 1 << 16
K = N // TOKENS                      # 512 outputs per coefficient block
H = int(math.log2(N // K))           # 7
LOG_K = int(math.log2(K))            # 9
SC_CTS = 2 * HIDDEN // 512           # 16 slot ciphertexts per 4096-feature activation


def _best_split(blocks, packs):
    cost = lambda a: blocks * ((1 << a) - 1) + packs * ((1 << (LOG_K - a)) - 1)
    return H + min(range(LOG_K + 1), key=cost)


def sc_pcmm(d_in, d_outs):
    """One S2C'd activation of ``d_in`` features projected to each width in ``d_outs`` (shared split)."""
    blocks, packs = d_in // K, sum(d_outs) // K
    split = _best_split(blocks, packs)
    a, pc = 1 << (split - H), N >> split
    return dict(s2c=blocks, pcmm_rot=blocks * (a - 1) + packs * int(math.log2(pc)), ptxct=packs * blocks * a * pc,     # = packs * blocks * K: each copy meets pc plaintexts
                pcmm=packs, halfboot=packs,
                notes=f"blocks {blocks}, out blocks {packs}, A={a} automorphic copies, pc={pc} packed per output; "
                      f"HalfBoot {packs} coeff ct -> {2 * packs} slot ct")


def sylph():
    t = {}
    rms = dict(ctxct=2 * SC_CTS, rot=2 * int(math.log2(512)), ptxct=0, poly=2,
               notes="square 16 + x*rsqrt 16 ctxct; 9 doubling rotations on the 2 summed cts (lo/hi tokens); "
                     "rsqrt poly + Newton on batch 2 (region inverse_sqrt, measured); gamma folded into QuaRot weights")
    t['RMSNorm(attn)'] = dict(rms)
    t['RMSNorm(ffn)'] = dict(rms)
    t['QKV proj'] = sc_pcmm(HIDDEN, (4096, 1024, 1024))
    t['O proj'] = sc_pcmm(HIDDEN, (4096,))
    gu = sc_pcmm(HIDDEN, (INTER, INTER))
    dn = sc_pcmm(INTER, (HIDDEN,))
    ffn = {k: gu[k] + dn[k] for k in ('s2c', 'pcmm_rot', 'ptxct', 'pcmm', 'halfboot')}
    ffn['ctxct'] = 2 * INTER // 512          # SiLU(gate) * up on 56 cts
    ffn['poly'] = 2 * INTER // 512
    ffn['notes'] = f"gate_up: {gu['notes']}; down: {dn['notes']}; SiLU poly on 56 cts + 56 gated ctxct"
    t['FFN'] = ffn
    t['Attention core'] = dict(notes="per-head slot attention (QK^T, softmax with inverse_sqrt, PV): measured only")
    return t


def holmes():
    t = {}
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # examples/holmes
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from llama3fhe.operators.attention import audit
        from llama3fhe.operators.norm.rmsnorm import rmsnorm_complexity
        from llama3fhe.operators.linear.prefill.mlp import mlp_complexity
        from llama3fhe.operators.attention.prefill.output import pairfm_output_complexity
    except Exception as e:  # noqa: BLE001
        return {'_error': repr(e)}
    FL, AL, CL = audit.FeatureMajorPrefillLayout(), audit.AttentionPairLayout(), audit.LinearCarrierLayout(128, HIDDEN)
    r = rmsnorm_complexity(FL, polynomial_degree=31, apply_gamma=False)
    rms = dict(ctxct=r.mean_square_ct_ct_multiplications + r.output_ct_ct_multiplications,
               rot=r.mean_square_rotations + r.scale_broadcast_rotations + r.slim_pack_rotations + r.slim_extract_rotations,
               ptxct=r.slim_pack_pt_ct_multiplications + r.slim_extract_pt_ct_multiplications + r.gamma_pt_ct_multiplications,
               poly=r.rsqrt_polynomial_evaluations,
               notes=f"audit: 16 ct x log2(4096)=12 sum rotations (192) + 192 broadcast rotations; runtime slim broadcast over "
                     f"16 rows is 4 steps x 16 = 64; rsqrt poly on 1 slim ct; + entry pair bootstrap (residual refresh, "
                     f"not RMSNorm cost) counted under 'refresh'")
    t['RMSNorm(attn)'] = dict(rms)
    t['RMSNorm(ffn)'] = dict(rms)
    q = audit.complex_token_diagonal_qkv_complexity()
    ai = audit.complex_token_attention_complexity(CL, AL)
    t['QKV proj'] = dict(rot=q.rotations_including_conjugations + ai.rotations + ai.conjugations,
                         ptxct=q.pt_ct_multiplications + ai.pt_ct_multiplications, ctxct=0,
                         notes=f"qkv BSGS: {q.pt_ct_multiplications} ptxct, {q.baby_rotations} baby + {q.giant_rotations} giant + "
                               f"{q.chunk_alignment_rotations} align rot; attn_input re-layout: {ai.pt_ct_multiplications} ptxct, "
                               f"{ai.rotations} rot")
    s = audit.delta_softmax_alg2_complexity(AL)
    t['Attention core'] = dict(ctxct=s.explicit_ct_ct_multiplications, ptxct=s.pt_ct_multiplications, rot=s.rotations,
                               bootstrap=s.lambda_bootstraps, poly=s.exp_polynomial_evaluations + s.rough_invsqrt_evaluations + s.precise_invsqrt_evaluations,
                               notes=f"softmax Alg2 audit only ({s.rounds} rounds; exp deg {s.exp_polynomial_degree} x{s.exp_polynomial_evaluations}, "
                                     f"invsqrt {s.rough_invsqrt_degree}/{s.precise_invsqrt_degree}); QK^T / PV BSGS not audited here")
    p = pairfm_output_complexity(FL, AL)
    t['Attn out repack'] = dict(ptxct=p.conversion_pt_ct, rot=p.conversion_rotations + p.conversion_conjugations, ctxct=0,
                                notes=f"PairFM conversion: {p.pairfm_pt_ct} + {p.blocked_pt_ct} + {p.owner_pt_ct} ptxct")
    t['O proj'] = dict(notes="no static audit for W_o (same BSGS family as QKV); measured only")
    m = mlp_complexity()
    t['FFN'] = dict(ptxct=m.pt_ct_multiplications, rot=m.rotations_including_conjugations, ctxct=m.gated_ct_ct_multiplications,
                    poly=m.silu_polynomial_evaluations,
                    notes=f"gate_up {m.gate_up_pt_ct_multiplications} + down {m.down_pt_ct_multiplications} ptxct; baby {m.gate_up_baby_rotations}+{m.down_baby_rotations}, "
                          f"giant {m.gate_up_giant_rotations}+{m.down_giant_rotations}; SiLU deg {m.silu_polynomial_degree} x{m.silu_polynomial_evaluations}")
    return t


if __name__ == '__main__':
    import json
    print(json.dumps({'holmes': holmes(), 'slot_coeff': sylph()}, indent=1))
