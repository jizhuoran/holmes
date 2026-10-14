"""E6 micro: Holmes's per-input BSGS baby MAC (EasyFHE batched_pairwise_mac x inputs) vs the fused multi-input
kernel (bench/cmp/holmes_multi_mac.py) on one 512-row plaintext chunk (8 giants x 64 babies), Holmes's context.

usage: python micro_multi_mac.py [--limbs 6,9,13] [--inputs 16] [--ti 1,2,4] [--out FILE] [--giant]
Checks bit-equality of the partial sums against the per-input path, then times both (CUDA events, median).
``--giant`` also times the whole hoisted_mac_sum tail (giant step) so the chunk-level saving can be read off.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--limbs', default='6,9,13')
ap.add_argument('--inputs', type=int, default=16)
ap.add_argument('--babies', type=int, default=64)
ap.add_argument('--giants', type=int, default=8)
ap.add_argument('--token-lanes', type=int, default=16, help='baby offset stride (plan.baby_offsets = b * lanes)')
ap.add_argument('--ti', default='2,10108,10204,10208,10402,10404')
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--giant', action='store_true')
ap.add_argument('--sylph', action='store_true', help='install Sylph key-switch kernels (as the layer runs do) and time the batched giant step')
ap.add_argument('--hoist', action='store_true', help='also time the double-hoisted giant step (giant_step_hoisted; adds the g * giant_offset keys)')
ap.add_argument('--out', default=None)
args = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import easyfhe as torch  # noqa: E402
from easyfhe import fhe  # noqa: E402
from easyfhe.fhe.ops import rotation  # noqa: E402
import holmes_multi_mac  # noqa: E402

if args.sylph:
    import holmes_sylph_kernels  # noqa: E402
    print('sylph kernels:', {k: v for k, v in holmes_sylph_kernels.install(hostcache=True).items() if k != 'modules'})

LOGN, N = 16, 1 << 16
SLOTS = N // 2
B, G, LANES = args.babies, args.giants, args.token_lanes
baby_offsets = tuple(b * LANES for b in range(B))
giant_offset = B * LANES
rots = set(baby_offsets[1:]) | {giant_offset}
if args.hoist:
    rots |= set(holmes_multi_mac.hoist_key_offsets(giant_offset, G)) | {G * giant_offset}
t0 = time.time()
client, ctx = fhe.generate_client_context(fhe.CKKSContextSpec(
    depth=37, log_n=LOGN, dnum=5, dcrt_bits=59, first_mod=60, secret_key_dist='SPARSE_TERNARY',
    scale_mode='fixed', rescale_policy='manual', rotations=tuple(sorted(rots)), auto_load_keys=True), device='cuda')
print(f'context {time.time() - t0:.1f}s L={ctx.L} gpu={torch.cuda.get_device_name(0)}', flush=True)


def timeit(fn, reps=args.reps, warm=1):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1))
    return round(statistics.median(out), 3)


def at_limbs(ct, limbs):
    out = fhe.align_to(ct, ct.state.replace(cur_limbs=limbs, scale_degree=1), ctx)
    return out.cipher_like([x[..., :limbs, :].contiguous() for x in out.cv], batch_size=out.batch_size)


def plaintexts(limbs, rows):
    from easyfhe.fhe.ops.encoding import encode_stage1, encode_stage2
    vals = np.random.uniform(-1, 1, (rows, SLOTS))
    prep = encode_stage1(vals, SLOTS, context=ctx)
    return encode_stage2(prep, None, SLOTS, False, ctx, cur_limbs=limbs)


results = {'gpu': torch.cuda.get_device_name(0), 'inputs': args.inputs, 'babies': B, 'giants': G, 'rows': {}}
for limbs in [int(v) for v in args.limbs.split(',')]:
    cts = [at_limbs(client.encrypt(np.random.uniform(-1, 1, SLOTS), slots=SLOTS), limbs) for _ in range(args.inputs)]
    babies = [fhe.prepare_hoisted_baby_rotations(ct, baby_offsets, ctx, strategy='normal', baby_anchor_step=B)
              for ct in cts]
    pts = plaintexts(limbs, G * B)
    torch.cuda.synchronize()
    _, specs = rotation._baby_block_specs(baby_offsets, B)

    def ref_partials():
        return [rotation._grouped_mac_baby_blocks(bb, pts, G, ctx, baby_count=B, specs=specs) for bb in babies]

    def ref_full():
        return [fhe.hoisted_mac_sum(ct, baby_offsets, pts, giant_offset, G, ctx, strategy='normal',
                                    baby_anchor_step=B, baby_rotations=bb) for ct, bb in zip(cts, babies)]

    row = {'ref_partials_ms': timeit(ref_partials)}
    ref = ref_partials()
    for ti in [int(v) for v in args.ti.split(',')]:
        got = holmes_multi_mac.partial_sums([bb[0] for bb in babies], pts, G, ctx, ti=ti)
        if got is None:
            raise SystemExit('partial_sums returned None (layout rejected)')
        bad = 0
        for a, b in zip(got, ref):
            for c in range(2):
                bad += holmes_multi_mac.count_diff(a.cv[c], b.cv[c])
        row[f'multi_ti{ti}_ms'] = timeit(lambda: holmes_multi_mac.partial_sums([bb[0] for bb in babies], pts, G, ctx, ti=ti))
        row[f'multi_ti{ti}_mismatch'] = bad
        del got
    if args.giant:
        row['ref_full_ms'] = timeit(ref_full)

        best_ti = min((int(v) for v in args.ti.split(',')), key=lambda t: row[f'multi_ti{t}_ms'])
        row['best_ti'] = best_ti

        def multi_full(ti=best_ti):
            parts = holmes_multi_mac.partial_sums([bb[0] for bb in babies], pts, G, ctx, ti=ti)
            return [holmes_multi_mac._giant_step(p, giant_offset, G, ctx, 'normal') for p in parts]

        row['multi_full_best_ms'] = timeit(multi_full)
        ks = holmes_multi_mac._sylph_keyswitch()
        if ks is not None:
            def multi_full_batched(ti=best_ti):
                res, _ = holmes_multi_mac.partial_sums_raw([bb[0] for bb in babies], pts, G, ctx, ti=ti)
                return holmes_multi_mac.giant_step_batched(res, giant_offset, G, ctx, ks)

            # Sylph's batched key switch rounds ModDown differently from EasyFHE's single-cipher cv_hrot:
            # compare the decoded messages (both are ~1e-12 from the exact value at these limbs)
            def dec(ct):
                v = client.decrypt(ct)
                return np.real(np.asarray(v.cpu().numpy() if hasattr(v, 'cpu') else v))

            full_ref = ref_full()
            out0, out1 = multi_full_batched()
            worst = scale = 0.0
            for i in range(args.inputs):
                want = dec(full_ref[i])
                got = dec(full_ref[i].cipher_like([out0[i], out1[i]], batch_size=1))  # same state (scale degree 2)
                worst = max(worst, float(np.max(np.abs(got - want))))
                scale = max(scale, float(np.max(np.abs(want))))
            row['multi_full_batched_best_ms'] = timeit(multi_full_batched)
            row['multi_full_batched_dec_maxdiff'] = worst
            row['multi_full_batched_dec_scale'] = scale
            if args.hoist:
                for keys in (4, 7):
                    def multi_full_hoisted(ti=best_ti, keys=keys):
                        res, _ = holmes_multi_mac.partial_sums_raw([bb[0] for bb in babies], pts, G, ctx, ti=ti)
                        return holmes_multi_mac.giant_step_hoisted(res, giant_offset, G, ctx, ks, max_keys=keys)

                    h0, h1 = multi_full_hoisted()
                    worst_h = 0.0
                    for i in range(args.inputs):
                        want = dec(full_ref[i])
                        got = dec(full_ref[i].cipher_like([h0[i], h1[i]], batch_size=1))
                        worst_h = max(worst_h, float(np.max(np.abs(got - want))))
                    row[f'multi_full_hoisted{keys}_best_ms'] = timeit(multi_full_hoisted)
                    row[f'multi_full_hoisted{keys}_dec_maxdiff'] = worst_h
                    del h0, h1
                # giant step alone (same partial sums), Horner vs hoisted
                res_g, _ = holmes_multi_mac.partial_sums_raw([bb[0] for bb in babies], pts, G, ctx, ti=best_ti)
                row['giant_horner_ms'] = timeit(lambda: holmes_multi_mac.giant_step_batched(res_g, giant_offset, G, ctx, ks))
                row['giant_hoisted4_ms'] = timeit(lambda: holmes_multi_mac.giant_step_hoisted(res_g, giant_offset, G, ctx, ks, max_keys=4))
                row['giant_hoisted7_ms'] = timeit(lambda: holmes_multi_mac.giant_step_hoisted(res_g, giant_offset, G, ctx, ks, max_keys=7))
                row['giant_horner_base_ms'] = timeit(lambda: holmes_multi_mac.giant_step_batched(res_g, giant_offset, G, ctx, ks, base_offset=G * giant_offset))
                del res_g
            del full_ref, out0, out1
    ptxct = args.inputs * G * B
    row['ref_us_per_ptxct_limb'] = round(row['ref_partials_ms'] * 1000 / (ptxct * limbs), 3)
    results['rows'][str(limbs)] = row
    print(f'limbs {limbs}: {row}', flush=True)
    del cts, babies, pts, ref
    torch.cuda.empty_cache()

print(json.dumps(results, indent=1))
if args.out:
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
