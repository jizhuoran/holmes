"""E7: fused Paterson-Stockmeyer linear combinations (HOLMES_CHEB_WSUM) on Holmes's context.

Times ``eval_chebyshev_series_cipher`` for the production rsqrt / SiLU / exp degrees with the per-term scalar
loop (``HOLMES_CHEB_WSUM=0``) and the fused ``grouped_scalar_weighted_acc`` path, and checks that both give the
same residues (``HOLMES_CHEB_WSUM_CHECK=1`` inside the fused run raises on any difference).  The polynomial is
evaluated on a fresh ciphertext at ``--limbs`` (the rsqrt in rmsnorm starts around 14-16 limbs).
usage: python bench/cmp/micro_cheb_wsum.py [--degrees 65,385,769,2049] [--limbs 16] [--reps 3] [--out FILE]
"""
import argparse
import importlib
import json
import os
import statistics
import time
import types

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--degrees', default='49,65,385,769,2049')
ap.add_argument('--limbs', type=int, default=16)
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--depth', type=int, default=37)
ap.add_argument('--out', default=None)
args = ap.parse_args()

import easyfhe as torch  # noqa: E402
from easyfhe import fhe  # noqa: E402

LOGN, N = 16, 1 << 16
SLOTS = N // 2
client, ctx = fhe.generate_client_context(fhe.CKKSContextSpec(
    depth=args.depth, log_n=LOGN, dnum=5, dcrt_bits=59, first_mod=60, secret_key_dist='SPARSE_TERNARY',
    scale_mode='fixed', rescale_policy='manual', rotations=(1,), auto_load_keys=True), device='cuda')
cc = types.SimpleNamespace(fhe=fhe, context=ctx, device='cuda')


def fresh(limbs):
    ct = client.encrypt(np.random.uniform(-0.9, 0.9, SLOTS), slots=SLOTS)
    out = fhe.align_to(ct, ct.state.replace(cur_limbs=limbs, scale_degree=1), ctx)
    return out.cipher_like([x[..., :limbs, :].contiguous() for x in out.cv], batch_size=out.batch_size)


def load_poly(fused, check):
    os.environ['HOLMES_CHEB_WSUM'] = '1' if fused else '0'
    os.environ['HOLMES_CHEB_WSUM_CHECK'] = '1' if check else '0'
    import llama3fhe.operators.nonlinear.polynomial as poly
    return importlib.reload(poly)


def run(poly, ct, coeffs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = poly.eval_chebyshev_series_cipher(ct, coeffs, lower_bound=-1.0, upper_bound=1.0, crypto_context=cc,
                                            input_is_chebyshev_mapped=True)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


results = {'gpu': torch.cuda.get_device_name(0), 'limbs': args.limbs, 'rows': []}
rng = np.random.default_rng(0)
for degree in [int(v) for v in args.degrees.split(',')]:
    coeffs = rng.normal(size=degree + 1) / (1.0 + np.arange(degree + 1)) ** 2
    ct = fresh(args.limbs)
    row = {'degree': degree}
    for name, fused, check in (('loop', False, False), ('fused_check', True, True), ('fused', True, False)):
        poly = load_poly(fused, check)
        times = []
        out = None
        for _ in range(args.reps):
            if out is not None:
                fhe.release(out) if hasattr(fhe, 'release') else None
            out, dt = run(poly, ct, coeffs)
            times.append(dt)
        row[name + '_s'] = round(statistics.median(times), 4)
        row[name + '_limbs'] = int(out.state.cur_limbs)
        dec = client.decrypt(out)
        dec = dec.cpu().numpy() if hasattr(dec, 'cpu') else np.asarray(dec)
        row[name + '_dec0'] = float(np.real(dec.ravel()[0]))
        print(name, row[name + '_s'], flush=True)
    row['speedup'] = round(row['loop_s'] / row['fused_s'], 2)
    print(row, flush=True)
    results['rows'].append(row)
if args.out:
    json.dump(results, open(args.out, 'w'), indent=1)
