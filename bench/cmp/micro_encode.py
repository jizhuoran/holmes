"""E6 micro: where the per-chunk plaintext encode time goes (Holmes encodes ~1800 weight plaintexts per layer online).

Times, for one 512-row chunk (32768 slots) on Holmes's context: stage 1 (``cv_pre_encode_stage1``: special inverse
FFT of the slot values, complex128) and stage 2 (``cv_encode``: scale + round + NTT per limb) at 6 / 9 / 13 limbs.
usage: python micro_encode.py [--rows 512] [--limbs 6,9,13] [--out FILE]
"""
import argparse, json, os, statistics, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--rows', type=int, default=512)
ap.add_argument('--limbs', default='6,9,13')
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--out', default=None)
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import easyfhe as torch  # noqa: E402
from easyfhe import fhe  # noqa: E402
from easyfhe.fhe.ops import kernels as F  # noqa: E402
from easyfhe.fhe.ops.encoding import encode_stage1_packed, encode_stage2  # noqa: E402

N = 1 << 16; SLOTS = N // 2
t0 = time.time()
client, ctx = fhe.generate_client_context(fhe.CKKSContextSpec(
    depth=37, log_n=16, dnum=5, dcrt_bits=59, first_mod=60, secret_key_dist='SPARSE_TERNARY',
    scale_mode='fixed', rescale_policy='manual', rotations=(), auto_load_keys=True), device='cuda')
print(f'context {time.time() - t0:.1f}s', flush=True)


def timeit(fn, reps=args.reps):
    fn(); torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1))
    return round(statistics.median(out), 3)


rows = args.rows
host = np.random.uniform(-1, 1, (rows, SLOTS)).astype(np.float64)
dev_real = torch.from_numpy(host).cuda()
dev_cplx = torch.from_numpy(host.astype(np.complex128)).cuda()
res = {'rows': rows, 'gpu': torch.cuda.get_device_name(0)}
res['h2d_f64_ms'] = timeit(lambda: torch.from_numpy(host).cuda())
res['h2d_c128_ms'] = timeit(lambda: torch.from_numpy(host.astype(np.complex128)).cuda())
res['stage1_ms'] = timeit(lambda: F.cv_pre_encode_stage1(dev_cplx, SLOTS, ctx))
mid = encode_stage1_packed(dev_cplx, SLOTS, ctx)
for limbs in [int(v) for v in args.limbs.split(',')]:
    res[f'stage2_l{limbs}_ms'] = timeit(lambda: encode_stage2(mid, None, SLOTS, False, ctx, cur_limbs=limbs))
    pt = encode_stage2(mid, None, SLOTS, False, ctx, cur_limbs=limbs)
    # the MAC that consumes it, for scale: 16 inputs x rows pt*ct products at this limb count
    res[f'stage2_l{limbs}_us_per_pt'] = round(res[f'stage2_l{limbs}_ms'] * 1000 / rows, 1)
res['stage1_us_per_pt'] = round(res['stage1_ms'] * 1000 / rows, 1)
print(json.dumps(res, indent=1))
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)
