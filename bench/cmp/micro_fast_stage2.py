"""E8 micro: fused encode stage 2 (kernels/holmes_encode.cu) vs EasyFHE ``cv_encode`` -- bit equality + device time.

usage: python micro_fast_stage2.py [--rows 512] [--limbs 3,12,13] [--out FILE]
"""
import argparse, json, os, statistics, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--rows', type=int, default=512)
ap.add_argument('--limbs', default='3,12,13')
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--out', default=None)
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import easyfhe as torch  # noqa: E402
from easyfhe import fhe  # noqa: E402
from easyfhe.fhe.ops import kernels as F  # noqa: E402
from easyfhe.fhe.ops.encoding import encode_stage1_packed, encode_stage2  # noqa: E402
import holmes_fast_stage2  # noqa: E402

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
res = {'rows': rows, 'gpu': torch.cuda.get_device_name(0), 'cases': []}
orig = F.cv_encode
for scale_name, amp in (('unit', 1.0), ('large', 1e6), ('tiny', 1e-9)):
    host = np.random.uniform(-amp, amp, (rows, SLOTS)).astype(np.float64)
    dev_cplx = torch.from_numpy(host.astype(np.complex128)).cuda()
    mid = encode_stage1_packed(dev_cplx, SLOTS, ctx)
    values = mid.encoded_values.reshape(rows, 2 * SLOTS)
    for limbs in [int(v) for v in args.limbs.split(',')]:
        sf = float(ctx.scale_at(limbs))
        ref = orig(values, N, limbs, SLOTS, sf, False, ctx)
        out = holmes_fast_stage2.encode_stage2_tensor(values, limbs, sf, ctx)
        torch.cuda.synchronize()
        ndiff = holmes_fast_stage2.count_diff(ref, out)
        equal = ndiff == 0
        case = {'amp': scale_name, 'limbs': limbs, 'bit_equal': equal, 'words_differ': ndiff}
        if scale_name == 'unit':
            case['easyfhe_ms'] = timeit(lambda: orig(values, N, limbs, SLOTS, sf, False, ctx))
            case['fused_ms'] = timeit(lambda: holmes_fast_stage2.encode_stage2_tensor(values, limbs, sf, ctx))
            case['speedup'] = round(case['easyfhe_ms'] / case['fused_ms'], 2)
            case['kernelA_ms'] = timeit(lambda: holmes_fast_stage2.encode_stage2_tensor(values, limbs, sf, ctx, parts=1))
            case['kernelB_ms'] = timeit(lambda: holmes_fast_stage2.encode_stage2_tensor(values, limbs, sf, ctx, parts=2))
            case['kernelA_lpb1_ms'] = timeit(lambda: holmes_fast_stage2.encode_stage2_tensor(values, limbs, sf, ctx, parts=1, lpb=1))
            for mb in (16, 32, 64):
                for lpb in (0, 1):
                    tag = f'fused_l2_{mb}mb_lpb{lpb}'
                    case[tag + '_ms'] = timeit(lambda: holmes_fast_stage2.encode_stage2_tensor(values, limbs, sf, ctx, batch_mb=mb, parts=3, lpb=lpb))
                    outb = holmes_fast_stage2.encode_stage2_tensor(values, limbs, sf, ctx, batch_mb=mb, parts=3, lpb=lpb)
                    case[tag + '_equal'] = holmes_fast_stage2.count_diff(ref, outb) == 0
                    del outb
        print(case, flush=True)
        res['cases'].append(case)
        del ref, out
# the installed path (encode_stage2 -> patched cv_encode) end to end
holmes_fast_stage2.install()
host = np.random.uniform(-1, 1, (rows, SLOTS)).astype(np.float64)
mid = encode_stage1_packed(torch.from_numpy(host.astype(np.complex128)).cuda(), SLOTS, ctx)
pt_fast = encode_stage2(mid, None, SLOTS, False, ctx, cur_limbs=12)
F.cv_encode = orig
pt_ref = encode_stage2(mid, None, SLOTS, False, ctx, cur_limbs=12)
res['installed_path_equal'] = holmes_fast_stage2.count_diff(pt_fast.cv[0], pt_ref.cv[0]) == 0
res['stats'] = dict(holmes_fast_stage2.STATS)
print(json.dumps({k: v for k, v in res.items() if k != 'cases'}, indent=1))
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)
