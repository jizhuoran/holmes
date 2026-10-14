"""E6 micro: the real ``encode_transform_rows`` path (constant_bundle -> ConstantBundle.plaintext) on Holmes's context."""
import argparse, json, os, statistics, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--rows', default='256,512')
ap.add_argument('--limbs', default='6,13,19')
ap.add_argument('--depth', type=int, default=37)
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--out', default=None)
args = ap.parse_args()
os.environ.setdefault('LLAMA_FHE_PLAIN_CACHE_GB', '20')
os.environ.setdefault('LLAMA_FHE_PLAIN_CACHE_MAX_SOURCE_MB', '4096')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import easyfhe as torch  # noqa: E402
from llama3fhe.backend import create_sim_context  # noqa: E402
from llama3fhe.config import SimulatorConfig  # noqa: E402
from llama3fhe.operators.transform import encode_transform_rows  # noqa: E402

t0 = time.time()
ctx = create_sim_context(SimulatorConfig(logN=16, maxLevelsRemaining=args.depth, dnum=3, dcrtBits=59, firstMod=60,
                                         device='cuda'), input_dim=128, block_out=128, rotations=(),
                         include_default_rotations=False)
L = int(ctx.context.L)
SLOTS = int(ctx.max_slots)
print(f'context L={L} slots={SLOTS} {time.time() - t0:.1f}s', flush=True)


def wall(fn, reps=args.reps):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); r = fn(); torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1e3)
        del r
    return round(statistics.median(out), 1)


res = {}
for rows in [int(v) for v in args.rows.split(',')]:
    rng = np.random.default_rng(rows)
    masks = (rng.uniform(0, 1, (rows, SLOTS)) < 0.01).astype(np.float64)   # sparse 0/1 like the layout masks
    masks.flags.writeable = False
    # 1. the host->device part alone
    res[f'r{rows}_bundle_transient_ms'] = wall(lambda: ctx.plaintext_cache.transient(
        lambda c: None, source_bytes=masks.nbytes))
    res[f'r{rows}_h2d_ms'] = wall(lambda: ctx.tensor.from_numpy(np.ascontiguousarray(masks)).to(ctx.device))
    res[f'r{rows}_signature_ms'] = wall(lambda: __import__('llama3fhe.backend', fromlist=['x'])._array_cache_signature(masks))
    for limbs in [int(v) for v in args.limbs.split(',')]:
        if limbs > L:
            continue
        level = L - limbs
        for is_ext in (False, True):
            name = f'micro.r{rows}.l{limbs}.{"ext" if is_ext else "plain"}'
            # transient (cache disabled for this name: oversize budget trick -> use a fresh name every call)
            counter = [0]

            def fresh():
                counter[0] += 1
                return encode_transform_rows(masks, name=f'{name}.n{counter[0]}', level=level, crypto_context=ctx,
                                             is_ext=is_ext)
            # distinct names -> distinct keys -> misses; stop the cache from keeping them
            ctx.plaintext_cache.limit_bytes = 0
            res[f'r{rows}_l{limbs}_{"ext" if is_ext else "plain"}_encode_ms'] = wall(fresh)
            ctx.plaintext_cache.limit_bytes = 20 << 30
            pt = encode_transform_rows(masks, name=name, level=level, crypto_context=ctx, is_ext=is_ext)
            nbytes = sum(int(c.numel()) * 8 for c in pt.cv)
            res[f'r{rows}_l{limbs}_{"ext" if is_ext else "plain"}_pt_gib'] = round(nbytes / 2**30, 3)
            res[f'r{rows}_l{limbs}_{"ext" if is_ext else "plain"}_hit_ms'] = wall(
                lambda: encode_transform_rows(masks, name=name, level=level, crypto_context=ctx, is_ext=is_ext))
            del pt
            ctx.clear_plaintext_cache(); ctx.plaintext_cache.closed = False
            torch.cuda.empty_cache()
    print(json.dumps({k: v for k, v in res.items() if k.startswith(f'r{rows}_')}, indent=1), flush=True)
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)
