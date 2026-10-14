"""Atomic CKKS operation microbenchmark: Holmes context vs Slot-Coeff (Sylph) context, native vs fused.

Three columns the comparison needs:
  * ``--scheme holmes``            depth 37 / dnum 5 / BS (5,5)/17 context, EasyFHE native ops (what Holmes runs)
  * ``--scheme sylph --native``  Sylph's L=27 / dnum 3 / BS (3,3) context, EasyFHE native ops (SYLPH_*=0)
  * ``--scheme sylph``           same context, Sylph's hand-written key-switch / hmul / fast-rotate kernels
                                 installed (``keyswitch.install`` + ``evalmod.install``), i.e. what Sylph runs
Every op is timed with CUDA events (median of ``--reps``) at each limb count of ``--limbs``.
usage: python micro_atomic.py --scheme holmes|sylph [--native] [--limbs 6,9,13,16,24] [--out FILE]
                              [--boot A,B,OUT] [--dnum D] [--depth L] [--bs-from LIMBS]

E2 (refresh audit) runs the *other* scheme's parameters: ``--scheme sylph --boot 5,5,17 --dnum 5`` times
Sylph's BS / HalfBoot / S2C on Holmes's context (depth from the bootstrap requirements, 37), and
``--scheme holmes --boot 3,3,10 --dnum 3`` times EasyFHE's bootstrap on Sylph's context.  Defaults are each
scheme's production parameters (holmes: 5,5,17 / dnum 5 / depth 37; sylph: 3,3,10 / dnum 3).
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--scheme', required=True, choices=['holmes', 'sylph'])
ap.add_argument('--native', action='store_true', help='sylph: SYLPH_*=0 (pure EasyFHE kernels)')
ap.add_argument('--limbs', default='6,9,13,16,24')
ap.add_argument('--reps', type=int, default=10)
ap.add_argument('--batch', default='1,16', help='cipher batch sizes to time (comma list)')
ap.add_argument('--out', default=None)
ap.add_argument('--boot', default=None, help='bootstrap level budget + output levels "A,B,OUT" (default per scheme)')
ap.add_argument('--dnum', type=int, default=None, help='key-switch dnum (default per scheme: holmes 5, sylph 3)')
ap.add_argument('--depth', type=int, default=None, help='holmes: context depth (default: bootstrap requirement, 37 for 5,5,17)')
ap.add_argument('--bs-from', default=None, help='comma list of bootstrap input limbs (default: "2,<min(--limbs)>"; EasyFHE needs >= 2)')
ap.add_argument('--only-bs', action='store_true', help='skip the per-op sweep, time only bootstrap / halfboot / s2c')
ap.add_argument('--batch-bs', default=None, metavar='B,B,..',
                help='holmes: also time bench/cmp/holmes_batch_bs.bootstrap_batch (Sylph batched C2S/EvalMod/S2C on '
                     "Holmes's program) at these batch sizes, and check its decrypt error against bs.bootstrap; "
                     'installs the Sylph kernels (SYLPH_ROOT)')
args = ap.parse_args()

if args.scheme == 'sylph' and args.native:
    for k in ('SYLPH_KS', 'SYLPH_HMUL', 'SYLPH_FASTROT', 'SYLPH_EVALMOD', 'SYLPH_BSGS', 'SYLPH_PACK', 'SYLPH_HOSTCACHE', 'SYLPH_SLOTMAC', 'SYLPH_MODUP'):
        os.environ[k] = '0'

import easyfhe as torch  # noqa: E402
from easyfhe import fhe  # noqa: E402
import easyfhe.bs.openfhe as bs  # noqa: E402

LOGN, N = 16, 1 << 16
SLOTS = N // 2
ROTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 2 * N - 1]

t0 = time.time()
boot = [int(v) for v in (args.boot or ('5,5,17' if args.scheme == 'holmes' else '3,3,10')).split(',')]
if args.scheme == 'holmes':
    dnum = args.dnum if args.dnum is not None else 5
    spec_bs = bs.BootstrapSpec(log_slots=15, level_budget=tuple(boot[:2]), output_levels=boot[2], strategy='double_hoist',
                               mode='modraise_first')
    req = bs.requirements(spec_bs, log_n=LOGN)
    depth = args.depth if args.depth is not None else int(req.context_depth)
    rots = set(req.rotations) | set(ROTS)
    client, ctx = fhe.generate_client_context(fhe.CKKSContextSpec(
        depth=depth, log_n=LOGN, dnum=dnum, dcrt_bits=59, first_mod=60, secret_key_dist='SPARSE_TERNARY',
        scale_mode='fixed', rescale_policy='manual', rotations=tuple(sorted(rots)), auto_load_keys=True), device='cuda')
    prog = bs.generate(ctx, spec_bs)
    raise_limbs = int(prog.raise_to_limbs)
    label = f'holmes(depth{depth},dnum{dnum},BS({boot[0]},{boot[1]})/{boot[2]})'
else:
    sys.path.insert(0, os.environ.get('SYLPH_ROOT', './third_party/sylph-easy'))
    from sylph_easy.layer.setup import create
    from sylph_easy.easyfhe_ext import keyswitch
    dnum = args.dnum if args.dnum is not None else 3
    client, eng = create(verbose=True, level_budget=tuple(boot[:2]), output_levels=boot[2], dnum=dnum)
    ctx, prog = eng.ctx, eng.program
    depth = int(ctx.L) - 1
    raise_limbs = int(prog.raise_to_limbs)
    label = f"sylph(L={ctx.L},dnum{dnum},BS({boot[0]},{boot[1]})/{boot[2]},{'native' if args.native else 'fused'})"
    print('sylph keyswitch fused:', keyswitch.enabled())
print(f'[{label}] context L={ctx.L} keygen {time.time() - t0:.1f}s', flush=True)


def timeit(fn, reps=args.reps, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1))
    return round(statistics.median(out), 3)


def fresh(batch=1):
    v = np.random.uniform(-1, 1, SLOTS)
    cts = [client.encrypt(v, slots=SLOTS) for _ in range(batch)]
    return cts[0] if batch == 1 else fhe.pack_cipher_batch(cts)


def at_limbs(ct, limbs):
    if int(ct.state.cur_limbs) == limbs:
        return ct
    out = fhe.align_to(ct, ct.state.replace(cur_limbs=limbs, scale_degree=1), ctx)
    # trim physical limbs to cur_limbs (batched kernels assume they match)
    return out.cipher_like([x[..., :limbs, :].contiguous() for x in out.cv], batch_size=out.batch_size)


def plaintext(limbs, batch):
    from easyfhe.fhe.ops.encoding import encode_stage1, encode_stage2
    vals = np.random.uniform(-1, 1, (batch, SLOTS)) if batch > 1 else np.random.uniform(-1, 1, SLOTS)
    prep = encode_stage1(vals, SLOTS, context=ctx)
    return encode_stage2(prep, None, SLOTS, False, ctx, cur_limbs=limbs)


results = {'label': label, 'scheme': args.scheme, 'native': bool(args.native), 'L': int(ctx.L), 'dnum': dnum,
           'boot': boot, 'depth': depth, 'raise_limbs': raise_limbs, 'gpu': torch.cuda.get_device_name(0), 'ops': {}}
limbs_list = [int(v) for v in args.limbs.split(',') if int(v) <= int(ctx.L)]
batches = [int(v) for v in args.batch.split(',')]
for batch in ([] if args.only_bs else batches):
    base = fresh(batch)
    for limbs in limbs_list:
        ct = at_limbs(base, limbs)
        ct2 = at_limbs(fresh(batch), limbs)
        pt = plaintext(limbs, batch)
        row = {}
        row['rotate'] = timeit(lambda: fhe.homo_rotate(ct, 1, ctx))
        row['conjugate'] = timeit(lambda: fhe.homo_rotate(ct, 2 * N - 1, ctx))
        if batch == 1:
            row['fast_rotate_x8'] = timeit(lambda: fhe.fast_rotate(ct, tuple(ROTS[:8]), ctx))
            row['fast_rotate_x8_ext'] = timeit(lambda: fhe.fast_rotate(ct, tuple(ROTS[:8]), ctx, output_ext=True))
        row['mul_relin'] = timeit(lambda: fhe.homo_mul_relin(ct, ct2, ctx))
        if batch == 1:
            row['mul_relin_rescale'] = timeit(lambda: fhe.homo_mul_relin_rescale_postop(ct, ct2, ctx))
        row['mul_no_relin'] = timeit(lambda: fhe.homo_mul_no_relin(ct, ct2, ctx))
        row['mul_pt'] = timeit(lambda: fhe.homo_mul_pt(ct, pt, ctx))
        row['add'] = timeit(lambda: fhe.homo_add(ct, ct2, ctx))
        row['mul_i'] = timeit(lambda: fhe.homo_mul_i(ct, ctx))
        prod = fhe.homo_mul_pt(ct, pt, ctx)
        row['rescale'] = timeit(lambda: fhe.rescale(prod, ctx))
        results['ops'][f'b{batch}_l{limbs}'] = row
        print(f'batch {batch:2d} limbs {limbs:2d}: ' + ' '.join(f'{k}={v:.3f}' for k, v in row.items()), flush=True)

# bootstrap at several input limb counts; EasyFHE full bootstrap for both.  The raise path does one scalar
# multiply + rescale on the input and then mod_raise reads limb 0 only, so the input limbs should not matter
# (Holmes's residual pair-BS already enters at 2 limbs; the 6 of the first E2 run was just the sweep's floor).
bs_from = [int(v) for v in args.bs_from.split(',')] if args.bs_from else sorted({2, min(limbs_list)})
for batch in [b for b in batches if b == 1]:     # EasyFHE bootstrap is batch-1 only
    for lo in bs_from:
        ct = at_limbs(fresh(batch), lo)
        ms = timeit(lambda: bs.bootstrap(ct, ctx, prog), reps=max(3, args.reps // 2), warm=1)
        results['ops'][f'b{batch}_bootstrap_from_l{lo}'] = {'bootstrap': ms}
        print(f'batch {batch:2d} bootstrap from {lo} limbs -> {prog.output_state.cur_limbs} limbs: {ms:.1f} ms', flush=True)
if args.scheme == 'sylph':
    kit = eng.kit
    hb_in = 2                     # S2C output = coefficient ct with 2 limbs (q0 + one rescale for the raise correction)
    for batch in batches:
        try:
            items = [at_limbs(fresh(1), hb_in) for _ in range(batch)]
            if batch == 1:
                ms = timeit(lambda: kit.halfboot(items[0]), reps=max(3, args.reps // 2), warm=1)
            else:
                ms = timeit(lambda: kit.halfboots(items), reps=max(3, args.reps // 2), warm=1)
            results['ops'][f'b{batch}_halfboot_from_l{hb_in}'] = {'halfboot': ms}
            print(f'batch {batch:2d} halfboot ({batch} coeff ct, {hb_in} limbs -> 2x{batch} slot ct): {ms:.1f} ms', flush=True)
        except Exception as e:  # noqa: BLE001
            print('halfboot microbench failed:', repr(e), flush=True)
    try:
        s2c_in = int(kit.s2c_start)
        for batch in batches:
            zs = [kit.complex_pack(at_limbs(fresh(1), s2c_in), at_limbs(fresh(1), s2c_in)) for _ in range(batch)]
            if batch == 1:
                ms = timeit(lambda: kit.to_coefficients(zs[0]), reps=max(3, args.reps // 2), warm=1)
            else:
                ms = timeit(lambda: kit.to_coefficients_many(zs), reps=max(3, args.reps // 2), warm=1)
            results['ops'][f'b{batch}_s2c_from_l{s2c_in}'] = {'s2c': ms}
            print(f'batch {batch:2d} s2c ({batch} complex-packed ct, {s2c_in} limbs): {ms:.1f} ms', flush=True)
    except Exception as e:  # noqa: BLE001
        print('s2c microbench failed:', repr(e), flush=True)

if args.scheme == 'holmes' and args.batch_bs:
    # E5: Holmes's dense program, several ciphertexts per bootstrap (bench/cmp/holmes_batch_bs.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ["HOLMES_BATCH_BS"] = "1"   # beta <= 5 BSGS kernels (holmes_sylph_kernel_src) for the dnum-5 tower
    import holmes_sylph_kernels, holmes_batch_bs  # noqa: E402
    print('sylph kernels:', {k: v for k, v in holmes_sylph_kernels.install().items() if k != 'modules'}, flush=True)
    print('batch bs:', holmes_batch_bs.install(), flush=True)
    lo = min(bs_from)
    ref_in = at_limbs(fresh(1), lo)
    ref_out = bs.bootstrap(ref_in, ctx, prog)
    def _dec(ct):
        v = client.decrypt(ct)
        v = v.cpu() if hasattr(v, 'cpu') else v
        return np.asarray(v, dtype=np.complex128).real
    ref_vals = _dec(ref_out)
    cmp_vals = _dec(ref_in)
    results['ops']['bootstrap_reference_err'] = {'max_abs_err': float(np.max(np.abs(ref_vals - cmp_vals)))}
    for batch in [int(v) for v in args.batch_bs.split(',')]:
        items = [ref_in] + [at_limbs(fresh(1), lo) for _ in range(batch - 1)]
        try:
            outs = holmes_batch_bs.bootstrap_batch(items, ctx, prog)
            got = _dec(outs[0])
            err = float(np.max(np.abs(got - ref_vals)))
            ms = timeit(lambda: holmes_batch_bs.bootstrap_batch(items, ctx, prog), reps=max(3, args.reps // 2), warm=1)
            results['ops'][f'b{batch}_batch_bootstrap_from_l{lo}'] = {'batch_bootstrap': ms, 'per_ct': round(ms / batch, 3),
                                                                     'max_abs_err_vs_single': err}
            print(f'batch {batch:2d} batched bootstrap from {lo} limbs: {ms:.1f} ms total, {ms / batch:.1f} ms/ct, '
                  f'max |batched - single| = {err:.2e} (single BS err vs input {results["ops"]["bootstrap_reference_err"]["max_abs_err"]:.2e})',
                  flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f'batched bootstrap (batch {batch}) failed:', repr(e), flush=True)

out = args.out or f"micro_{args.scheme}{'_native' if args.native else ''}.json"
json.dump(results, open(out, 'w'), indent=1)
print('wrote', out)
