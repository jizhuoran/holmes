"""E4: is the linear-layer gap an *algorithm* gap or a *limb-count* gap?

One 4096x4096 projection over 128 tokens, run by both schemes at a sweep of input limb counts:
  * ``--scheme holmes``   Holmes BSGS (``LinearOperator`` + ``LinearWeight``, production schedule baby 64 / anchor 64,
                        16 input ciphertexts of 4096 features x 8 token lanes = what one QKV/W_o block sees),
                        timed per ``apply`` with the baby cache cleared, so the profile keys ``online`` (PT x CT MAC
                        + giant rotations), ``baby_precompute`` (hoisted baby rotations), ``pack_encode`` (online
                        weight cv_encode) and ``rescale`` are all included and reported separately.
  * ``--scheme sylph``  Slot-Coeff Rhombus PCMM (``Rhombus.inputs`` automorphic copies + ``Rhombus.project``,
                        fused encode+NTT+MAC kernel, 8 coefficient blocks of 512 features x 128 tokens -> 8 packed
                        output blocks); at the production limb (2) the following HalfBoot is timed as well.
Inputs are fresh encryptions aligned to each limb count (content is irrelevant for timing; Holmes is checked against
the plaintext product at the first and last limb so the schedule really ran).

Hypothesis under test: cost per (PT x CT, limb) is of the same order in both schemes, so time ~ limb, and Holmes at
2 limbs lands near SC's PCMM; the layer-level gap then comes from *where* each pipeline places the projection
(Holmes: 8-13 limbs because downstream levels are consumed before the next bootstrap; SC: always 2 limbs because
S2C -> PCMM -> HalfBoot refreshes right after the projection).

usage: linear_limb_sweep.py --scheme holmes  [--limbs 2,3,4,6,8,10,13,16] [--inputs 16] [--max-rows 2048] [--out F]
       linear_limb_sweep.py --scheme sylph [--limbs 2,3,4,6,8,10,13,16] [--packs 12] [--out F]
       linear_limb_sweep.py --report holmes.json sylph.json [--bs-ms 57.7] [--md F]   (no GPU needed)
"""
import argparse
import json
import os
import statistics
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--scheme', choices=['holmes', 'sylph'])
ap.add_argument('--limbs', default='2,3,4,6,8,10,13,16')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--dim', type=int, default=4096)
ap.add_argument('--inputs', type=int, default=16, help='holmes: input ciphertexts (production QKV block: 8 complex + 8 conj = 16)')
ap.add_argument('--max-rows', type=int, default=2048, help='holmes: max_plaintext_rows (rtx6000pro-high preset 2048, A100 512)')
ap.add_argument('--baby', type=int, default=64)
ap.add_argument('--depth', type=int, default=37, help='holmes: context depth (production 37)')
ap.add_argument('--dnum', type=int, default=None, help='key-switch dnum (default holmes 5 / sylph 3)')
ap.add_argument('--packs', type=int, default=12, help='sylph: output blocks used for best_split (production QKV: 8+2+2)')
ap.add_argument('--no-halfboot', action='store_true')
ap.add_argument('--out', default=None)
ap.add_argument('--report', nargs='*', default=None, help='JSON files from the runs -> markdown tables')
ap.add_argument('--bs-ms', type=float, default=57.7, help='report: Holmes bootstrap ms per ct (micro_holmes, dnum 5)')
ap.add_argument('--halfboot-ms', type=float, default=20.7, help='report: SC HalfBoot ms per coeff ct (layer 5 avg)')
ap.add_argument('--md', default=None)
args = ap.parse_args()


# ---------------------------------------------------------------------------------------------------------- report
def report(files):
    runs = {}
    for f in files:
        d = json.load(open(f))
        runs[d['scheme']] = d
    lines = ['# E4 linear-layer limb sweep: one 4096x4096 projection over 128 tokens', '']
    for scheme in ('holmes', 'sylph'):
        d = runs.get(scheme)
        if not d:
            continue
        lines += [f"## {d['label']}  (GPU {d['gpu']}, {d['ptxct']} PT x CT per projection, {d['reps']} reps, median)", '']
        if scheme == 'holmes':
            lines += ['| limb | wall ms | online | baby | encode | rescale | ms/limb | us per PTxCT.limb | rel err |',
                      '|------|---------|--------|------|--------|---------|---------|-------------------|---------|']
            for r in d['rows']:
                if 'error' in r:
                    lines.append(f"| {r['limbs']} | error: {r['error'][:60]} | | | | | | | |")
                    continue
                err = r.get('rel_err'); err = f'{err:.1e}' if err is not None else ''
                lines.append(f"| {r['limbs']} | {r['wall_ms']:.0f} | {r['online_ms']:.0f} | {r['baby_ms']:.0f} | {r['encode_ms']:.0f} | "
                             f"{r['rescale_ms']:.0f} | {r['wall_ms'] / r['limbs']:.0f} | {1e3 * r['wall_ms'] / d['ptxct'] / r['limbs']:.3f} | {err} |")
        else:
            lines += ['| limb | wall ms | copies (A autos) | project (MAC+pack) | ms/limb | us per PTxCT.limb | HalfBoot ms (8 ct) |',
                      '|------|---------|------------------|--------------------|---------|-------------------|--------------------|']
            for r in d['rows']:
                if 'error' in r:
                    lines.append(f"| {r['limbs']} | error: {r['error'][:60]} | | | | | |")
                    continue
                hb = r.get('halfboot_ms'); hb = f'{hb:.0f}' if hb is not None else ''
                lines.append(f"| {r['limbs']} | {r['wall_ms']:.0f} | {r['copies_ms']:.0f} | {r['project_ms']:.0f} | {r['wall_ms'] / r['limbs']:.0f} | "
                             f"{1e3 * r['wall_ms'] / d['ptxct'] / r['limbs']:.3f} | {hb} |")
        lines.append('')
    if 'holmes' in runs and 'sylph' in runs:
        o = {r['limbs']: r for r in runs['holmes']['rows'] if 'error' not in r}
        s = {r['limbs']: r for r in runs['sylph']['rows'] if 'error' not in r}
        common = sorted(set(o) & set(s))
        lines += ['## Equal-limb comparison', '',
                  '| limb | Holmes ms | SC ms | Holmes/SC | Holmes us/(PTxCT.limb) | SC us/(PTxCT.limb) |',
                  '|------|---------|-------|---------|----------------------|--------------------|']
        for l in common:
            po = 1e3 * o[l]['wall_ms'] / runs['holmes']['ptxct'] / l
            ps = 1e3 * s[l]['wall_ms'] / runs['sylph']['ptxct'] / l
            lines.append(f"| {l} | {o[l]['wall_ms']:.0f} | {s[l]['wall_ms']:.0f} | {o[l]['wall_ms'] / s[l]['wall_ms']:.2f} | {po:.3f} | {ps:.3f} |")
        lines.append('')
        # where production actually runs: Holmes qkv 13 / mlp 8.3 / w_o 3 limbs, SC 2 limbs
        if 2 in o and 2 in s:
            lo2, ls2 = o[2]['wall_ms'], s[2]['wall_ms']
            hb = s[2].get('halfboot_ms') or args.halfboot_ms * runs['sylph']['blocks']
            lines += ['## Projection placed where each pipeline runs it', '',
                      f"- SC: PCMM at 2 limbs {ls2:.0f} ms + HalfBoot of {runs['sylph']['blocks']} output blocks {hb:.0f} ms = **{ls2 + hb:.0f} ms** per 4096x4096 block",
                      f"- Holmes at 2 limbs {lo2:.0f} ms (+ one bootstrap of {runs['holmes']['inputs']} cts to buy the levels back, "
                      f"{runs['holmes']['inputs'] * args.bs_ms:.0f} ms at {args.bs_ms} ms/ct) = **{lo2 + runs['holmes']['inputs'] * args.bs_ms:.0f} ms**"]
            for l, where in ((13, 'QKV station (avg 13 limbs)'), (8, 'MLP station (avg 8.3 limbs)'), (3, 'W_o (3 limbs)')):
                if l in o:
                    lines.append(f"- Holmes at {l} limbs, i.e. {where}: **{o[l]['wall_ms']:.0f} ms** = {o[l]['wall_ms'] / ls2:.1f}x SC PCMM, "
                                 f"{o[l]['wall_ms'] / (ls2 + hb):.1f}x SC PCMM+HalfBoot")
            lines.append('')
    text = '\n'.join(lines)
    print(text)
    if args.md:
        open(args.md, 'w').write(text + '\n')


if args.report is not None:
    report(args.report)
    sys.exit(0)
if not args.scheme:
    ap.error('--scheme or --report required')

import numpy as np  # noqa: E402
import easyfhe as torch  # noqa: E402
from easyfhe import fhe  # noqa: E402

LOGN, N = 16, 1 << 16
SLOTS = N // 2
limbs_list = [int(v) for v in args.limbs.split(',')]
rng = np.random.default_rng(20261013)


def med(fn, reps):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1e3)
    return statistics.median(out), r


t0 = time.time()
rows = []
if args.scheme == 'holmes':
    # the script lives in bench/cmp: put the checkout (not an installed llama3fhe) first on sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from llama3fhe.backend import create_sim_context
    from llama3fhe.config import SimulatorConfig
    from llama3fhe.operators.linear import LinearOperator, LinearOperatorConfig, LinearWeight, linear_rotations

    dnum = args.dnum or 5
    lanes = SLOTS // args.dim
    cfg = LinearOperatorConfig(baby_steps=args.baby, baby_anchor_step=args.baby, max_plaintext_rows=args.max_rows,
                               reuse_baby_rotations=True, hoist_strategy='normal', constant_cache_mode='none',
                               raw_weight_policy='device', encode_depth=0)
    context = create_sim_context(
        SimulatorConfig(logN=LOGN, maxLevelsRemaining=args.depth, dnum=dnum, dcrtBits=59, firstMod=60, device='cuda'),
        input_dim=args.dim, block_out=args.dim,
        rotations=linear_rotations(dimension=args.dim, token_lanes=lanes, operator_config=cfg),
        include_default_rotations=False)
    L = int(context.context.L)
    label = f'holmes BSGS (depth {args.depth}, dnum {dnum}, baby {args.baby}, max_rows {args.max_rows}, {args.inputs} cts x {args.dim} diag)'
    print(f'[{label}] L={L} keygen {time.time() - t0:.1f}s', flush=True)
    weight = rng.normal(scale=0.02, size=(args.dim, args.dim))
    values = [rng.normal(scale=1.0, size=(lanes, args.dim)) for _ in range(args.inputs)]
    operator = LinearOperator(crypto_context=context, token_lanes=lanes, config=cfg)
    prepared = LinearWeight(weight, dtype=np.float64, crypto_context=context)
    ptxct = args.inputs * args.dim
    for limbs in limbs_list:
        if limbs > L - 1:
            continue
        try:
            cts = [context.encrypt(v.T.reshape(-1), level=L - limbs, slots=SLOTS) for v in values]
            prof_acc = None
            walls = []
            outs = None

            def run(prev):
                del prev                             # free the previous outputs before the next schedule
                operator.clear_baby_cache()          # production computes the babies once per layer: include them
                return operator.apply(tuple(cts), prepared)

            outs, _ = run(None)                      # warm (weight upload, allocator)
            for _ in range(args.reps):
                torch.cuda.synchronize(); t = time.perf_counter()
                outs, prof = run(outs)
                torch.cuda.synchronize(); walls.append((time.perf_counter() - t) * 1e3)
                prof_acc = prof if prof_acc is None else {k: prof_acc[k] + v for k, v in prof.items() if k in prof_acc}
            r = {'limbs': limbs, 'wall_ms': statistics.median(walls), 'walls_ms': walls,
                 'online_ms': 1e3 * prof_acc['online'] / args.reps, 'baby_ms': 1e3 * prof_acc['baby_precompute'] / args.reps,
                 'encode_ms': 1e3 * prof_acc['pack_encode'] / args.reps, 'rescale_ms': 1e3 * prof_acc['rescale'] / args.reps,
                 'out_limbs': int(outs[0].state.cur_limbs)}
            if limbs in (limbs_list[0], limbs_list[-1]):
                got = np.asarray(context.decrypt(outs[0])).real.reshape(args.dim, lanes).T
                exp = values[0] @ weight.T
                r['rel_err'] = float(np.max(np.abs(got - exp)) / np.max(np.abs(exp)))
            print(f"limbs {limbs:2d}: wall {r['wall_ms']:.0f} ms  online {r['online_ms']:.0f}  baby {r['baby_ms']:.0f}  encode {r['encode_ms']:.0f}"
                  f"  rescale {r['rescale_ms']:.0f}  -> out limbs {r['out_limbs']}  {'rel_err %.1e' % r['rel_err'] if 'rel_err' in r else ''}", flush=True)
            del outs, cts
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f'limbs {limbs}: ERROR {e!r}', flush=True)
            r = {'limbs': limbs, 'error': repr(e)}
            torch.cuda.empty_cache()
        rows.append(r)
    operator.close(); prepared.close()
    meta = {'inputs': args.inputs, 'blocks': args.inputs, 'L': L, 'dnum': dnum, 'depth': args.depth, 'baby': args.baby, 'max_rows': args.max_rows}
else:
    sys.path.insert(0, os.environ.get('SYLPH_ROOT', './third_party/sylph-easy'))
    from easyfhe.fhe.ops import alignment
    from sylph_easy.layer.setup import create
    from sylph_easy.easyfhe_ext.rhombus import Rhombus
    from sylph_easy.easyfhe_ext import keyswitch

    dnum = args.dnum or 3
    client, eng = create(verbose=True, dnum=dnum)
    ctx, kit = eng.ctx, eng.kit
    L = int(ctx.L)
    rh = Rhombus(kit)
    blocks = args.dim // kit.k
    out_blocks = args.dim // kit.k
    split = Rhombus.best_split(rh.h, int(np.log2(kit.k)), blocks, args.packs)
    plan = rh.plan(split)
    ptxct = plan['pc'] * plan['a'] * blocks * out_blocks
    label = (f"sylph Rhombus PCMM (L={L}, dnum {dnum}, K={kit.k}, {blocks} coeff blocks -> {out_blocks} out blocks, split {split}, "
             f"A={plan['a']} pc={plan['pc']}, fused ks {keyswitch.enabled()})")
    print(f'[{label}] keygen {time.time() - t0:.1f}s', flush=True)
    dev_w = rh.upload(rng.normal(scale=0.02, size=(args.dim, args.dim)))

    def at_limbs(ct, limbs):
        if int(ct.state.cur_limbs) == limbs:
            return ct
        out = alignment.align_to(ct, ct.state.replace(cur_limbs=limbs, scale_degree=1), ctx)
        return out.cipher_like([x[..., :limbs, :].contiguous() for x in out.cv], batch_size=out.batch_size)

    base = [client.encrypt(rng.uniform(-1, 1, SLOTS), slots=SLOTS) for _ in range(blocks)]
    for limbs in limbs_list:
        if limbs > L - 1:
            continue
        try:
            cts = [at_limbs(c, limbs) for c in base]
            scale = kit.raise_adjustment(limbs) / 8.0        # production: raise correction / stage bound rides on the weights
            copies_ms, copies = med(lambda: rh.inputs(cts, split), args.reps)
            project_ms, packed = med(lambda: rh.project(copies, dev_w, split, scale=scale), args.reps)
            r = {'limbs': limbs, 'copies_ms': copies_ms, 'project_ms': project_ms, 'wall_ms': copies_ms + project_ms,
                 'out_limbs': int(packed[0].state.cur_limbs)}
            if not args.no_halfboot and int(packed[0].state.cur_limbs) == 1:
                hb_ms, _ = med(lambda: kit.halfboots(packed, folded=True), args.reps)
                r['halfboot_ms'] = hb_ms
            print(f"limbs {limbs:2d}: copies {copies_ms:.0f} ms  project {project_ms:.0f} ms  -> out limbs {r['out_limbs']}"
                  f"  {'halfboot %.0f ms' % r['halfboot_ms'] if 'halfboot_ms' in r else ''}", flush=True)
            del cts, copies, packed
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f'limbs {limbs}: ERROR {e!r}', flush=True)
            r = {'limbs': limbs, 'error': repr(e)}
            torch.cuda.empty_cache()
        rows.append(r)
    meta = {'blocks': blocks, 'out_blocks': out_blocks, 'split': split, 'A': plan['a'], 'pc': plan['pc'], 'L': L, 'dnum': dnum, 'packs': args.packs}

result = dict(scheme=args.scheme, label=label, gpu=torch.cuda.get_device_name(0), dim=args.dim, ptxct=ptxct, reps=args.reps,
              limbs=limbs_list, rows=rows, **meta)
if args.out:
    json.dump(result, open(args.out, 'w'), indent=1)
    print('wrote', args.out)
