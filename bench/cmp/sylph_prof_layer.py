"""Slot-Coeff (Sylph-on-EasyFHE) single-layer profile with kernel-level op counts per decoder stage.

Companion of ``bench/t2_model.py`` + ``OPCOUNT=1`` on the Holmes side; both use ``bench/cmp/opcount.py``.

usage: OPCOUNT=1 [OPCOUNT_TIME=1] python sylph_prof_layer.py LAYER [RUNS] [--out FILE]
Runs one warm-up forward, then RUNS instrumented forwards of chain layer LAYER (input from the
plaintext chain); writes per-stage wall, Sylph's own ``b.counts`` deltas and the kernel snapshot.
"""
import collections
import json
import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, os.environ.get('SYLPH_ROOT', './third_party/sylph-easy'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import easyfhe as torch  # noqa: E402

import opcount  # noqa: E402
from sylph_easy.chain.manifest import PLAIN_INPUTS, layers, load  # noqa: E402
from sylph_easy.layer.setup import create, SLOTS  # noqa: E402
from sylph_easy.layer import layout, attention as A  # noqa: E402
from sylph_easy.layer.decoder import DecoderLayer  # noqa: E402
from sylph_easy.easyfhe_ext import coeff as CO, rhombus as R, fused as FU, keyswitch, rotate, bsgs, evalmod, packing  # noqa: E402
from sylph_easy.easyfhe_ext.engine import Engine  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith('--')]
li = int(args[0]) if args else 5
runs = int(args[1]) if len(args) > 1 else 1
out_path = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else f'sylph_prof_layer{li:02d}.json'

boot = [int(v) for v in os.environ.get('SYLPH_BOOT', '3,3,10').split(',')]
dnum = int(os.environ.get('SYLPH_DNUM', '3'))
w, cal = load(layers()[1][li])
x = np.load(PLAIN_INPUTS)[f'layer{li:02d}']
client, b = create(level_budget=boot[:2], output_levels=boot[2], dnum=dnum)
print('boot', boot, 'dnum', dnum, 'L', b.ctx.L, 'floor', b.floor, flush=True)

# kernel wrappers go on *after* setup (keyswitch/evalmod.install patched the kernels module)
extra = [m for m in (keyswitch, rotate, bsgs, evalmod, packing) if m is not None]
try:
    from sylph_easy.easyfhe_ext import slotmac
    extra.append(slotmac)
except Exception:
    pass
opcount.install(extra_modules=extra)
opcount.wrap_method(FU.FusedMac, 'mac', 'fused.mac')          # Rhombus PCMM kernels (ctypes leaf)
opcount.wrap_region(CO.CoefficientKit, 'halfboot', 'halfboot')
opcount.wrap_region(CO.CoefficientKit, 'halfboots', 'halfboot')
opcount.wrap_region(CO.CoefficientKit, 'to_coefficients', 's2c')
opcount.wrap_region(CO.CoefficientKit, 'to_coefficients_many', 's2c')
opcount.wrap_region(R.Rhombus, 'project', 'pcmm')
opcount.wrap_region(R.Rhombus, 'inputs', 'pcmm_inputs')
opcount.wrap_region(A.Attention, 'softmax', 'softmax')
opcount.wrap_region(A.Attention, '_sv', 'sv')
opcount.wrap_region(A.Attention, '_expand', 'kv_expand')
opcount.wrap_region(DecoderLayer, 'rmsnorm', 'rmsnorm')
opcount.wrap_region(DecoderLayer, 'silu_gate', 'silu')
try:
    opcount.wrap_region(A, 'inverse_sqrt', 'inverse_sqrt')
except Exception:
    pass
# API-level (ciphertext-semantic) counters + refresh trace; installed after the region wrappers so the api
# record of a HalfBoot / S2C is charged to the operator calling it, not to its own region
opcount.install_api_sylph(Engine, CO.CoefficientKit, R.Rhombus)

layer = DecoderLayer(b, w, cal, progress=False)
ct = b.stack([client.encrypt(v, slots=SLOTS) for v in layout.pack(x)])

# stage hook: snapshot kernel counts and Sylph's slot-level counters at every decoder stage boundary
_orig_t = DecoderLayer._t
stage_ops, stage_counts, stage_wall, stage_trace = {}, {}, {}, {}
_prev_counts = [dict(b.counts)]


def _t(self, name, t0):
    torch.cuda.synchronize()
    wall = time.time() - t0
    r = _orig_t(self, name, t0)
    stage_wall[name] = stage_wall.get(name, 0.0) + wall
    stage_ops[name] = opcount.merge(stage_ops.get(name, {}), opcount.take())
    stage_trace.setdefault(name, []).extend(opcount.take_trace())
    cur = dict(b.counts)
    delta = {k: cur.get(k, 0) - _prev_counts[0].get(k, 0) for k in cur}
    _prev_counts[0] = cur
    sc = stage_counts.setdefault(name, {})
    for k, v in delta.items():
        sc[k] = sc.get(k, 0) + v
    return r


DecoderLayer._t = _t

# warm-up (compiles kernels, fills plaintext caches); discard its counts
torch.cuda.synchronize(); t0 = time.time()
layer.forward(ct); torch.cuda.synchronize()
print(f'warm-up forward {time.time() - t0:.2f}s', flush=True)
stage_ops.clear(); stage_counts.clear(); stage_wall.clear(); stage_trace.clear(); _prev_counts[0] = dict(b.counts)
opcount.take(); opcount.take_trace()

walls = []
for r in range(runs):
    torch.cuda.synchronize(); t0 = time.time()
    out = layer.forward(ct)
    torch.cuda.synchronize(); walls.append(time.time() - t0)
    print(f'run {r} layer wall {walls[-1]:.2f}s  stages ' + ' '.join(f'{k}={v:.2f}' for k, v in layer.times.items()), flush=True)
    layer.times = {}

report = dict(scheme='slot-coeff', layer=li, runs=runs, boot=boot, dnum=dnum, L=int(b.ctx.L), floor=int(b.floor),
              opcount_time=opcount.timing(), layer_wall_s=walls, layer_wall_med_s=float(np.median(walls)),
              stages={k: dict(wall_s=stage_wall[k] / runs, counts={c: v / runs for c, v in stage_counts[k].items()},
                              ops={reg: {kn: {f: (v[f] / runs) for f in v} for kn, v in ks.items()}
                                   for reg, ks in stage_ops[k].items()},
                              # refresh schedule of the *first* instrumented run (identical every run)
                              refresh_trace=stage_trace.get(k, [])[:len(stage_trace.get(k, [])) // runs])
                      for k in stage_wall})
with open(out_path, 'w') as f:
    json.dump(report, f, indent=1)
print('wrote', out_path)
for k in stage_wall:
    s = opcount.summarize(stage_ops[k])
    regs = s.pop('_regions')
    top = sorted(s.items(), key=lambda kv: -kv[1]['ms'] if opcount.timing() else -kv[1]['calls'])[:6]
    print(f'[{k:9s}] {stage_wall[k] / runs:6.2f}s  counts/run {{{", ".join(f"{c}: {v / runs:g}" for c, v in stage_counts[k].items() if v)}}}')
    print('           regions(ms):', {r: round(v / runs) for r, v in regs.items()})
    print('           top kernels:', [(n, int(v['calls'] / runs), round(v['ms'] / runs)) for n, v in top])
    api = {n[4:]: int(v['cts'] / runs) for n, v in s.items() if n.startswith('api.')}
    print('           api cts:', api)
