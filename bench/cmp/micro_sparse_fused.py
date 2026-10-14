"""E7: fused sparse scheduler (HOLMES_SPARSE_FUSED) vs the batched loop on the production value_delta / key_shear
routes, isolated (sim context, depth --depth, 2 / 4 inputs).  Checks the decrypted outputs agree and times both.

    HOLMES_BATCH_BS=1 python bench/cmp/micro_sparse_fused.py --depth 12 --reps 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HOLMES_SPARSE_NONEXT_BABY", "1")
os.environ.setdefault("HOLMES_SPARSE_GPU_MASKS", "1")
os.environ.setdefault("HOLMES_SPARSE_FUSED", "1")
os.environ.setdefault("HOLMES_BATCH_BS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, default=12)
ap.add_argument("--dnum", type=int, default=5)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--stage", default="value_delta,key_shear_lower")
ap.add_argument("--profile", action="store_true")
ap.add_argument("--out", default=None)
args = ap.parse_args()
if args.profile:
    os.environ["HOLMES_SPARSE_PROFILE"] = "1"

import numpy as np  # noqa: E402
import easyfhe as torch  # noqa: E402
from easyfhe import fhe  # noqa: E402

import holmes_sylph_kernels  # noqa: E402

print("sylph:", {k: v for k, v in holmes_sylph_kernels.install(hostcache=True).items() if k not in ("modules", "patched")})

from llama3fhe.backend import release_if_supported  # noqa: E402
from llama3fhe.layouts.attention import AttentionPairLayout  # noqa: E402
from llama3fhe.layouts.linear import LinearCarrierLayout  # noqa: E402
from llama3fhe.operators.attention import sparse_fused  # noqa: E402
from llama3fhe.operators.attention.codec import decrypt_complex_rows  # noqa: E402
from llama3fhe.operators.attention.prefill.input import (  # noqa: E402
    _apply_sparse_routes_batched,
    _key_shear_routes,
    _layout_parameters,
    _value_delta_routes,
    choose_sparse_diagonal_plan,
)
from llama3fhe.operators.attention.sparse_bsgs_scheduler import build_sparse_route_schedule  # noqa: E402

params = _layout_parameters(LinearCarrierLayout(seq_len=128, dimension=4096, slots=32768), AttentionPairLayout())
SLOTS = params["slots"]
stages = {}
for name in args.stage.split(","):
    if name == "value_delta":
        routes, inputs, padded = _value_delta_routes(params), 2, None
    elif name == "key_shear_lower":
        routes, inputs, padded = _key_shear_routes(params, upper=False), 4, 128
    elif name == "key_shear_upper":
        routes, inputs, padded = _key_shear_routes(params, upper=True), 4, 128
    else:
        raise SystemExit(name)
    plan = choose_sparse_diagonal_plan(tuple(routes), slots=SLOTS, max_baby_offsets=32)
    sched = build_sparse_route_schedule(diagonal_offsets=plan.diagonal_offsets, baby_offsets=plan.baby_offsets,
                                        modulus=int(plan.modulus), slots=SLOTS, batch_cap=32,
                                        padded_plaintext_rows=padded)
    stages[name] = (routes, plan, inputs, padded, sched)

rots = set()
for routes, plan, inputs, padded, sched in stages.values():
    rots |= {int(o) for o in sched.baby_offsets if int(o)}
    rots |= {int(g.giant_offset) for g in sched.groups if int(g.giant_offset)}
print(f"rotation keys: {len(rots)}", flush=True)

t0 = time.time()
from llama3fhe.backend import create_sim_context  # noqa: E402
from llama3fhe.config import Llama3CKKSConfig  # noqa: E402

cfg = Llama3CKKSConfig()
cfg.simulator.logN, cfg.simulator.maxLevelsRemaining, cfg.simulator.device = 16, args.depth, "cuda"
cfg.simulator.dnum = args.dnum
cc = create_sim_context(cfg.simulator, input_dim=32, block_out=8, rotations=tuple(sorted(rots)),
                        include_default_rotations=False)
ctx = cc.context
print(f"context {time.time() - t0:.1f}s L={ctx.L} alpha={ctx.alpha} K={ctx.K}", flush=True)

rng = np.random.default_rng(0)
results = {}
for name, (routes, plan, inputs, padded, sched) in stages.items():
    ciphers = [cc.encrypt(rng.uniform(-0.05, 0.05, SLOTS), level=0, slots=SLOTS) for _ in range(inputs)]
    torch.cuda.synchronize()

    def run(fused: bool, profile=None):
        sparse_fused.ENABLED = fused
        return _apply_sparse_routes_batched(tuple(ciphers), routes=routes, plan=plan, slots=SLOTS, crypto_context=cc,
                                            function_prefix=f"micro.{name}", padded_plaintext_rows=padded,
                                            batch_cap=32, profile=profile)

    ref = run(False)
    out = run(True)
    torch.cuda.synchronize()
    a = decrypt_complex_rows(ref, crypto_context=cc)
    b = decrypt_complex_rows(out, crypto_context=cc)
    err = max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))
    scale = max(float(np.max(np.abs(x))) for x in a)
    for v in ref + out:
        release_if_supported(v)
    row = {"inputs": inputs, "groups": len(sched.groups), "batches": len(sched.batches), "rows": sched.encoded_rows,
           "max_abs_diff": err, "ref_max_abs": scale, "fused_stats": dict(sparse_fused.STATS)}
    for fused in (False, True):
        times = []
        for _ in range(args.reps):
            prof = {} if args.profile else None
            torch.cuda.synchronize()
            t = time.perf_counter()
            o = run(fused, prof)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            for v in o:
                release_if_supported(v)
        row[f"{'fused' if fused else 'loop'}_ms"] = 1e3 * min(times)
        if args.profile and prof:
            row[f"{'fused' if fused else 'loop'}_phases"] = {k[6:]: round(1e3 * v, 2) for k, v in prof.items() if k.startswith("phase.")}
    results[name] = row
    print(name, json.dumps(row, default=str), flush=True)
    for c in ciphers:
        release_if_supported(c)

if args.out:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=str)
