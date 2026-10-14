"""Hoisted lane/phase (attention input step 1) vs the sequential path vs a plaintext oracle, on a small GPU context.

The lane/phase transform of one FM carrier is sum_d rotate(ct, d) * mask_d over the 2*token_lanes-1 diagonals
(``_lane_phase_masks``); ``_apply_lane_phase_hoisted`` (LLAMA_FHE_LANE_PHASE_HOISTED=1, the default) computes it as one
fast-rotation batch + one grouped ext MAC + one ModDown instead of one full key switch per diagonal.  Prints the max
abs error of both paths against the oracle, the max abs difference between them and the device time of each.
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
ap = argparse.ArgumentParser()
ap.add_argument('--depth', type=int, default=12)
ap.add_argument('--limbs', type=int, default=10, help='input cur_limbs')
ap.add_argument('--sources', default='0,5,15')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--sylph', action='store_true', help='install the Sylph key-switch kernels (as the layer runs do)')
args = ap.parse_args()

import easyfhe as torch  # noqa: E402
from llama3fhe.backend import create_sim_context  # noqa: E402
from llama3fhe.config import SimulatorConfig  # noqa: E402
from llama3fhe.layouts.attention import AttentionPairLayout  # noqa: E402
from llama3fhe.layouts.linear import LinearCarrierLayout  # noqa: E402
from llama3fhe.approx.attention import rope_cos_sin  # noqa: E402
from llama3fhe.operators.attention.prefill import input as core  # noqa: E402

if args.sylph:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import holmes_sylph_kernels  # noqa: E402
    print('sylph kernels:', {k: v for k, v in holmes_sylph_kernels.install(hostcache=True).items() if k != 'modules'})

attention = AttentionPairLayout()
carrier = LinearCarrierLayout(seq_len=int(attention.seq_len), dimension=4096, slots=int(attention.slots))
params = core._layout_parameters(carrier, attention)
lanes = int(params['token_lanes'])
offsets = [o for o in range(-(lanes - 1), lanes) if o]
t0 = time.time()
context = create_sim_context(
    SimulatorConfig(logN=16, maxLevelsRemaining=args.depth, dnum=5, dcrtBits=59, firstMod=60, device='cuda'),
    input_dim=4096, block_out=4096, rotations=[core._canonical_rotation(o, slots=int(attention.slots)) for o in offsets],
    include_default_rotations=False)
L = int(context.context.L)
print(f'context L={L} keygen {time.time() - t0:.1f}s lanes={lanes} diagonals={2 * lanes - 1}', flush=True)
cos, sin = rope_cos_sin(seq_len=int(attention.seq_len), head_dim=int(attention.head_dim))
rng = np.random.default_rng(0)
slots = int(attention.slots)


def oracle(values, masks, sign):
    out = np.zeros(slots, dtype=np.complex128)
    for offset, mask in masks.items():
        out += np.roll(values, sign * offset) * mask
    return out


def run(fn, ct, source):
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    (out,) = fn((ct,), params=params, cos=cos, sin=sin, crypto_context=context, function_prefix='check', source_indices=(source,))
    e1.record(); torch.cuda.synchronize()
    return out, e0.elapsed_time(e1)


level = L - args.limbs
for source in (int(s) for s in args.sources.split(',')):
    values = rng.normal(scale=1.0, size=slots) + 1j * rng.normal(scale=1.0, size=slots)
    ct = context.encrypt(values, level=level, slots=slots)
    masks = core._lane_phase_masks(source=source, params=params, cos=cos, sin=sin)
    want = oracle(values, masks, -1)
    os.environ['LLAMA_FHE_LANE_PHASE_HOISTED'] = '0'
    seq, t_seq = run(core._apply_lane_phase, ct, source)
    os.environ['LLAMA_FHE_LANE_PHASE_HOISTED'] = '1'
    hoist, t_hoist = run(core._apply_lane_phase, ct, source)
    times_h = [run(core._apply_lane_phase, ct, source)[1] for _ in range(args.reps)]
    os.environ['LLAMA_FHE_LANE_PHASE_HOISTED'] = '0'
    times_s = [run(core._apply_lane_phase, ct, source)[1] for _ in range(args.reps)]
    d_seq, d_hoist = context.decrypt(seq)[:slots], context.decrypt(hoist)[:slots]
    if np.max(np.abs(d_seq - want)) > np.max(np.abs(d_seq - oracle(values, masks, +1))):
        want = oracle(values, masks, +1)
        print('  (oracle uses roll(+offset): homo_rotate(ct, d)[i] = ct[i - d])')
    print(f'source {source:2d}: limbs {args.limbs} -> {int(hoist.state.cur_limbs)}  seq err {np.max(np.abs(d_seq - want)):.3e}  '
          f'hoisted err {np.max(np.abs(d_hoist - want)):.3e}  |seq-hoisted| {np.max(np.abs(d_seq - d_hoist)):.3e}  '
          f'ms seq {statistics.median(times_s):.2f}  hoisted {statistics.median(times_h):.2f}', flush=True)
