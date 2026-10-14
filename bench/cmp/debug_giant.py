"""debug: Sylph rotate_pair (batched, strided) vs EasyFHE homo_rotate_add on Holmes's context."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import easyfhe as torch
from easyfhe import fhe
from easyfhe.fhe.ops import rotation
import holmes_multi_mac, holmes_sylph_kernels
print(holmes_sylph_kernels.install(hostcache=True)['keyswitch'])
from sylph_easy.easyfhe_ext import keyswitch as ks

N = 1 << 16; SLOTS = N // 2
off = 64 * 16
client, ctx = fhe.generate_client_context(fhe.CKKSContextSpec(
    depth=37, log_n=16, dnum=5, dcrt_bits=59, first_mod=60, secret_key_dist='SPARSE_TERNARY',
    scale_mode='fixed', rescale_policy='manual', rotations=(off,), auto_load_keys=True), device='cuda')
limbs = 6
def at_limbs(ct):
    out = fhe.align_to(ct, ct.state.replace(cur_limbs=limbs, scale_degree=1), ctx)
    return out.cipher_like([x[..., :limbs, :].contiguous() for x in out.cv], batch_size=out.batch_size)
cts = [at_limbs(client.encrypt(np.random.uniform(-1, 1, SLOTS), slots=SLOTS)) for _ in range(4)]
adds = [at_limbs(client.encrypt(np.random.uniform(-1, 1, SLOTS), slots=SLOTS)) for _ in range(4)]
def d2(t): return t if t.dim() == 2 else t[0]
def diff(a, b): return holmes_multi_mac.count_diff(a, b)

ref = [rotation.homo_rotate_add(c, off, ctx) for c in cts]
ref_add = [rotation.homo_rotate_add(c, off, ctx, addend=a) for c, a in zip(cts, adds)]
# 1. batch 1 contiguous, no P1
for p1 in (False, True):
    os.environ['SYLPH_P1'] = '1' if p1 else '0'
    c0 = d2(cts[0].cv[0]).unsqueeze(0); c1 = d2(cts[0].cv[1]).unsqueeze(0)
    o0, o1 = holmes_multi_mac._rotate_batch(ks, c0, c1, off, limbs, ctx)
    print('single p1', p1, 'mismatch', diff(o0[0], d2(ref[0].cv[0])), diff(o1[0], d2(ref[0].cv[1])))
    # 2. batch 4 contiguous stacked
    c0 = torch.stack([d2(c.cv[0]) for c in cts]); c1 = torch.stack([d2(c.cv[1]) for c in cts])
    o0, o1 = holmes_multi_mac._rotate_batch(ks, c0, c1, off, limbs, ctx)
    print('batch4 p1', p1, 'mismatch', [diff(o0[i], d2(ref[i].cv[0])) + diff(o1[i], d2(ref[i].cv[1])) for i in range(4)])
    # 3. batch 4 with add
    a0 = torch.stack([d2(c.cv[0]) for c in adds]); a1 = torch.stack([d2(c.cv[1]) for c in adds])
    o0, o1 = holmes_multi_mac._rotate_batch(ks, c0, c1, off, limbs, ctx, a0, a1)
    print('batch4+add p1', p1, 'mismatch', [diff(o0[i], d2(ref_add[i].cv[0])) + diff(o1[i], d2(ref_add[i].cv[1])) for i in range(4)])
    # 4. strided views: res [4][2][2][limbs][N], item g=1
    res = torch.empty((2, 2, 4, limbs, N), dtype=torch.uint64, device='cuda')
    for i in range(4):
        res[0, 1, i].copy_(d2(cts[i].cv[0])); res[1, 1, i].copy_(d2(cts[i].cv[1]))
        res[0, 0, i].copy_(d2(adds[i].cv[0])); res[1, 0, i].copy_(d2(adds[i].cv[1]))
    o0, o1 = holmes_multi_mac._rotate_batch(ks, res[0, 1], res[1, 1], off, limbs, ctx, res[0, 0], res[1, 0])
    print('strided+add p1', p1, 'mismatch', [diff(o0[i], d2(ref_add[i].cv[0])) + diff(o1[i], d2(ref_add[i].cv[1])) for i in range(4)])
    # 5. the EasyFHE batched path (_batch_same_rotate) as a sanity reference
    bat = cts[0].cipher_like([c0, c1], batch_size=4)
    rb = rotation._batch_same_rotate(bat, off, ctx)
    print('easyfhe batch4 mismatch', [diff(rb.cv[0][i], d2(ref[i].cv[0])) + diff(rb.cv[1][i], d2(ref[i].cv[1])) for i in range(4)])

# decode-level comparison
os.environ['SYLPH_P1'] = '1'
def dec(c):
    v = client.decrypt(c)
    return np.asarray(v.cpu().numpy() if hasattr(v, 'cpu') else v)
plain = [dec(c) for c in cts]
c0 = torch.stack([d2(c.cv[0]) for c in cts]); c1 = torch.stack([d2(c.cv[1]) for c in cts])
a0 = torch.stack([d2(c.cv[0]) for c in adds]); a1 = torch.stack([d2(c.cv[1]) for c in adds])
o0, o1 = holmes_multi_mac._rotate_batch(ks, c0, c1, off, limbs, ctx, a0, a1)
for i in range(4):
    got = dec(cts[i].cipher_like([o0[i], o1[i]], batch_size=1))
    want = dec(ref_add[i])
    exact = np.roll(np.real(plain[i]), -off) + np.real(dec(adds[i]))
    print(f'input {i}: |batched-ref|max {np.max(np.abs(np.real(got) - np.real(want))):.3e}  '
          f'|ref-exact| {np.max(np.abs(np.real(want) - exact)):.3e}  |batched-exact| {np.max(np.abs(np.real(got) - exact)):.3e}')
