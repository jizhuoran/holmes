"""E9 warm-vs-warm 32-layer table: Holmes (t2_model summary.json) vs Slot-Coeff (run_chain.py json with SYLPH_WARMUP=1).

usage: e9_table.py OUTDIR [REPS]   (OUTDIR holds holmes_best10_r{i}/summary.json and sc_warm_r{i}.json)
"""
import json, os, statistics, sys

out = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
PICK = [0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 31]
HOLMES_ATTN = ['rmsnorm_in', 'qkv', 'attn_input', 'attn_core', 'pairfm', 'w_o', 'residual_attn']
HOLMES_MLP = ['rmsnorm_post', 'mlp', 'residual_mlp']
SC_ATTN = ['attn_norm', 'qkv', 'attention', 'o']
SC_MLP = ['ffn_norm', 'gate_up', 'silu', 'down']


def holmes(i):
    s = json.load(open(f'{out}/holmes_best10_r{i}/summary.json'))
    w = {r['layer']: r['wall_med_s'] for r in s['table']}
    st = {r['layer']: {k: v['wall_med_s'] for k, v in r['stations'].items()} for r in s['table']}
    return dict(w=w, st=st, warm=s['warmup_layer0']['layer_wall_max_s'], proc=s['total_wall_s'],
                bs=s['bootstrap_calls_total'], keys=s.get('warm_keys_seconds_max'),
                tok=(s.get('logits') or {}), attn=lambda L: sum(st[L].get(k, 0) for k in HOLMES_ATTN),
                mlp=lambda L: sum(st[L].get(k, 0) for k in HOLMES_MLP))


def sc(i):
    s = json.load(open(f'{out}/sc_warm_r{i}.json'))
    w = {r['layer']: r['seconds'] for r in s['layers']}
    st = {r['layer']: r['stages'] for r in s['layers']}
    return dict(w=w, st=st, warm=s['warmup']['forward_seconds'], proc=s['chain_seconds'],
                bs=s['counts'].get('refresh'), load=sum(r['load_seconds'] for r in s['layers']),
                tok=s.get('terminal', {}).get('comparison', {}), paris=s.get('paris'),
                attn=lambda L: sum(st[L].get(k, 0) for k in SC_ATTN),
                mlp=lambda L: sum(st[L].get(k, 0) for k in SC_MLP))


O = [holmes(i) for i in range(1, reps + 1)]
S = [sc(i) for i in range(1, reps + 1)]
f = lambda v: f'{v:.2f}'
lines = []
hdr = '| Metric | ' + ' | '.join(f'Holmes r{i}' for i in range(1, reps + 1)) + ' | ' + ' | '.join(f'SC warm r{i}' for i in range(1, reps + 1)) + ' | Holmes/SC mean ratio |'
lines += [hdr, '|' + '---|' * (2 * reps + 2)]


def row(name, fo, fs, nd=2):
    vo = [fo(o) for o in O]; vs = [fs(s) for s in S]
    ratio = statistics.mean(vo) / statistics.mean(vs) if all(v is not None for v in vo + vs) else None
    g = lambda v: '-' if v is None else f'{v:.{nd}f}'
    lines.append(f'| {name} | ' + ' | '.join(g(v) for v in vo) + ' | ' + ' | '.join(g(v) for v in vs) + ' | ' + ('-' if ratio is None else f'{ratio:.3f}') + ' |')


row('warmup L0 forward (excluded) s', lambda o: o['warm'], lambda s: s['warm'])
for L in PICK:
    row(f'L{L} s', lambda o, L=L: o['w'][L], lambda s, L=L: s['w'][L])
row('L3–30 median s', lambda o: statistics.median(o['w'][i] for i in range(3, 31)), lambda s: statistics.median(s['w'][i] for i in range(3, 31)))
row('sum L1–31 s', lambda o: sum(o['w'][i] for i in range(1, 32)), lambda s: sum(s['w'][i] for i in range(1, 32)), 1)
row('sum L0–31 s (32-layer forward)', lambda o: sum(o['w'].values()), lambda s: sum(s['w'].values()), 1)
row('L5 attention block s', lambda o: o['attn'](5), lambda s: s['attn'](5))
row('L5 MLP block s', lambda o: o['mlp'](5), lambda s: s['mlp'](5))
row('BS count (Holmes full / SC half-boot)', lambda o: float(o['bs']), lambda s: float(s['bs']), 0)
row('Process wall s (including setup/loading)', lambda o: o['proc'], lambda s: s['proc'], 1)
lines.append('')
lines.append('token: Holmes ' + ', '.join(f"r{i+1} argmax={o['tok'].get('argmax_match')} top5={o['tok'].get('top5_overlap')}/5" for i, o in enumerate(O))
             + '; SC ' + ', '.join(f"r{i+1} paris={s['paris']} top_k_match={s['tok'].get('top_k_match')}" for i, s in enumerate(S)))
lines.append('SC process wall includes weight-loading wait: ' + ', '.join(f"r{i+1} {s['load']:.1f}s" for i, s in enumerate(S))
             + ' (Holmes setup: keygen+warm keys ' + ', '.join(f"r{i+1} {o['keys']:.0f}s" for i, o in enumerate(O)) + ', excluded from forward sums)')
lines.append('')
# L5 station breakdown
lines.append('| L5 stage | ' + ' | '.join(f'Holmes r{i}' for i in range(1, reps + 1)) + ' | SC stage | ' + ' | '.join(f'SC r{i}' for i in range(1, reps + 1)) + ' |')
lines.append('|' + '---|' * (2 * reps + 2))
ok = list(O[0]['st'][5].keys()); sk = list(S[0]['st'][5].keys())
for j in range(max(len(ok), len(sk))):
    a = ok[j] if j < len(ok) else ''; b = sk[j] if j < len(sk) else ''
    lines.append(f'| {a} | ' + ' | '.join(f(o['st'][5][a]) if a else '' for o in O) + f' | {b} | ' + ' | '.join(f(s['st'][5][b]) if b else '' for s in S) + ' |')
print('\n'.join(lines))
