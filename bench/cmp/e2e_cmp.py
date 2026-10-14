"""End-to-end Holmes vs Slot-Coeff table (E8).

usage: e2e_cmp.py --holmes NAME=path/to/holmes_run_dir ... --sc NAME=path/to/chain-xx-yy-diag.json ... [--md out.md]

Holmes: bench/t2_model.py output dir (summary.json: per-layer table with station walls / bs counts).
SC:   sylph_easy/chain/run_chain.py --diagnose json (per-layer seconds + stage walls, cumulative b.counts).
Blocks: attention = Holmes rmsnorm_in+qkv+attn_input+attn_core+pairfm+w_o+residual_attn
                  = SC attn_norm+qkv+attention+o
        mlp       = Holmes rmsnorm_post+mlp+residual_mlp = SC ffn_norm+gate_up+silu+down
"""
import argparse, json, os, statistics

HOLMES_ATTN = ['rmsnorm_in', 'qkv', 'attn_input', 'attn_core', 'pairfm', 'w_o', 'residual_attn']
HOLMES_MLP = ['rmsnorm_post', 'mlp', 'residual_mlp']
SC_ATTN = ['attn_norm', 'qkv', 'attention', 'o']
SC_MLP = ['ffn_norm', 'gate_up', 'silu', 'down']


def load_holmes(path):
    s = json.load(open(os.path.join(path, 'summary.json')))
    rows = []
    for r in s['table']:
        st = r['stations']
        rows.append(dict(layer=r['layer'], wall=r['wall_med_s'], bs=r['bs'], bs_s=r['bs_s_med'],
                         attn=sum(st[k]['wall_med_s'] for k in HOLMES_ATTN if k in st),
                         mlp=sum(st[k]['wall_med_s'] for k in HOLMES_MLP if k in st),
                         stations={k: v['wall_med_s'] for k, v in st.items()}))
    err = s['final_hidden_error']
    return dict(rows=rows, steady=s['layer_wall_steady_med_s'], first=s['layer_wall_first_s'],
                total=s['layer_wall_sum_s'], warm0=(s.get('warmup_layer0') or {}).get('layer_wall_max_s'),
                err=f"rel_max {err['rel_max_err']:.2e} (max_abs {err['max_abs_err']:.3f} vs oracle max {err['oracle_max_abs']:.1f})",
                bs_total=s['bootstrap_calls_total'])


def load_sc(path):
    d = json.load(open(path))
    rows = []
    for r in d['layers']:
        st = r['stages']
        rows.append(dict(layer=r['layer'], wall=r['seconds'], bs=None, bs_s=None,
                         attn=sum(st.get(k, 0) for k in SC_ATTN), mlp=sum(st.get(k, 0) for k in SC_MLP),
                         stations=st, rel=r.get('rel_error'), limbs=r.get('limbs')))
    steady = [r['wall'] for r in rows[1:]]
    return dict(rows=rows, steady=statistics.median(steady) if steady else None, first=rows[0]['wall'],
                total=sum(r['wall'] for r in rows), warm0=None,
                err=f"rel {rows[-1]['rel']:.2e} (vs plaintext chain, layer {rows[-1]['layer']} output)",
                bs_total=d['counts'].get('refresh'), counts=d['counts'])


def fmt(v, nd=3):
    return '-' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holmes', action='append', default=[])
    ap.add_argument('--sc', action='append', default=[])
    ap.add_argument('--md')
    a = ap.parse_args()
    runs = []
    for spec in a.holmes:
        n, p = spec.split('=', 1); runs.append((n, load_holmes(p)))
    for spec in a.sc:
        n, p = spec.split('=', 1); runs.append((n, load_sc(p)))
    names = [n for n, _ in runs]
    out = []
    out.append('## Per-layer wall (s)\n')
    out.append('| layer | ' + ' | '.join(names) + ' |')
    out.append('|---|' + '---|' * len(names))
    nl = max(len(r['rows']) for _, r in runs)
    for i in range(nl):
        out.append(f'| L{i} | ' + ' | '.join(fmt(r['rows'][i]['wall'], 2) if i < len(r['rows']) else '-' for _, r in runs) + ' |')
    out.append('| L0 (warmup pass, Holmes only) | ' + ' | '.join(fmt(r['warm0'], 2) for _, r in runs) + ' |')
    out.append('| steady median (L1..) | ' + ' | '.join(fmt(r['steady'], 3) for _, r in runs) + ' |')
    out.append('| sum L0..L5 | ' + ' | '.join(fmt(r['total'], 2) for _, r in runs) + ' |')
    out.append('| sum L1..L5 | ' + ' | '.join(fmt(sum(x['wall'] for x in r['rows'][1:]), 2) for _, r in runs) + ' |')
    out.append('| bootstraps (6 layers) | ' + ' | '.join(fmt(r['bs_total']) for _, r in runs) + ' |')
    out.append('| final error | ' + ' | '.join(r['err'] for _, r in runs) + ' |')
    out.append('\n## Block split, layer 5 (s)\n')
    out.append('| block | ' + ' | '.join(names) + ' |')
    out.append('|---|' + '---|' * len(names))
    for key, label in [('attn', 'attention block (norm+qkv+attn+o)'), ('mlp', 'mlp block (norm+gate_up+silu+down)'), ('wall', 'layer wall')]:
        out.append(f'| {label} | ' + ' | '.join(fmt(r['rows'][-1][key], 3) for _, r in runs) + ' |')
    out.append('\n## Stations, layer 5 (s)\n')
    for n, r in runs:
        st = r['rows'][-1]['stations']
        out.append(f'- {n}: ' + ', '.join(f'{k} {fmt(v, 3)}' for k, v in st.items()) +
                   (f"; bs {r['rows'][-1]['bs']} ({fmt(r['rows'][-1]['bs_s'], 3)} s)" if r['rows'][-1]['bs'] is not None else ''))
    txt = '\n'.join(out)
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
