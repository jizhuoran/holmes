"""Stage-aligned comparison tables: Holmes (bench/t2_model.py + OPCOUNT) vs Slot-Coeff (sylph_prof_layer.py).

usage: python report.py --holmes SUMMARY.json --sylph PROF.json [--holmes-layer 5] [--holmes-baseline SUMMARY.json]
        [--sylph-chain chain_layers.jsonl] [--md OUT.md]
Prints (and optionally writes as markdown):
  1. layer-level wall per transformer stage (both schemes, aligned by stage) + sub-stage breakdown
  2. per stage, split into main / refresh (Holmes bootstrap; SC S2C + HalfBoot) / pcmm parts:
     2a. API-level ciphertext-semantic op counts (api.* labels of opcount.py) against the static theory (theory.py)
     2b. kernel-class calls / cts / GPU-ms (kernel calls are not comparable across schemes: SC batches; cts are)
     2c. refresh schedule (every bootstrap / S2C / HalfBoot call: region, cts, in -> out limbs)
  3. scheme-specific operators (bootstrap vs S2C/HalfBoot/PCMM/PairFM) [+ --sylph-native kernel ablation]
  4. --micro: atomic-op microbenchmark columns (micro_atomic.py JSONs)
Works with the pre-E0 JSONs too (no api.* / cts / refresh_trace: those columns stay empty).
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CATS = [
    ('KS.modup', ('F.cv_modup', 'keyswitch.modup')),
    ('KS.inner', ('F.cv_innerproduct_broadcast', 'F.cv_innerproduct_pairwise')),
    ('KS.moddown', ('F.cv_moddown', 'keyswitch.moddown_pair')),
    ('Rot', ('F.cv_hrot', 'F.cv_finalize_fast_rotation_q', 'F.cv_finalize_fast_rotation_ext',
             'F.cv_double_hoist_giant_sum_ext', 'keyswitch.rotate_pair', 'keyswitch.fast_rotate_ext',
             'keyswitch.giant_sum_ext', 'bsgs.baby_partial_sums', 'bsgs.giant_sum', 'bsgs.conversion',
             'bsgs.coeffs_to_slots', 'bsgs.slots_to_coeffs')),
    ('PTxCT MAC', ('F.cipher_grouped_pairwise_mac', 'F.cv_mul_pt_pair', 'F.cv_mul_pt_pair_', 'keyswitch.bsgs_mac_ext',
                   'slotmac.pt_mac', 'fused.mac')),
    # CT x CT with relinearisation: EasyFHE's fused hmul_relin_rescale, Sylph's fused version, and the tensor
    # product + key switch path (Holmes homo_mul_relin -> F.cv_keyswitch; Sylph Engine.mul -> keyswitch.keyswitch).
    # keyswitch.keyswitch / keyswitch.hmul_relin_rescale are containers: their modup / inner / moddown show up
    # under KS.* (self time), so compare the 'KS-bearing' sum, not this row alone, across schemes.
    ('CTxCT relin', ('F.cv_hmul_relin_rescale', 'keyswitch.hmul_relin_rescale', 'F.cv_keyswitch', 'keyswitch.keyswitch')),
    ('EvalMod', ('evalmod.eval_mod',)),
    ('packing', ('packing.butterfly', 'packing.monomial_ntt')),
    ('rescale', ('F.cv_rescale_one_level', 'keyswitch.rescale')),
    ('encode', ('F.cv_encode', 'F.cv_pre_encode_stage1')),
    ('modraise', ('F.cv_mod_raise',)),
    ('elementwise', ('F.cv_mul', 'F.cv_add', 'F.cv_add_pair', 'F.cv_sub_pair', 'F.cv_mul_scalar_pair', 'F.cv_mul_scalar_pair_',
                     'F.cv_add_scalar', 'F.cv_mul_by_monomial', 'F.cipher_grouped_scalar_weighted_acc',
                     'F.cv_sub', 'F.cv_neg', 'F.cv_add_scalar_pair', 'F.cv_sub_scalar')),
]
CONTAINERS = ('keyswitch.enabled', 'keyswitch.install', 'evalmod.install', 'evalmod.enabled', 'bsgs.enabled', 'packing.enabled',
              'slotmac.enabled', 'rotate.clear')  # pure flags; everything else keeps its self time under 'other'


def cat(name):
    for c, names in CATS:
        if name in names:
            return c
    if name.startswith(CONTAINERS):
        return None
    return 'other'


PARTS = ('main', 'refresh', 'pcmm')


def part_of(region):
    """main / refresh / pcmm part of a region path (Holmes: bootstrap; SC: halfboot, s2c, pcmm, pcmm_inputs)."""
    segs = region.split('/')
    if any(x in ('bootstrap', 'halfboot', 's2c') for x in segs):
        return 'refresh'
    if any(x in ('pcmm', 'pcmm_inputs') for x in segs):
        return 'pcmm'
    return 'main'


def by_cat(ops, part=None):
    """ops: {region: {kernel: {calls, cts, units, ms}}} -> {cat: {calls, cts, units, ms}} (kernel labels only;
    api.* labels are skipped).  ``part`` restricts to one of PARTS."""
    out = {}
    for reg, ks in ops.items():
        if part is not None and part_of(reg) != part:
            continue
        for k, v in ks.items():
            if k.startswith('api.'):
                continue
            c = cat(k)
            if c is None:
                continue
            s = out.setdefault(c, {'calls': 0, 'cts': 0, 'units': 0, 'ms': 0.0})
            for f in ('calls', 'cts', 'units', 'ms'):
                s[f] += v.get(f, 0)
    return out


API_LABELS = ['rot', 'ctxct', 'ctxct_norelin', 'relin', 'ptxct', 'bsgs_mac', 'bootstrap', 's2c', 'halfboot', 'pcmm',
              'pcmm_rot', 'refresh', 'poly', 'scalar', 'add', 'add_pt', 'rescale']


REFRESH_APIS = ('bootstrap', 's2c', 'halfboot', 'refresh')


def by_api(ops, part=None):
    """{label: {calls, cts}} of the api.* records (cts = ciphertext-semantic units)."""
    out = {}
    for reg, ks in ops.items():
        for k, v in ks.items():
            if not k.startswith('api.'):
                continue
            # refresh APIs (bootstrap / s2c / halfboot) are recorded in the *calling* region; file them under 'refresh'
            kpart = 'refresh' if k[4:] in REFRESH_APIS else part_of(reg)
            if part is not None and kpart != part:
                continue
            s = out.setdefault(k[4:], {'calls': 0, 'cts': 0})
            s['calls'] += v.get('calls', 0)
            s['cts'] += v.get('cts', v.get('units', 0))
    return out


def merge_api(a, b):
    for c, v in b.items():
        s = a.setdefault(c, {'calls': 0, 'cts': 0})
        for f in v:
            s[f] += v[f]
    return a


def by_region(ops, depth=1):
    out = {}
    for reg, ks in ops.items():
        key = '/'.join(reg.split('/')[:depth])
        out[key] = out.get(key, 0.0) + sum(v['ms'] for v in ks.values())
    return out


def merge_cat(a, b):
    for c, v in b.items():
        s = a.setdefault(c, {'calls': 0, 'cts': 0, 'units': 0, 'ms': 0.0})
        for f in v:
            s[f] = s.get(f, 0) + v[f]
    return a


# Holmes stations -> transformer stage; Sylph decoder stages -> transformer stage
HOLMES_STAGE = {'rmsnorm_in': 'RMSNorm(attn)', 'qkv': 'QKV proj', 'attn_input': 'QKV proj', 'attn_core': 'Attention core',
              'pairfm': 'Attn out repack', 'w_o': 'O proj', 'residual_attn': 'O proj', 'rmsnorm_post': 'RMSNorm(ffn)',
              'mlp': 'FFN', 'residual_mlp': 'FFN'}
SYLPH_STAGE = {'attn_norm': 'RMSNorm(attn)', 'qkv': 'QKV proj', 'attention': 'Attention core', 'o': 'O proj',
               'ffn_norm': 'RMSNorm(ffn)', 'gate_up': 'FFN', 'silu': 'FFN', 'down': 'FFN'}
STAGES = ['RMSNorm(attn)', 'QKV proj', 'Attention core', 'Attn out repack', 'O proj', 'RMSNorm(ffn)', 'FFN']


KS_BEARING = ('KS.modup', 'KS.inner', 'KS.moddown', 'Rot', 'CTxCT relin', 'EvalMod')
ZERO = {'calls': 0, 'cts': 0, 'units': 0, 'ms': 0.0}


def cat_rows(oc, sc, cats):
    rows = []
    for c in cats:
        a_ = oc.get(c, ZERO); b_ = sc.get(c, ZERO)
        if a_['calls'] == 0 and b_['calls'] == 0:
            continue
        rows.append([c, int(a_['calls']), int(a_.get('cts', 0)), int(a_['units']), a_['ms'] / 1e3,
                     int(b_['calls']), int(b_.get('cts', 0)), int(b_['units']), b_['ms'] / 1e3])
    ks = [r for r in rows if r[0] in KS_BEARING]
    rows.append(['= KS-bearing (modup+inner+moddown+Rot+CTxCT+EvalMod)', '', '', '', sum(r[4] for r in ks), '', '', '', sum(r[8] for r in ks)])
    body = rows[:-1]
    rows.append(['= all', sum(r[1] for r in body), sum(r[2] for r in body), sum(r[3] for r in body), sum(r[4] for r in body),
                 sum(r[5] for r in body), sum(r[6] for r in body), sum(r[7] for r in body), sum(r[8] for r in body)])
    return rows


KCOLS = ['kernel class', 'Holmes calls', 'Holmes cts', 'Holmes vol', 'Holmes s', 'SC calls', 'SC cts', 'SC vol', 'SC s']


def api_rows(o_parts, s_parts, th_o, th_s):
    """Rows: api op x (Holmes main / refresh | SC main / refresh / pcmm | theory Holmes / SC)."""
    rows = []
    labels = [l for l in API_LABELS if any(l in p for p in list(o_parts.values()) + list(s_parts.values()))
              or l in th_o or l in th_s]
    for l in labels:
        def g(parts, part):
            v = parts.get(part, {}).get(l)
            return '' if v is None else f"{int(v['cts'])} ({int(v['calls'])})"
        rows.append([l, g(o_parts, 'main'), g(o_parts, 'refresh'), g(s_parts, 'main'), g(s_parts, 'refresh'), g(s_parts, 'pcmm'),
                     th_o.get(l, ''), th_s.get(l, '')])
    return rows


def fmt(v, nd=2):
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def table(headers, rows):
    w = [max(len(str(h)), *(len(fmt(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: '| ' + ' | '.join(fmt(c).rjust(w[i]) if i else fmt(c).ljust(w[i]) for i, c in enumerate(cells)) + ' |'
    out = [line(headers), '|' + '|'.join('-' * (x + 2) for x in w) + '|'] + [line(r) for r in rows]
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holmes', required=True)
    ap.add_argument('--holmes-layer', type=int, default=5)
    ap.add_argument('--holmes-baseline', default=None, help='t2_model summary without OPCOUNT (for undisturbed walls)')
    ap.add_argument('--sylph', required=True)
    ap.add_argument('--sylph-chain', default=None, help='run_chain.py per-layer jsonl (undisturbed walls)')
    ap.add_argument('--micro', nargs='*', default=[], help='micro_atomic.py JSONs (holmes, sylph fused, sylph native)')
    ap.add_argument('--sylph-native', nargs='*', default=[], help='sylph_prof_layer.py JSONs run with SYLPH_*=0 (label=path)')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    od = json.load(open(a.holmes))
    rec = next(r for r in od['ranks'][0]['layer_records'] if r['layer'] == a.holmes_layer)
    sy = json.load(open(a.sylph))
    out = []
    P = out.append

    P(f"# Holmes vs Slot-Coeff, one steady-state Llama-3-8B layer (Holmes layer {a.holmes_layer}, Sylph layer {sy['layer']})\n")
    P(f"GPU: {od['ranks'][0].get('device_env', '?')} (Holmes device env) / Sylph context L={sy['L']} dnum={sy['dnum']} "
      f"BS budget {sy['boot'][:2]} output_levels {sy['boot'][2]}; Holmes context depth 37 dnum 5 BS (5,5) output_levels 17.")
    P("Kernel GPU-ms = sum of CUDA-event self times of the wrapped EasyFHE/Sylph kernels inside the stage (container time minus nested wrapped calls, so no double counting) (OPCOUNT_TIME=1 run; "
      "profiling adds CUDA-event overhead to wall time). 'wall' columns from the undisturbed baseline runs "
      "when given.\n")

    # ---- 1. stage walls
    o_base = None
    if a.holmes_baseline:
        ob = json.load(open(a.holmes_baseline))
        steady = [r for r in ob['table'] if r['layer'] >= 3]
        o_base = {st: statistics.median(r['stations'][st]['wall_med_s'] for r in steady) for st in steady[0]['stations']}
        o_base_bs = {st: statistics.median(r['stations'][st]['bs'] for r in steady) for st in steady[0]['stations']}
        o_layer_med = statistics.median(r['wall_med_s'] for r in steady)
    s_base = None
    if a.sylph_chain:
        rows = [json.loads(l) for l in open(a.sylph_chain)]
        steady = [r for r in rows if r['layer'] >= 3]
        s_base = {k: statistics.median(r['stages'][k] for r in steady) for k in steady[0]['stages']}
        s_layer_med = statistics.median(r['seconds'] for r in steady)
        s_chain = sum(r['seconds'] for r in rows), sum(r['load_seconds'] for r in rows)

    def agg(stage_map, items, key_wall, key_ops):
        res = {}
        for name, v in items.items():
            st = stage_map.get(name)
            if st is None:
                continue
            r = res.setdefault(st, {'wall': 0.0, 'ops': {}, 'parts': []})
            r['wall'] += key_wall(name, v)
            r['parts'].append(name)
            merge_cat(r['ops'], by_cat(key_ops(v)))
        return res

    o = agg(HOLMES_STAGE, rec['stations'], lambda n, v: (o_base[n] if o_base else v['wall_s']), lambda v: v.get('ops', {}))
    s = agg(SYLPH_STAGE, sy['stages'], lambda n, v: (s_base[n] if s_base else v['wall_s']), lambda v: v['ops'])
    o_bs = {}
    for n, v in rec['stations'].items():
        o_bs[HOLMES_STAGE[n]] = o_bs.get(HOLMES_STAGE[n], 0) + v['bootstrap_calls']
    s_hb = {}
    for n, v in sy['stages'].items():
        hb = sum(x['calls'] for reg, ks in v['ops'].items() if reg.split('/')[-1] == 'halfboot' for k, x in ks.items() if k == 'F.cv_mod_raise')
        s_hb[SYLPH_STAGE[n]] = s_hb.get(SYLPH_STAGE[n], 0) + hb / 2     # mod_raise runs once per cv component (2 per ct)

    P('## 1. Layer-level: wall per transformer stage (s)\n')
    rows = []
    for st in STAGES:
        ow = o.get(st, {}).get('wall', 0.0); sw = s.get(st, {}).get('wall', 0.0)
        okms = sum(v['ms'] for v in o.get(st, {}).get('ops', {}).values()) / 1e3
        skms = sum(v['ms'] for v in s.get(st, {}).get('ops', {}).values()) / 1e3
        rows.append([st, ow, okms, int(o_bs.get(st, 0)), sw, skms, int(s_hb.get(st, 0)),
                     ','.join(o.get(st, {}).get('parts', [])), ','.join(s.get(st, {}).get('parts', []))])
    tot_o = sum(r[1] for r in rows); tot_s = sum(r[4] for r in rows)
    rows.append(['layer total', tot_o, sum(r[2] for r in rows), sum(r[3] for r in rows), tot_s, sum(r[5] for r in rows),
                 sum(r[6] for r in rows), '', ''])
    P(table(['stage', 'Holmes wall', 'Holmes kernel-s', 'Holmes BS', 'SC wall', 'SC kernel-s', 'SC HalfBoot coeff-ct',
             'Holmes stations', 'SC stages'], rows))
    if o_base:
        P(f"\nOdin steady layer median (layers>=3, undisturbed run): {o_layer_med:.2f} s; model wall "
          f"{ob['model_wall_s']['max']:.1f} s / 32 layers; BS total {ob['bootstrap_calls_total']}.")
    if s_base:
        P(f"Slot-Coeff steady layer median (layers>=3, chain run): {s_layer_med:.2f} s; chain GPU sum {s_chain[0]:.1f} s + "
          f"host weight-prep waits {s_chain[1]:.1f} s (not overlapped) = {s_chain[0] + s_chain[1]:.1f} s / 32 layers.")
    P(f"Instrumented single-layer walls: Holmes layer {a.holmes_layer} {rec['layer_wall_s']:.2f} s (OPCOUNT_TIME on); "
      f"Slot-Coeff {sy['layer_wall_med_s']:.2f} s (warm caches, layer plaintexts already encoded).\n")

    # ---- 1b. sub-stages (Holmes: station stage_seconds wall; Sylph: region kernel-s)
    P('\n### 1b. Sub-stage breakdown (Holmes: station stage_seconds, wall s of the instrumented run; '
      'Slot-Coeff: GPU kernel-s per region of the instrumented run)\n')
    ost = rec['stations']
    def ss(st, k):
        return (ost.get(st, {}).get('stage_seconds') or {}).get(k, 0.0)
    orows = [
        ['RMSNorm(attn)', 'entry pair bootstrap (9 BS)', ss('rmsnorm_in', 'entry_pair_bootstrap')],
        ['RMSNorm(attn)', 'rmsnorm poly', ss('rmsnorm_in', 'rmsnorm')],
        ['QKV proj', 'paired_projection online (BSGS PT x CT)', ss('qkv', 'projection.paired_projection_online')],
        ['QKV proj', 'baby precompute (hoisted rotations)', ss('qkv', 'projection.paired_projection_baby_precompute')],
        ['QKV proj', 'weight pack_encode (online cv_encode)', ss('qkv', 'projection.paired_projection_pack_encode')],
        ['QKV proj', 'attn_input lane/query/key/value re-layout', sum(ss('attn_input', k) for k in ('lane', 'lane_isend', 'value_partial', 'query', 'key', 'exchange', 'value_finish'))],
        ['Attention core', 'QK^T', ss('attn_core', 'qk')],
        ['Attention core', 'softmax: pre-exp bootstrap', ss('attn_core', 'softmax.pre_exp_bootstrap')],
        ['Attention core', 'softmax: exp + alg2 rounds', ss('attn_core', 'softmax.exp') + ss('attn_core', 'softmax.alg2_round_1') + ss('attn_core', 'softmax.alg2_round_2') + ss('attn_core', 'softmax.shift_scale_mask')],
        ['Attention core', 'PV', ss('attn_core', 'pv')],
        ['Attn out repack', 'PairFM source (incl. refresh BS)', ss('pairfm', 'source')],
        ['O proj', 'online + baby + encode', ss('w_o', 'online') + ss('w_o', 'baby_precompute') + ss('w_o', 'pack_encode')],
        ['RMSNorm(ffn)', 'entry pair bootstrap (9 BS)', ss('rmsnorm_post', 'entry_pair_bootstrap')],
        ['RMSNorm(ffn)', 'rmsnorm poly', ss('rmsnorm_post', 'rmsnorm')],
        ['FFN', 'gate_up online', ss('mlp', 'gate_up_projection_online')],
        ['FFN', 'gate_up baby precompute + pack_encode', ss('mlp', 'gate_up_projection_baby_precompute') + ss('mlp', 'gate_up_projection_pack_encode')],
        ['FFN', 'SiLU gated', ss('mlp', 'silu_gated')],
        ['FFN', 'down online', ss('mlp', 'down_projection_online')],
        ['FFN', 'down baby precompute + pack_encode', ss('mlp', 'down_projection_baby_precompute') + ss('mlp', 'down_projection_pack_encode')],
    ]
    P('Holmes:\n')
    P(table(['stage', 'sub-stage', 'wall s'], orows))
    srows = []
    for n, v in sy['stages'].items():
        regs = {}
        for reg, ks in v['ops'].items():
            regs[reg] = regs.get(reg, 0.0) + sum(x['ms'] for x in ks.values())
        for reg, ms in regs.items():
            srows.append([SYLPH_STAGE[n], f'{n}: {reg}', ms / 1e3])
    P('\nSlot-Coeff (region "-" = the stage body outside the named operators; halfboot = C2S + conj + 2 x EvalMod on a coefficient ct):\n')
    P(table(['stage', 'decoder stage: region', 'kernel s'], srows))
    hb = {}
    for n, v in sy['stages'].items():
        for reg, ks in v['ops'].items():
            if reg.split('/')[-1] == 'halfboot':
                e = hb.setdefault(n, [0, 0.0])
                e[0] += ks.get('F.cv_mod_raise', {}).get('calls', 0) / 2
                e[1] += sum(x['ms'] for x in ks.values())
    P('\nSlot-Coeff HalfBoot per stage: ' + ', '.join(f'{n}: {int(c)} coeff-ct in -> {c * 2:.0f} slot-ct out, {ms / 1e3:.2f} s ({ms / max(c, 1):.1f} ms per input ct)' for n, (c, ms) in hb.items()))
    P(f"Holmes bootstrap per stage: " + ', '.join(f"{n}: {v['bootstrap_calls']} calls {v['bootstrap_seconds']:.2f} s" for n, v in rec['stations'].items() if v['bootstrap_calls']))
    P('')

    # ---- 2. per stage: API-level counts vs theory, kernel classes by part, refresh schedule
    try:
        import theory
        th_o, th_s = theory.holmes(), theory.sylph()
    except Exception as e:  # noqa: BLE001
        th_o, th_s = {'_error': repr(e)}, {}

    def parts_of(stage_map, items, key_ops, fn):
        res = {}
        for name, v in items.items():
            st = stage_map.get(name)
            if st is None:
                continue
            for part in PARTS:
                res.setdefault(st, {}).setdefault(part, {})
                (merge_api if fn is by_api else merge_cat)(res[st][part], fn(key_ops(v), part))
        return res

    o_api = parts_of(HOLMES_STAGE, rec['stations'], lambda v: v.get('ops', {}), by_api)
    s_api = parts_of(SYLPH_STAGE, sy['stages'], lambda v: v['ops'], by_api)
    o_kc = parts_of(HOLMES_STAGE, rec['stations'], lambda v: v.get('ops', {}), by_cat)
    s_kc = parts_of(SYLPH_STAGE, sy['stages'], lambda v: v['ops'], by_cat)
    have_api = any(v for st in o_api.values() for v in st.values()) or any(v for st in s_api.values() for v in st.values())

    P('## 2. Per stage: main / refresh / pcmm parts\n')
    P('Parts by region path: `refresh` = Holmes `bootstrap` region, SC `s2c` + `halfboot` regions (incl. the ones inside '
      'softmax / inverse_sqrt / rmsnorm / silu); `pcmm` = SC `pcmm_inputs` + `pcmm` regions; `main` = everything else. '
      'Holmes RMSNorm stations include the entry pair bootstrap of the residual stream (refresh part); SC projections '
      'include S2C + HalfBoot (refresh part), so the main parts are the like-for-like operator bodies.\n')
    P('API-level cells are `cts (calls)`: cts = ciphertext-semantic units (one ciphertext rotated by one offset / one '
      'ct x ct product / one ct x pt product / one ct refreshed); calls = scheme-level API calls (SC batches 16 cts per '
      'call, Holmes BSGS runs batch 1, so only cts compare).  Theory = static audit (Holmes, `*_complexity` defaults) / '
      'layout formula (SC); polynomial bodies (rsqrt, exp, SiLU) are not expanded there (`poly` rows give the count of '
      'polynomial evaluations; their ctxct are inside the measured main part).\n')
    cats = [c for c, _ in CATS] + ['other']
    for st in STAGES:
        oc_all = o_kc.get(st, {}); sc_all = s_kc.get(st, {})
        if not any(oc_all.values()) and not any(sc_all.values()):
            continue
        P(f'### {st}\n')
        if have_api:
            P('API-level (cts (calls)):\n')
            P(table(['api op', 'Holmes main', 'Holmes refresh', 'SC main', 'SC refresh', 'SC pcmm', 'theory Holmes', 'theory SC'],
                    api_rows(o_api.get(st, {}), s_api.get(st, {}), {k: v for k, v in th_o.get(st, {}).items() if k != 'notes'},
                             {k: v for k, v in th_s.get(st, {}).items() if k != 'notes'})))
            P('')
        for part in PARTS:
            oc = oc_all.get(part, {}); sc = sc_all.get(part, {})
            if not oc and not sc:
                continue
            P(f'Kernel classes, **{part}** part:\n')
            P(table(KCOLS, cat_rows(oc, sc, cats)))
            P('')
        notes = [f"theory Holmes: {th_o[st]['notes']}" for _ in [0] if st in th_o and 'notes' in th_o[st]]
        notes += [f"theory SC: {th_s[st]['notes']}" for _ in [0] if st in th_s and 'notes' in th_s[st]]
        for n_ in notes:
            P(f'- {n_}')
        P('')
    if '_error' in th_o:
        P(f"(Holmes theory unavailable: {th_o['_error']})\n")

    # ---- 2c. refresh schedule
    o_tr = [(n, t) for n, v in rec['stations'].items() for t in v.get('refresh_trace', [])]
    s_tr = [(n, t) for n, v in sy['stages'].items() for t in v.get('refresh_trace', [])]
    if o_tr or s_tr:
        P('### 2c. Refresh schedule (every bootstrap / S2C / HalfBoot call; cts = ciphertexts in; limbs in -> out)\n')
        for label, tr in (('Holmes', o_tr), ('Slot-Coeff', s_tr)):
            if not tr:
                continue
            rows = [[n, t['op'], t['region'], int(t['cts']), t.get('in_limbs', ''), t.get('out_limbs', '')] for n, t in tr]
            P(f'{label}:\n')
            P(table(['station/stage', 'op', 'region', 'cts', 'in limbs', 'out limbs'], rows))
            agg = {}
            for n, t in tr:
                a_ = agg.setdefault(t['op'], [0, 0]); a_[0] += 1; a_[1] += int(t['cts'])
            P('\n' + ', '.join(f'{k}: {v[0]} calls / {v[1]} cts' for k, v in agg.items()) + '\n')

    # ---- 3. whole layer by category and by scheme-specific operator region
    P('## 3. Whole layer\n')
    oc, sc = {}, {}
    for st in STAGES:
        merge_cat(oc, o.get(st, {}).get('ops', {})); merge_cat(sc, s.get(st, {}).get('ops', {}))
    P(table(KCOLS, cat_rows(oc, sc, cats)))
    if have_api:
        P('\nAPI-level, whole layer (cts (calls)); theory columns are the sums of the audited stages only:\n')
        oa, sa = {}, {}
        for st in STAGES:
            for part in PARTS:
                merge_api(oa.setdefault(part, {}), o_api.get(st, {}).get(part, {}))
                merge_api(sa.setdefault(part, {}), s_api.get(st, {}).get(part, {}))
        tho = {}; ths = {}
        for st in STAGES:
            for k, v in th_o.get(st, {}).items():
                if k != 'notes' and k != '_error':
                    tho[k] = tho.get(k, 0) + v
            for k, v in th_s.get(st, {}).items():
                if k != 'notes':
                    ths[k] = ths.get(k, 0) + v
        P(table(['api op', 'Holmes main', 'Holmes refresh', 'SC main', 'SC refresh', 'SC pcmm', 'theory Holmes', 'theory SC'],
                api_rows(oa, sa, tho, ths)))
    P('\n### Scheme-specific operators (GPU s, leaf kernels inside the region)\n')
    oreg, sreg = {}, {}
    for n, v in rec['stations'].items():
        for k, ms in by_region(v.get('ops', {}), depth=1).items():
            key = ('bootstrap' if k == 'bootstrap' else f'{n}(main)')
            oreg[key] = oreg.get(key, 0.0) + ms
    for n, v in sy['stages'].items():
        for reg, ks in v['ops'].items():
            leaf = reg.split('/')[-1] if reg != '-' else f'{n}(main)'
            sreg[leaf] = sreg.get(leaf, 0.0) + sum(x['ms'] for x in ks.values())
    rows = [[k, v / 1e3] for k, v in sorted(oreg.items(), key=lambda kv: -kv[1])]
    P('Holmes: ' + ', '.join(f'{k} {v:.2f}' for k, v in rows))
    rows = [[k, v / 1e3] for k, v in sorted(sreg.items(), key=lambda kv: -kv[1])]
    P('Slot-Coeff: ' + ', '.join(f'{k} {v:.2f}' for k, v in rows))
    P('\nSlot-Coeff slot-level op counters (engine b.counts, per layer): ' +
      json.dumps({k: round(sum(v['counts'].get(k, 0) for v in sy['stages'].values())) for k in ('mul', 'cmul', 'rotate', 'rotate_hoisted', 'poly', 'refresh', 'inverse_sqrt')}))
    if a.sylph_native:
        variants = [('fused (default)', sy)]
        for item in a.sylph_native:
            label, _, path = item.rpartition('=')
            variants.append((label or path, json.load(open(path))))
        P('\n### Slot-Coeff: contribution of its hand-written CUDA kernels (same layer, warm; wall s per stage)\n')
        hdr = ['stage'] + [v[0] for v in variants]
        rows = []
        for n in sy['stages']:
            rows.append([n] + [v[1]['stages'][n]['wall_s'] for v in variants])
        rows.append(['layer'] + [v[1]['layer_wall_med_s'] for v in variants])
        P(table(hdr, rows))
        P('')

    if a.micro:
        P('\n## 4. Atomic-operation microbenchmark (ms per call, CUDA events, median of 10; N=2^16, 2^15 slots)\n')
        mics = [json.load(open(f)) for f in a.micro]
        labels = [m['label'] for m in mics]
        P('Columns: ' + ' | '.join(labels) + f"; GPU {mics[0]['gpu']}\n")
        keys = []
        for m in mics:
            for k, row in m['ops'].items():
                for op in row:
                    if (k, op) not in keys:
                        keys.append((k, op))
        rows = []
        for k, op in keys:
            vals = [m['ops'].get(k, {}).get(op) for m in mics]
            r = [f'{k} {op}'] + [('' if v is None else v) for v in vals]
            if vals[0] and all(v is not None for v in vals):
                r.append(' / '.join(f'{v / vals[0]:.2f}' for v in vals[1:]))
            else:
                r.append('')
            rows.append(r)
        P(table(['op (b=batch, l=limbs)'] + [f'col{i + 1}' for i in range(len(mics))] + ['ratio vs col1'], rows))
        P('')
    text = '\n'.join(out)
    print(text)
    if a.md:
        open(a.md, 'w').write(text + '\n')


if __name__ == '__main__':
    main()
