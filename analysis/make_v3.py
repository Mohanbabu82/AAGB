# Tables for the additional experiments (aagb3.py): TabPFN and tuned deep baselines, TabNet-feature stacking
# audit, and the two larger datasets. Input: log files with checksummed @@ lines. Usage: python make_v3.py out logs...
import sys, os, re, zlib, json, math, itertools
import numpy as np
from scipy import stats

FMK = ['roc', 'pr', 'brier', 'ece', 'f15', 'f1t', 'senst', 'spect', 'thr', 'secs10']
NAMES = {'heart': 'Heart Disease', 'pima': 'Pima', 'ckd': 'CKD', 'stroke': 'Stroke', 'diab130': 'Diabetes 130-US',
         'cdc': 'CDC BRFSS'}
MOD = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'cat': 'CatBoost', 'mlp': 'MLP',
       'tabpfn': 'TabPFN v2', 'tabnet_t': 'TabNet (tuned)', 'ftt_t': 'FT-Transformer (tuned)', 'aagb_rm': 'AAGB-RM',
       'aagb_ig': 'AAGB-IG', 'aagb_ig_ins': 'AAGB-IG, in-sample attention'}
LEAK = {'L:tn_mask_ins': 'TabNet masks, in-sample', 'L:tn_mask_cf': 'TabNet masks, cross-fitted',
        'L:tn_emb_ins': 'TabNet embedding, in-sample', 'L:tn_emb_cf': 'TabNet embedding, cross-fitted'}
K = 5


def parse(paths):
    FM, LK, meta, bad, n = {}, {}, {}, 0, 0
    for p in paths:
        for line in open(p, encoding='utf-8', errors='ignore'):
            m = re.search(r'@@(\S+)\|([0-9a-f]{8})\s*$', line.strip())
            if not m:
                v = re.search(r'@@(VER|HASH|JOB|ALLDONE3|TIME)\|(.*)$', line.strip())
                if v:
                    meta.setdefault(v.group(1), []).append(v.group(2))
                continue
            body, crc = m.group(1), m.group(2)
            if '%08x' % (zlib.crc32(body.encode()) & 0xffffffff) != crc:
                bad += 1
                continue
            ds, blk, key, vals = body.split('|', 3)
            n += 1
            if blk in ('FM', 'LK'):
                v = [np.nan if x == 'x' else int(x) / 1e4 for x in vals.split(',')]
                mdl, r, f = key.rsplit(':', 2)
                (FM if blk == 'FM' else LK).setdefault(ds, {}).setdefault(mdl, {})[(int(r), int(f))] = v
            else:
                meta.setdefault(blk, {})[(ds, key)] = vals
    return FM, LK, meta, n, bad


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return float(np.mean(v)) if n else np.nan, np.nan, np.nan, n
    mu, sd = v.mean(), v.std(ddof=1)
    se = math.sqrt((1 / n + 1 / (K - 1)) * sd ** 2)
    hw = stats.t.ppf(0.975, n - 1) * se
    p = 2 * stats.t.sf(abs(mu) / se, n - 1) if se > 0 else (0.0 if mu != 0 else 1.0)
    return mu, hw, p, n


def paired(A, B, k):
    keys = sorted(set(A) & set(B))
    return np.array([A[x][k] - B[x][k] for x in keys]), len(keys)


def f3(x, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    s = ('%+.3f' if sign else '%.3f') % x
    s = s.replace('-', '−')
    return s.replace('+0.000', '0.000').replace('−0.000', '0.000')


def holm(ps):
    ps = np.asarray(ps)
    o = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


def main(out, paths):
    os.makedirs(os.path.join(out, 'tables'), exist_ok=True)
    FM, LK, meta, n, bad = parse(paths)
    print('records', n, 'bad', bad)
    idx = {k: i for i, k in enumerate(FMK)}
    NUM = {'records': n, 'bad': bad, 'meta': {k: v for k, v in meta.items() if isinstance(v, list)}}
    T = {}
    # ---- table: new baselines on the small datasets
    rows, fam = [], []
    for ds in ('heart', 'pima', 'ckd', 'stroke'):
        if ds not in FM:
            continue
        for m in ('xgb', 'aagb_ig', 'tabpfn', 'tabnet_t', 'ftt_t'):
            if m not in FM[ds]:
                continue
            d = FM[ds][m]
            st = {k: ci([v[idx[k]] for v in d.values()]) for k in ('roc', 'pr', 'brier', 'f1t')}
            NUM['%s.%s' % (ds, m)] = {k: dict(mean=st[k][0], hw=st[k][1], n=st[k][3]) for k in st}
            NUM['%s.%s' % (ds, m)]['secs'] = float(np.nanmean([v[idx['secs10']] * 1e3 for v in d.values()]))
            row = [NAMES[ds], MOD[m], '%d' % st['roc'][3]] + ['%s ± %s' % (f3(st[k][0]), f3(st[k][1])) for k in ('roc', 'pr', 'brier', 'f1t')]
            if m != 'aagb_ig' and 'aagb_ig' in FM[ds]:
                dl, nn = paired(FM[ds]['aagb_ig'], d, idx['roc'])
                mu, hw, p, _ = ci(dl)
                fam.append((ds, m, p))
                NUM['%s.aagb_ig_vs_%s.roc' % (ds, m)] = dict(mean=mu, lo=mu - hw, hi=mu + hw, p=p, n=nn)
                row.append('%s [%s, %s]' % (f3(mu, True), f3(mu - hw, True), f3(mu + hw, True)))
                row.append('PH%d' % (len(fam) - 1))
            else:
                row += ['—', '—']
            rows.append(row)
    adj = holm([p for _, _, p in fam]) if fam else []
    for r in rows:
        if r[-1].startswith('PH'):
            i = int(r[-1][2:])
            r[-1] = ('%.3f' % adj[i]) if adj[i] >= 0.001 else '<0.001'
            NUM['%s.aagb_ig_vs_%s.roc' % (fam[i][0], fam[i][1])]['p_holm'] = float(adj[i])
    if fam:
        NUM['small_min_holm'] = float(min(adj))
        NUM['small_nfam'] = len(fam)
    hdr = ['Dataset', 'Model', 'Folds', 'ROC-AUC', 'PR-AUC', 'Brier', 'F1 (τ\\*)', 'AAGB-IG − model, ROC-AUC [95% CI]', 'Holm *p*']
    T['newbase'] = (hdr, rows)
    # ---- table: TabNet-feature stacking audit
    rows = []
    for ds in ('heart', 'pima', 'ckd', 'stroke'):
        if ds not in LK or 'L:xgb' not in LK[ds]:
            continue
        for v, lab in LEAK.items():
            if v not in LK[ds]:
                continue
            row = [NAMES[ds], lab]
            for j, k in enumerate(('roc', 'pr', 'brier')):
                dl, nn = paired(LK[ds][v], LK[ds]['L:xgb'], j)
                mu, hw, p, _ = ci(dl)
                NUM['%s.leak.%s.%s' % (ds, v[2:], k)] = dict(mean=mu, lo=mu - hw, hi=mu + hw, p=p, n=nn)
                row.append('%s [%s, %s]%s' % (f3(mu, True), f3(mu - hw, True), f3(mu + hw, True), '\\*' if p < 0.05 else ''))
            rows.append(row)
    T['leak'] = (['Dataset', 'Extra XGBoost inputs', 'ΔROC-AUC', 'ΔPR-AUC', 'ΔBrier'], rows)
    # ---- table: larger datasets
    rows, lfam = [], []
    for ds in ('diab130', 'cdc'):
        if ds not in FM:
            continue
        for m in ('rf', 'xgb', 'lgbm', 'cat', 'mlp', 'tabpfn', 'aagb_rm', 'aagb_ig', 'aagb_ig_ins'):
            if m not in FM[ds]:
                continue
            d = FM[ds][m]
            st = {k: ci([v[idx[k]] for v in d.values()]) for k in ('roc', 'pr', 'brier', 'ece', 'f1t')}
            NUM['%s.%s' % (ds, m)] = {k: dict(mean=st[k][0], hw=st[k][1], n=st[k][3]) for k in st}
            NUM['%s.%s' % (ds, m)]['secs'] = float(np.nanmean([v[idx['secs10']] * 1e3 for v in d.values()]))
            row = [NAMES[ds], MOD[m]] + ['%s ± %s' % (f3(st[k][0]), f3(st[k][1])) for k in ('roc', 'pr', 'brier', 'ece', 'f1t')]
            if m != 'aagb_ig' and 'aagb_ig' in FM[ds]:
                dl, nn = paired(FM[ds]['aagb_ig'], d, idx['roc'])
                mu, hw, p, _ = ci(dl)
                NUM['%s.aagb_ig_vs_%s.roc' % (ds, m)] = dict(mean=mu, lo=mu - hw, hi=mu + hw, p=p, n=nn)
                row.append('%s [%s, %s]' % (f3(mu, True), f3(mu - hw, True), f3(mu + hw, True)))
                lfam.append((ds, m, p))
                row.append('LH%d' % (len(lfam) - 1))
            else:
                row += ['—', '—']
            rows.append(row)
    ladj = holm([p for _, _, p in lfam]) if lfam else []
    for r in rows:
        if r[-1].startswith('LH'):
            i = int(r[-1][2:])
            r[-1] = ('%.3f' % ladj[i]) if ladj[i] >= 0.001 else '<0.001'
            NUM['%s.aagb_ig_vs_%s.roc' % (lfam[i][0], lfam[i][1])]['p_holm'] = float(ladj[i])
    if lfam:
        NUM['large_min_holm'] = float(min(ladj))
        NUM['large_nfam'] = len(lfam)
    T['large'] = (['Dataset', 'Model', 'ROC-AUC', 'PR-AUC', 'Brier', 'ECE', 'F1 (τ\\*)', 'AAGB-IG − model, ROC-AUC [95% CI]', 'Holm *p*'], rows)
    au = {ds: [int(x) for x in v.split(',')] for (ds, k), v in meta.get('AUD3', {}).items()}
    inf = {ds: [int(x) for x in v.split(',')] for (ds, k), v in meta.get('INFO3', {}).items()}
    if 'diab130' in au:
        a = au['diab130']
        NUM.update({'diab130.n0': a[0], 'diab130.n1': a[1], 'diab130.n2': a[2], 'diab130.prev2': a[3] / a[2], 'diab130.n': a[4],
                    'diab130.prev': a[5] / a[4], 'diab130.dup': a[6]})
    if 'cdc' in au:
        a = au['cdc']
        NUM.update({'cdc.n0': a[0], 'cdc.dup0': a[1], 'cdc.prev0': a[2] / a[0], 'cdc.n': a[3], 'cdc.prev': a[4] / a[3], 'cdc.dup': a[5]})
    for ds, v in inf.items():
        NUM[ds + '.dcols'], NUM[ds + '.nvars'] = v[4], v[5]
    NUM['ctx'] = 10000
    for name, (h, rows) in T.items():
        with open(os.path.join(out, 'tables', 'v3_%s.md' % name), 'w') as fh:
            fh.write('| ' + ' | '.join(h) + ' |\n| ' + ' | '.join([':--', ':--'] + [':-:'] * (len(h) - 2)) + ' |\n')
            for r in rows:
                fh.write('| ' + ' | '.join(r) + ' |\n')
    with open(os.path.join(out, 'numbers_v3.json'), 'w') as fh:
        json.dump(NUM, fh, indent=1, default=float)
    for name, (h, rows) in T.items():
        print('==', name)
        for r in rows:
            print(' | '.join(r))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
