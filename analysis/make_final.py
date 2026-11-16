"""Tables, figures and quoted numbers for the revised manuscript, computed from the checksummed result lines of the
Kaggle runs (v2 Heart+Pima, v3 CKD, v4 Stroke, v5 Stroke FT-Transformer, v6 supplementary experiments)."""
import os, sys, json, math, glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from lib import f3, fp

ROOT = '/home/claude/work/results'
OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/work/out'
os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'figs'), exist_ok=True)
FILES = sorted(glob.glob(os.path.join(ROOT, 'v*', 'log_lines.txt')))
REP, BAD, SEEN = lib.parse_report(FILES)
assert BAD == 0, BAD
HW = [l.strip()[5:] for f in FILES for l in open(f) if l.startswith('@@HW|')]
DS = ['heart', 'pima', 'ckd', 'stroke']
DSN = {'heart': 'Heart Disease', 'pima': 'Pima', 'ckd': 'CKD', 'stroke': 'Stroke'}
MODELS = ['rf', 'xgb', 'lgbm', 'cat', 'mlp', 'tabnet', 'ftt', 'aagb_rm', 'aagb_ig']
MN = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'cat': 'CatBoost', 'mlp': 'MLP', 'tabnet': 'TabNet',
      'ftt': 'FT-Transformer', 'aagb_rm': 'AAGB-RM', 'aagb_ig': 'AAGB-IG'}
MET = ['roc', 'pr', 'brier', 'ece', 'f15', 'f1t', 'mcct', 'senst', 'spect', 'prect', 'bacct', 'thr']
PT = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
PM, MINUS = '±', '−'
N, T = {}, {}
N['lines'], N['bad_lines'], N['hw'] = SEEN, BAD, HW[0] if HW else ''


def R(d):
    return REP[d]


def md(header, rows, align=None):
    align = align or ['l'] + ['c'] * (len(header) - 1)
    sep = {'l': ':--', 'c': ':-:', 'r': '--:'}
    out = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(sep[a] for a in align) + ' |']
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def ptest(m, sd, n, rope=0.01):
    se = math.sqrt((1.0 / n + lib.RHO) * sd ** 2)
    if se == 0:
        return dict(mean=m, lo=m, hi=m, p=1.0 if m == 0 else 0.0, pl=float(m < -rope), pr0=float(abs(m) <= rope), pg=float(m > rope), dz=float('nan'))
    h = stats.t.ppf(0.975, n - 1) * se
    post = stats.t(n - 1, loc=m, scale=se)
    return dict(mean=m, lo=m - h, hi=m + h, p=float(2 * stats.t.sf(abs(m / se), n - 1)), pl=float(post.cdf(-rope)),
                pr0=float(post.cdf(rope) - post.cdf(-rope)), pg=float(post.sf(rope)), dz=m / sd if sd > 0 else float('nan'))


def ci(x, plus=True):
    return '%s [%s, %s]' % (f3(x['mean'], plus), f3(x['lo'], True), f3(x['hi'], True))


# ------------------------------------------------------------- dataset facts
for d in DS:
    r = R(d)
    cfg = [int(v) for v in r['INFO']['cfg']]
    N[d + '.R'], N[d + '.trials'], N[d + '.n'], N[d + '.pos'], N[d + '.dcols'], N[d + '.nvars'], N[d + '.nfold'] = cfg
    N[d + '.neg'] = cfg[2] - cfg[3]
    N[d + '.prev'] = cfg[3] / cfg[2]
    N[d + '.vars'] = r['VARS']['names'].split(';')
    for k, v in r['AUD'].items():
        N['%s.aud.%s' % (d, k)] = [int(x) for x in v]
    N[d + '.det'] = r['DET']['maxdiff']
    N[d + '.gc'] = r['GC']['spread']
    N[d + '.runtime_h'] = r['DONE']['secs'][0] / 3600.0
NF = 50
N['nfold'] = NF
N['mde.n5'], N['mde.n50'] = lib.min_detectable_dz(5), lib.min_detectable_dz(50)
N['mde.n50.holm'] = float(stats.t.ppf(1 - 0.05 / 2 / 32, NF - 1) * math.sqrt(1.0 / NF + lib.RHO))
N['rho'] = lib.RHO

# ------------------------------------------------------------- main benchmark
ra, rb = [], []
for d in DS:
    M = R(d)['M']
    models = [m for m in MODELS if m in M]
    v = {m: dict(zip(MET, [(M[m][2 * i], M[m][2 * i + 1]) for i in range(len(MET))])) for m in models}
    for m in models:
        for k in MET:
            N['%s.%s.%s' % (d, m, k)], N['%s.%s.%s.ci' % (d, m, k)] = v[m][k]
    best = {}
    for k in ('roc', 'pr', 'brier', 'ece', 'f15', 'f1t', 'mcct'):
        vals = [v[m][k][0] for m in models]
        best[k] = models[int(np.argmin(vals) if k in ('brier', 'ece') else np.argmax(vals))]
        N['%s.best.%s' % (d, k)] = best[k]
    order = sorted(models, key=lambda m: -v[m]['roc'][0])
    for m in models:
        N['%s.%s.rank_roc' % (d, m)] = order.index(m) + 1
    N[d + '.order_roc'] = order
    for i, m in enumerate(models):
        def c(k):
            s = '%s %s %s' % (f3(v[m][k][0]), PM, f3(v[m][k][1]))
            return '**%s**' % s if best.get(k) == m else s
        ra.append([DSN[d] if i == 0 else '', MN[m]] + [c(k) for k in ('roc', 'pr', 'brier', 'ece')])
        rb.append([DSN[d] if i == 0 else '', MN[m]] + [c(k) for k in ('f15', 'f1t', 'mcct')] +
                  [f3(v[m][k][0]) for k in ('senst', 'spect', 'prect')] + ['%.2f' % v[m]['thr'][0]])
T['main_a'] = md(['Dataset', 'Model', 'ROC-AUC', 'PR-AUC', 'Brier', 'ECE'], ra, ['l', 'l', 'c', 'c', 'c', 'c'])
T['main_b'] = md(['Dataset', 'Model', 'F1 (0.5)', 'F1 (τ\\*)', 'MCC (τ\\*)', 'Sens.', 'Spec.', 'Prec.', 'τ\\*'], rb, ['l', 'l'] + ['c'] * 7)

# ------------------------------------------------------------- paired tests: AAGB-IG vs every baseline
TK = ['roc', 'pr', 'brier', 'f1t', 'f15']
comp = []
for d in DS:
    for key, vals in R(d)['T'].items():
        a, b = key.split('>')
        res = {k: ptest(vals[2 * i], vals[2 * i + 1], NF) for i, k in enumerate(TK)}
        for k, x in res.items():
            for q, val in x.items():
                N['%s.%s_vs_%s.%s.%s' % (d, a, b, k, q)] = val
        if a == 'aagb_ig':
            comp.append((d, b, res))
adj = lib.holm([c[2]['roc']['p'] for c in comp])
N['tests.n'] = len(comp)
for (d, b, res), a in zip(comp, adj):
    N['%s.aagb_ig_vs_%s.roc.holm' % (d, b)] = float(a)
N['tests.sig_raw'] = [(d, b, round(res['roc']['mean'], 4)) for d, b, res in comp if res['roc']['p'] < 0.05]
N['tests.sig_holm'] = [(d, b, round(res['roc']['mean'], 4)) for (d, b, res), a in zip(comp, adj) if a < 0.05]
N['tests.n_sig_holm'] = len(N['tests.sig_holm'])
N['tests.n_sig_holm_worse'] = sum(1 for x in N['tests.sig_holm'] if x[2] < 0)
N['tests.n_rope_gt_half'] = sum(1 for d, b, res in comp if res['roc']['pr0'] > 0.5)
N['tests.min_holm'] = float(min(adj))
rows, last = [], None
for (d, b, res), a in zip(comp, adj):
    x = res['roc']
    rows.append([DSN[d] if d != last else '', MN[b], ci(x), fp(x['p']), fp(a), '%.2f / %.2f / %.2f' % (x['pl'], x['pr0'], x['pg']),
                 ci(res['pr']), ci(res['brier'])])
    last = d
T['tests'] = md(['Dataset', 'Baseline', 'ΔROC-AUC [95% CI]', '*p*', '*p* (Holm)', 'P(Δ < −0.01) / P(ROPE) / P(Δ > 0.01)',
                 'ΔPR-AUC [95% CI]', 'ΔBrier [95% CI]'], rows, ['l', 'l', 'c', 'c', 'c', 'c', 'c', 'c'])
# Stroke FT-Transformer: separate run -> unpaired description only
N['stroke.ftt_unpaired.roc_diff'] = N['stroke.aagb_ig.roc'] - N['stroke.ftt.roc']

# ------------------------------------------------------------- ablation + supplementary variants
AB = [('full_ig', 'Full AAGB-IG: input gate, cross-fitted'), ('full_ig_ins', 'Input gate, in-sample attention'),
      ('full_rm', 'Row-mean gate, cross-fitted'), ('full_rm_ins', 'Row-mean gate, in-sample (original design)'),
      ('full_pj', 'Projection gate, cross-fitted'), ('w_only', '*X* + *X*~w~ (no products)'), ('int_only', '*X* + *X*~int~ (attention-selected pairs)'),
      ('int_rand', '*X* + products of 5 random variables'), ('int_gain', '*X* + products of top-5 gain variables'),
      ('full_ig_nosa', 'Without sample attention'), ('nofa', 'Without feature attention (*X* + *a*~s~*X*)'),
      ('full_ig_untr', 'Untrained (random-weight) attention module'), ('full_ig_l1a', 'Sparsity weight 0.01'), ('full_ig_l1b', 'Sparsity weight 0.05')]
XA = [('full_ig_sm', 'Row-normalised (softmax) input gate'), ('attnb_ins', 'Hidden-unit attention [16], in-sample'),
      ('attnb_cf', 'Hidden-unit attention [16], cross-fitted')]
rows = [['XGBoost on *X* (reference ROC-AUC)'] + [f3(R(d)['AB']['xgb'][0]) for d in DS]]
for d in DS:
    N['%s.abl.xgb.roc' % d], N['%s.abl.xgb.pr' % d], N['%s.abl.xgb.brier' % d] = R(d)['AB']['xgb']
for v, name in AB + XA:
    row = [name]
    for d in DS:
        r = R(d)
        if v in r.get('ABD', {}):
            vals = r['ABD'][v]
            N['%s.abl.%s.roc' % (d, v)] = r['AB'][v][0]
        else:
            vals = r['XBD'][v]
            N['%s.abl.%s.roc' % (d, v)] = r['XB'][v][0]
        x = ptest(vals[0], vals[1], NF)
        xb = ptest(vals[4], vals[5], NF)
        for q, val in x.items():
            N['%s.abl.%s.droc.%s' % (d, v, q)] = val
        for q, val in xb.items():
            N['%s.abl.%s.dbrier.%s' % (d, v, q)] = val
        row.append(ci(x) + ('\\*' if x['p'] < 0.05 else ''))
    rows.append(row)
T['ablation'] = md(['Variant (XGBoost hyperparameters fixed)'] + [DSN[d] for d in DS], rows)
# vs full_ig (second half of ABD)
for d in DS:
    for v, _ in AB:
        vals = R(d)['ABD'][v]
        x = ptest(vals[6], vals[7], NF)
        for q, val in x.items():
            N['%s.abl.%s.droc_vs_ig.%s' % (d, v, q)] = val

# ------------------------------------------------------------- gate diagnostics
GN = [('rm', 'Row-mean'), ('pj', 'Projection'), ('ig', 'Input-level'), ('ig_sm', 'Input-level, softmax')]
rows = []
for d in DS:
    r = R(d)
    for i, (g, name) in enumerate(GN):
        src = ('GD', 'GDL', 'GDT', 'GA') if g in r['GD'] else ('XGD', 'XGDL', 'XGDT', 'XGA')
        gd, dl, dt, ga = (r[s][g] for s in src)
        keys = ('rng', 'mean', 'sat', 'wsd', 'asd', 'spear', 'jac', 'vrange')
        for k, val in zip(keys, gd):
            N['%s.gate.%s.%s' % (d, g, k)] = val
        dlr = np.array(dl).reshape(4, 3)
        for s, vals in zip(('top', 'bot', 'rnd', 'shp'), dlr):
            for k, val in zip((1, 2, 3), vals):
                N['%s.gate.%s.del.%s.%d' % (d, g, s, k)] = float(val)
        x = ptest(dt[0], dt[1], NF)
        for q, val in x.items():
            N['%s.gate.%s.topminusrnd.%s' % (d, g, q)] = val
        seedk = {'rm': 'rm', 'pj': 'pj', 'ig': 'ig'}.get(g)
        seed = r['SEED'][seedk][0] if seedk else float('nan')
        if seedk:
            N['%s.seed.%s.spear' % (d, g)], N['%s.seed.%s.jac' % (d, g)] = r['SEED'][seedk]
        vn = N[d + '.vars']
        a = np.array(ga)
        N['%s.gate.%s.att' % (d, g)] = dict(zip(vn, a.tolist()))
        N['%s.gate.%s.top3' % (d, g)] = [vn[j] for j in np.argsort(-a, kind='stable')[:3]]
        N['%s.gate.%s.bottom' % (d, g)] = vn[int(np.argmin(a))]
        N['%s.gate.%s.rank' % (d, g)] = dict(zip(vn, stats.rankdata(-a).tolist()))
        rows.append([DSN[d] if i == 0 else '', name, f3(gd[1]), f3(gd[0]), '%.1f%%' % (100 * gd[2]), f3(gd[3]), f3(gd[4]),
                     f3(gd[5]), 'n/a' if not seedk else f3(seed), f3(dlr[0][0]), f3(dlr[2][0]), f3(dlr[3][0]),
                     ci(x) + ('\\*' if x['p'] < 0.05 else '')])
    if 'GRM' in r:
        N[d + '.gate.rm.agree'], N[d + '.gate.rm.gprod'] = r['GRM']['rm']
T['gates'] = md(['Dataset', 'Gate', 'Mean *A*~f~', 'Range of variable means', '*A*~f~ > 0.95', 'Within-row SD', 'Between-row SD',
                 'Rank stability, folds', 'Rank stability, seeds', 'Δ*p*: top-1 attended', 'Δ*p*: random', 'Δ*p*: top-1 SHAP',
                 'Top-1 − random [95% CI]'], rows, ['l', 'l'] + ['c'] * 11)

# ------------------------------------------------------------- explanations
rows = []
for d in DS:
    r = R(d)
    vn = N[d + '.vars']
    for m in ('xgb', 'aagb_ig'):
        sp, jc, s1, s2, s3 = r['SHS'][m]
        N['%s.shap.%s.spear' % (d, m)], N['%s.shap.%s.jac' % (d, m)] = sp, jc
        N['%s.shap.%s.share' % (d, m)] = [s1, s2, s3]
        a = np.array(r['SH'][m])
        N['%s.shap.%s.top3' % (d, m)] = [vn[j] for j in np.argsort(-a, kind='stable')[:3]]
        N['%s.shap.%s.rank' % (d, m)] = dict(zip(vn, stats.rankdata(-a).tolist()))
        N['%s.shap.%s.val' % (d, m)] = dict(zip(vn, a.tolist()))
    N['%s.seed.shap_xgb' % d], N['%s.seed.shap_ig' % d] = r['SEED']['shap_xgb'], r['SEED']['shap_ig']
    keys = r['AGR']
    k0 = list(keys)[0]
    names = k0.split(',')
    import itertools
    for (a_, b_), val in zip(itertools.combinations(names, 2), keys[k0]):
        N['%s.agree.%s~%s' % (d, a_, b_)] = val
    rows.append([DSN[d], ', '.join(N['%s.shap.xgb.top3' % d]), ', '.join(N['%s.shap.aagb_ig.top3' % d]), ', '.join(N['%s.gate.ig.top3' % d]),
                 '%.2f / %.2f' % (r['SHS']['xgb'][0], r['SEED']['shap_xgb'][0]), '%.2f / %.2f' % (r['SHS']['aagb_ig'][0], r['SEED']['shap_ig'][0]),
                 '%.2f' % N['%s.agree.shap_aagb_ig~shap_xgb' % d], '%.2f' % N['%s.agree.att~shap_aagb_ig' % d],
                 '%.0f / %.0f / %.0f%%' % tuple(100 * np.array(r['SHS']['aagb_ig'][2:]))])
T['explain'] = md(['Dataset', 'Top-3, XGBoost SHAP', 'Top-3, AAGB-IG SHAP (aggregated)', 'Top-3, AAGB-IG attention',
                   'SHAP rank stability XGBoost (folds / seeds)', 'SHAP rank stability AAGB-IG (folds / seeds)',
                   'ρ(SHAP AAGB-IG, SHAP XGBoost)', 'ρ(attention, SHAP AAGB-IG)', 'SHAP share *X* / *X*~w~ / *X*~int~'],
                  rows, ['l'] + ['c'] * 8)

# ------------------------------------------------------------- imbalance strategies and calibration
VAR = [('', 'None'), ('+cal', 'None + Platt'), ('+os', 'Oversampling'), ('+os+cal', 'Oversampling + Platt'),
       ('+spw', 'Class weighting'), ('+spw+cal', 'Class weighting + Platt')]
rows = []
for d in DS:
    r = R(d)
    for m in list(r['M']):
        cal = r['CAL'][m]
        N['%s.%s.cslope' % (d, m)], N['%s.%s.cint' % (d, m)], N['%s.%s.citl' % (d, m)] = cal
        for k, i in (('roc', 0), ('brier', 2), ('ece', 3), ('f15', 4), ('f1t', 5), ('senst', 7), ('spect', 8), ('prect', 9), ('thr', 11)):
            N['%s.%s.%s' % (d, m, k)] = r['M'][m][2 * i]
    if 'xgb+os' not in r['M']:
        continue
    first = True
    for m in ('xgb', 'aagb_ig'):
        for sfx, name in VAR:
            mm = m + sfx
            if mm not in r['M']:
                continue
            M_ = r['M'][mm]
            cal = r['CAL'][mm]
            rows.append([DSN[d] if first else '', MN[m] if sfx == '' else '', name, f3(M_[0]), f3(M_[4]), f3(M_[6]), '%.2f' % cal[0],
                         f3(cal[2], True), f3(M_[8]), f3(M_[10]), f3(M_[14]), f3(M_[16]), f3(M_[18])])
            first = False
T['imbalance'] = md(['Dataset', 'Model', 'Correction', 'ROC-AUC', 'Brier', 'ECE', 'Cal. slope', 'Mean *p* − prevalence', 'F1 (0.5)',
                     'F1 (τ\\*)', 'Sens. (τ\\*)', 'Spec. (τ\\*)', 'Prec. (τ\\*)'], rows, ['l', 'l', 'l'] + ['c'] * 10)

# ------------------------------------------------------------- factorial
EFF = ['A', 'C', 'W', 'S', 'A×C', 'A×W', 'A×S', 'C×W', 'C×S', 'W×S']
EFN = {'A': 'A: oversampling for the attention module', 'C': 'C: oversampling for XGBoost', 'W': 'W: class-weighted BCE',
       'S': 'S: scale_pos_weight'}
rows = []
for i, e in enumerate(EFF):
    row = [EFN.get(e, e)]
    for d in DS:
        for k in ('f15', 'brier'):
            v = R(d)['F'][k]
            m_, s_ = v[2 * i], v[2 * i + 1]
            key = '%s.fact.%s.%s' % (d, e.replace('×', 'x'), k)
            if m_ == 0 and s_ == 0:
                N[key] = None
                row.append('—')
                continue
            x = ptest(m_, s_, NF)
            for q, val in x.items():
                N[key + '.' + q] = val
            N[key] = m_
            row.append(ci(x) + ('\\*' if x['p'] < 0.05 else ''))
    rows.append(row)
hdr = ['Effect']
for d in DS:
    hdr += [DSN[d] + ' F1 (0.5)', DSN[d] + ' Brier']
T['factorial'] = md(hdr, rows)
for d in DS:
    order = R(d)['FCOL']['order'].split(';')
    for k in ('f15', 'pr', 'brier'):
        N['%s.cell.%s' % (d, k)] = dict(zip(order, R(d)['FC'][k]))

# ------------------------------------------------------------- dataset sensitivity
SN = {'redist': 'Redistributed file as first submitted', 'redist_fix': 'Redistributed file, cleaned', 'zeros': 'Zeros kept as values',
      'no_sc': 'Without serum creatinine', 'no_hemo': 'Without haemoglobin', 'no_pcv': 'Without packed cell volume', 'no_all3': 'Without all three'}
rows = []
for d in DS:
    S = R(d).get('S', {})
    if not S:
        continue
    rows.append([DSN[d], 'Main analysis'] + ['%s %s %s' % (f3(N['%s.%s.%s' % (d, m, k)]), PM, f3(N['%s.%s.%s.ci' % (d, m, k)]))
                                             for m in ('xgb', 'aagb_ig') for k in ('roc', 'brier')])
    for v in SN:
        if v + ':xgb' not in S:
            continue
        cells = []
        for m in ('xgb', 'aagb_ig'):
            x = S[v + ':' + m]
            for k, i in (('roc', 0), ('pr', 1), ('brier', 2), ('f1t', 3), ('f15', 4)):
                N['%s.sens.%s.%s.%s' % (d, v, m, k)], N['%s.sens.%s.%s.%s.ci' % (d, v, m, k)] = x[2 * i], x[2 * i + 1]
            cells += ['%s %s %s' % (f3(x[0]), PM, f3(x[1])), '%s %s %s' % (f3(x[4]), PM, f3(x[5]))]
        rows.append(['', SN[v]] + cells)
T['sensitivity'] = md(['Dataset', 'Variant', 'XGBoost ROC-AUC', 'XGBoost Brier', 'AAGB-IG ROC-AUC', 'AAGB-IG Brier'], rows, ['l', 'l', 'c', 'c', 'c', 'c'])

# ------------------------------------------------------------- timing
rows = []
for m in MODELS:
    row = [MN[m]]
    for d in DS:
        tm = R(d)['TM'][m]
        N['%s.time.%s.fit_s' % (d, m)], N['%s.time.%s.pred_ms' % (d, m)] = tm[0] / 1000.0, tm[1] / 1000.0
        row += ['%.2f' % (tm[0] / 1000.0), '%.1f' % (tm[1] / 1000.0)]
    sec = [R(d)['SEC'][m][0] for d in DS if m in R(d)['SEC']]
    N['time.%s.tune_h_total' % m] = sum(sec) / 3600.0
    rows.append(row)
hdr = ['Model']
for d in DS:
    hdr += [DSN[d] + ' fit (s)', DSN[d] + ' ms/1,000']
T['timing'] = md(hdr, rows)
for d in DS:
    for m in MODELS:
        if m in R(d)['SEC']:
            N['%s.sec.%s' % (d, m)] = R(d)['SEC'][m][0]

# ================================================================= figures
INK, BLUE, ORANGE, GREEN, GREY, RED, PURPLE = '#1a1a1a', '#1f5fa8', '#c96a00', '#2e7d32', '#8a8a8a', '#b3261e', '#6a3d9a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.linewidth': 0.7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})


def save(fig, name):
    fig.savefig(os.path.join(OUT, 'figs', name), dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def short(v):
    return {'DiabetesPedigreeFunction': 'DPF', 'avg_glucose_level': 'avg_glucose', 'Residence_type': 'residence', 'smoking_status': 'smoking',
            'SkinThickness': 'SkinThick.', 'BloodPressure': 'BloodPress.'}.get(v, v)


# Fig. 2 attention by source variable, three gates
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
for ax, d, let in zip(axes.ravel(), DS, 'abcd'):
    vn = N[d + '.vars']
    ig = np.array(R(d)['GA']['ig'])
    order = np.argsort(ig, kind='stable')
    if len(order) > 14:
        order = order[-14:]
    y = np.arange(len(order))
    for off, g, col, lab in ((0.27, 'ig', BLUE, 'Input-level (AAGB-IG)'), (0.0, 'pj', GREEN, 'Projection'), (-0.27, 'rm', ORANGE, 'Row-mean (original)')):
        ax.barh(y + off, np.array(R(d)['GA'][g])[order], 0.26, color=col, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([short(vn[j]) for j in order], fontsize=6.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Mean feature attention on test rows')
    ax.set_title('(%s) %s%s' % (let, DSN[d], ' (14 of %d variables)' % len(vn) if len(vn) > 14 else ''), loc='left', fontsize=8, fontweight='bold')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.97))
save(fig, 'fig2_attention.png')

# Fig. 3 aggregated SHAP
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
for ax, d, let in zip(axes.ravel(), DS, 'abcd'):
    vn = N[d + '.vars']
    s1, s0 = np.array(R(d)['SH']['aagb_ig']), np.array(R(d)['SH']['xgb'])
    order = np.argsort(s1, kind='stable')
    if len(order) > 12:
        order = order[-12:]
    y = np.arange(len(order))
    ax.barh(y + 0.19, s1[order], 0.36, color=BLUE, label='AAGB-IG (aggregated to source variable)')
    ax.barh(y - 0.19, s0[order], 0.36, color=GREY, label='XGBoost on X')
    ax.set_yticks(y)
    ax.set_yticklabels([short(vn[j]) for j in order], fontsize=6.4)
    ax.set_xlabel('Mean |SHAP| per source variable (log-odds)')
    ax.set_title('(%s) %s%s' % (let, DSN[d], ' (12 of %d variables)' % len(vn) if len(vn) > 12 else ''), loc='left', fontsize=8, fontweight='bold')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.97))
save(fig, 'fig3_shap.png')

# Fig. 4 calibration curves
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
for ax, d, let in zip(axes.ravel(), DS, 'abcd'):
    cc = R(d)['CC']
    sty = [('xgb', INK, '-', 'o', 'XGBoost'), ('aagb_ig', BLUE, '-', 's', 'AAGB-IG'), ('aagb_rm', ORANGE, '-.', 'D', 'AAGB-RM'),
           ('xgb+os', RED, '--', '^', 'XGBoost + oversampling'), ('xgb+os+cal', GREEN, ':', 'v', 'XGBoost + oversampling + Platt')]
    hi = 0
    for m, c, ls, mk, lab in sty:
        if m in cc:
            a = np.array(cc[m]).reshape(10, 2)
            ax.plot(a[:, 0], a[:, 1], color=c, ls=ls, marker=mk, ms=2.6, lw=1.0, label=lab)
            hi = max(hi, a.max())
    lim = 1.0 if hi > 0.6 else min(1.0, 1.1 * hi)
    ax.plot([0, lim], [0, lim], color=GREY, lw=0.6, ls=(0, (1, 2)))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Mean predicted risk (decile of predictions)')
    ax.set_ylabel('Observed event rate')
    ax.set_title('(%s) %s' % (let, DSN[d]), loc='left', fontsize=8, fontweight='bold')
    ax.legend(loc='upper left', fontsize=6.0)
fig.tight_layout()
save(fig, 'fig4_calibration.png')

# Fig. 5 decision curves
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
for ax, d, let in zip(axes.ravel(), DS, 'abcd'):
    dc = R(d)['DC']
    for m, c, ls, lab in (('treat_all', GREY, '--', 'Treat all'), ('xgb', INK, '-', 'XGBoost'), ('aagb_ig', BLUE, '-', 'AAGB-IG'),
                          ('cat', GREEN, ':', 'CatBoost'), ('xgb+os', RED, '-.', 'XGBoost + oversampling')):
        if m in dc:
            ax.plot(PT, dc[m], color=c, ls=ls, lw=1.0, marker='o', ms=2, label=lab)
            for pt, val in zip(PT, dc[m]):
                N['%s.nb.%s.%s' % (d, m, pt)] = val
    ax.axhline(0, color=INK, lw=0.6, ls=(0, (1, 2)), label='Treat none')
    prev = N[d + '.prev']
    ax.set_ylim(-0.3 * prev, 1.08 * prev)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title('(%s) %s' % (let, DSN[d]), loc='left', fontsize=8, fontweight='bold')
    ax.legend(loc='upper right', fontsize=6.0)
fig.tight_layout()
save(fig, 'fig5_decision.png')

# Fig. 6 ablation forest plot
keys = [k for k, _ in AB] + [k for k, _ in XA]
lab = {'full_ig': 'Full AAGB-IG (input gate, cross-fitted)', 'full_ig_ins': 'Input gate, in-sample', 'full_rm': 'Row-mean gate, cross-fitted',
       'full_rm_ins': 'Row-mean gate, in-sample (original)', 'full_pj': 'Projection gate, cross-fitted', 'w_only': 'X + X_w only',
       'int_only': 'X + X_int only', 'int_rand': 'X + random-pair products', 'int_gain': 'X + top-gain-pair products',
       'full_ig_nosa': 'No sample attention', 'nofa': 'No feature attention', 'full_ig_untr': 'Untrained attention module',
       'full_ig_l1a': 'Sparsity 0.01', 'full_ig_l1b': 'Sparsity 0.05', 'full_ig_sm': 'Softmax input gate',
       'attnb_ins': 'Hidden-unit attention, in-sample', 'attnb_cf': 'Hidden-unit attention, cross-fitted'}
fig, axes = plt.subplots(1, 4, figsize=(7.2, 4.4), sharey=True)
ks = keys[::-1]
for ax, d, let in zip(axes, DS, 'abcd'):
    for i, v in enumerate(ks):
        m_, lo, hi = (N['%s.abl.%s.droc.%s' % (d, v, q)] for q in ('mean', 'lo', 'hi'))
        c = BLUE if v == 'full_ig' else (ORANGE if v == 'full_rm_ins' else (PURPLE if v.startswith('attnb') else INK))
        ax.plot([lo, hi], [i, i], color=c, lw=1.0)
        ax.plot([m_], [i], 'o', color=c, ms=2.8)
    ax.axvline(0, color=GREY, lw=0.7, ls='--')
    ax.set_title('(%s) %s' % (let, DSN[d]), loc='left', fontsize=8, fontweight='bold')
    ax.set_xlabel('ΔROC-AUC vs XGBoost', fontsize=7)
    ax.tick_params(axis='x', labelsize=6.2)
axes[0].set_yticks(range(len(ks)))
axes[0].set_yticklabels([lab[k] for k in ks], fontsize=6.3)
fig.tight_layout()
save(fig, 'fig6_ablation.png')


def default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


json.dump(N, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1, default=default)
for k, v in T.items():
    open(os.path.join(OUT, 'tables', k + '.md'), 'w', encoding='utf8').write(v + '\n')
print('lines', SEEN, 'bad', BAD, 'numbers', len(N), 'tables', list(T))
