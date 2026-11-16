# AAGB v3 - additional experiments (imports aagb2.py):
#   (1) TabPFN v2 and TabNet / FT-Transformer tuned with the same 15-trial TPE budget on the four small datasets,
#       with XGBoost and AAGB-IG re-run in the same job so that every comparison is paired by outer fold;
#   (2) leakage audit of TabNet-feature stacking (as in Xia et al.): in-sample versus cross-fitted TabNet masks
#       and TabNet decision embeddings as extra XGBoost inputs;
#   (3) two larger datasets (Diabetes 130-US Hospitals, CDC Diabetes Health Indicators; 10,000-record stratified samples) with all baselines,
#       AAGB-RM, AAGB-IG and in-sample AAGB-IG.
# Every result is printed as a checksummed @@ line (one line per model and outer fold).
import os, time, itertools, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import optuna
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score
import aagb2 as A2

K_OUT = A2.K_OUT
FMK = ['roc', 'pr', 'brier', 'ece', 'f15', 'f1t', 'senst', 'spect', 'thr', 'secs']
LARGE = ('diab130', 'cdc')
N_LARGE = 10000        # each larger dataset is reduced to a stratified random sample of 10,000 records (seed 42)
MEDS_MIN = 0.01        # medication columns kept when at least 1% of encounters are not 'No'
DROP_DISP = {11, 13, 14, 19, 20, 21}  # discharged as expired or to hospice (Strack et al.)


# ---------------------------------------------------------------- data for the larger datasets
def _uci(i):
    from ucimlrepo import fetch_ucirepo
    return fetch_ucirepo(id=i).data.original


def _icd(v):
    if not isinstance(v, str) or v == '':
        return 'Missing'
    if v[0] in 'VE':
        return 'Other'
    x = float(v)
    if 250 <= x < 251:
        return 'Diabetes'
    if 390 <= x <= 459 or int(x) == 785:
        return 'Circulatory'
    if 460 <= x <= 519 or int(x) == 786:
        return 'Respiratory'
    if 520 <= x <= 579 or int(x) == 787:
        return 'Digestive'
    if 800 <= x <= 999:
        return 'Injury'
    if 710 <= x <= 739:
        return 'Musculoskeletal'
    if 580 <= x <= 629 or int(x) == 788:
        return 'Genitourinary'
    if 140 <= x <= 239:
        return 'Neoplasms'
    return 'Other'


def _rare(s, frac=0.01):
    vc = s.value_counts(normalize=True)
    keep = set(vc[vc >= frac].index)
    return s.where(s.isin(keep), 'Other')


AUDIT = {}


def load_large(ds):
    key = (ds, 'main')
    if key in A2._CACHE:
        return A2._CACHE[key]
    pk = os.path.join(A2.OUT, ds + '_prepared.pkl')
    if os.path.exists(pk):  # written by the parent process so that workers do not download again
        A2._CACHE[key] = pd.read_pickle(pk)
        return A2._CACHE[key]
    if ds == 'diab130':
        try:
            raw = pd.read_csv(A2._find('diabetic_data.csv'), dtype=str)
        except FileNotFoundError:
            raw = _uci(296).astype(str).replace({'nan': np.nan, 'None': np.nan})
        raw = raw.replace('?', np.nan)
        n0 = len(raw)
        raw['encounter_id'] = raw['encounter_id'].astype(int)
        raw = raw.sort_values('encounter_id', kind='stable').drop_duplicates('patient_nbr', keep='first')
        n1 = len(raw)
        raw = raw[~raw['discharge_disposition_id'].astype(int).isin(DROP_DISP)].reset_index(drop=True)
        n2, yall = len(raw), (raw['readmitted'] == '<30').astype(int).values
        keep = train_test_split(np.arange(n2), train_size=N_LARGE, stratify=yall, random_state=A2.SEED)[0]
        raw = raw.iloc[np.sort(keep)].reset_index(drop=True)
        y = (raw['readmitted'] == '<30').astype(int).values
        num = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
               'number_emergency', 'number_inpatient', 'number_diagnoses']
        df = pd.DataFrame(index=raw.index)
        df['age'] = raw['age'].str.extract(r'\[(\d+)-')[0].astype(float) + 5.0
        for c in num[1:]:
            df[c] = raw[c].astype(float)
        cat = []
        for c in ('race', 'gender', 'medical_specialty', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id'):
            df[c] = _rare(raw[c].fillna('Missing').astype(str))
            cat.append(c)
        for c in ('diag_1', 'diag_2', 'diag_3'):
            df[c] = raw[c].map(_icd)
            cat.append(c)
        for c in ('max_glu_serum', 'A1Cresult'):
            df[c] = raw[c].fillna('None').astype(str)
            cat.append(c)
        meds = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
                'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
        for c in meds:
            if c in raw and (raw[c] != 'No').mean() >= MEDS_MIN:
                df[c] = raw[c].astype(str)
                cat.append(c)
        for c in ('change', 'diabetesMed'):
            df[c] = raw[c].astype(str)
            cat.append(c)
        AUDIT[ds] = [n0, n1, n2, int(yall.sum()), len(df), int(y.sum()), int(df.duplicated().sum())]
    elif ds == 'cdc':
        try:
            raw = pd.read_csv(A2._find('diabetes_binary_health_indicators_BRFSS2015.csv'))
        except FileNotFoundError:
            raw = _uci(891)
        raw = raw.drop(columns=[c for c in raw.columns if c.lower() == 'id'])
        n0, ydup = len(raw), int(raw.duplicated().sum())
        yall = raw['Diabetes_binary'].astype(int).values
        idx = train_test_split(np.arange(n0), train_size=N_LARGE, stratify=yall, random_state=A2.SEED)[0]
        raw = raw.iloc[np.sort(idx)].reset_index(drop=True)
        y = raw.pop('Diabetes_binary').astype(int).values
        df = raw.astype(float)
        cat = []
        AUDIT[ds] = [n0, ydup, int(yall.sum()), len(df), int(y.sum()), int(df.assign(_y=y).duplicated().sum())]
    else:
        raise ValueError(ds)
    num = [c for c in df.columns if c not in cat]
    for c in num:
        df[c] = df[c].astype(float)
    for c in cat:
        df[c] = df[c].astype(object)
    cats = {c: sorted(df[c].dropna().unique().tolist()) for c in cat}
    A2._CACHE[key] = (df, y, num, cat, cats)
    pd.to_pickle(A2._CACHE[key], pk)
    return A2._CACHE[key]


def load_any(ds):
    return load_large(ds) if ds in LARGE else A2.load(ds)


# ---------------------------------------------------------------- TabPFN v2
_PFN = {}


def pfn_predict(Xa, ya, Xbs, seed, ctx):
    """TabPFN v2 (open v2 classifier weights), default settings, on CPU. Training sets larger than ctx rows are
    replaced by a stratified random subsample of ctx rows (the context). Test rows are predicted in chunks."""
    os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET', '1')
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    if len(ya) > ctx:
        keep = train_test_split(np.arange(len(ya)), train_size=ctx, stratify=ya, random_state=seed)[0]
        Xa, ya = Xa[np.sort(keep)], ya[np.sort(keep)]
    kw = dict(device='cpu', random_state=seed, ignore_pretraining_limits=True)
    try:
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, **kw)
    except TypeError:
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        clf.set_params(**{k: v for k, v in kw.items() if k in clf.get_params()})
    clf.fit(Xa.astype(np.float32), ya)
    out = []
    for Xb in Xbs:
        out.append(np.concatenate([clf.predict_proba(Xb[i:i + 2000].astype(np.float32))[:, 1]
                                   for i in range(0, len(Xb), 2000)]).astype(np.float64))
    return out


# ---------------------------------------------------------------- tuned TabNet and FT-Transformer
def deep_space(kind, t):
    if kind == 'tabnet':
        return dict(nd=t.suggest_categorical('nd', [8, 16, 24, 32, 64]), steps=t.suggest_int('steps', 3, 7),
                    gamma=t.suggest_float('gamma', 1.0, 2.0), lr=t.suggest_float('lr', 5e-3, 3e-2, log=True),
                    lam=t.suggest_float('lam', 1e-6, 1e-2, log=True))
    return dict(blocks=t.suggest_int('blocks', 1, 3), dt=t.suggest_categorical('dt', [64, 96, 128, 192]),
                ffr=t.suggest_categorical('ffr', [1.0, 4.0 / 3.0, 2.0]), adrop=t.suggest_float('adrop', 0.0, 0.5),
                fdrop=t.suggest_float('fdrop', 0.0, 0.5), lr=t.suggest_float('lr', 1e-5, 1e-3, log=True),
                wd=t.suggest_float('wd', 1e-6, 1e-3, log=True))


def fit_deep(kind, P, Xtr, ytr, Xbs, seed, E, pat):
    """Train with the given hyperparameters; early stopping on an internal stratified 15% split (as in aagb2)."""
    torch.manual_seed(seed)
    rs = np.random.RandomState(seed)
    itr, iva = A2._split(ytr, seed)
    Xa, ya, Xv, yv = A2._T(Xtr[itr]), A2._T(ytr[itr]), A2._T(Xtr[iva]), A2._T(ytr[iva])
    d, n = Xtr.shape[1], len(itr)
    if kind == 'tabnet':
        net = A2.TabNet(d, n_d=P['nd'], n_a=P['nd'], steps=P['steps'], gamma=P['gamma'])
        opt, bs, lam = torch.optim.Adam(net.parameters(), lr=P['lr']), min(256, n), P['lam']
    else:
        net = A2.FTT(d, dt=P['dt'], blocks=P['blocks'], ffn=int(round(P['dt'] * P['ffr'])), adrop=P['adrop'], fdrop=P['fdrop'])
        opt, bs, lam = torch.optim.AdamW(net.parameters(), lr=P['lr'], weight_decay=P['wd']), (64 if n < 2000 else 256), 0.0

    def fwd(x):
        o = net(x)
        return o if isinstance(o, tuple) else (o, 0.0)

    best, state, bad = float('inf'), None, 0
    for ep in range(E):
        net.train()
        for b in A2._batches(rs, n, bs):
            bt = torch.from_numpy(b)
            z, reg = fwd(Xa[bt])
            loss = F.binary_cross_entropy_with_logits(z, ya[bt]) + lam * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(F.binary_cross_entropy_with_logits(fwd(Xv)[0], yv))
        if vl < best - 1e-5:
            best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        return [torch.sigmoid(fwd(A2._T(X))[0]).numpy().astype(np.float64) for X in Xbs]


def tune_deep(kind, D, tr, ytr, Xtr, Xte, s0, C):
    """Seeded TPE search with the same number of trials as the other models (C['trials']). Because a 3-fold
    inner loop for the deep models was not affordable on CPU, each trial is scored by PR-AUC on one stratified
    20% inner hold-out of the outer training rows; the threshold is chosen on that hold-out."""
    a, b = train_test_split(np.arange(len(ytr)), test_size=0.2, stratify=ytr, random_state=s0)
    Xa, Xb = A2.prep(D, tr[a], tr[b])
    E, pat = (3, 2) if C.get('quick') else C['deep_epochs'][kind]
    store = {}

    def obj(t):
        P = deep_space(kind, t)
        pb = fit_deep(kind, P, Xa, ytr[a], [Xb], s0, E, pat)[0]
        store[t.number] = pb
        return float(average_precision_score(ytr[b], pb))

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=s0))
    st.optimize(obj, n_trials=C['trials'])
    thr = A2.best_thr(ytr[b], store[st.best_trial.number])
    p = fit_deep(kind, st.best_params, Xtr, ytr, [Xte], s0, E, pat)[0]
    return p, thr, st.best_params


# ---------------------------------------------------------------- TabNet-feature stacking (Xia et al. style)
def tn_feats(net, X):
    """Aggregate TabNet feature masks (n x d, input space) and decision embedding (n x n_d, hidden units)."""
    with torch.no_grad():
        x = net.bn(A2._T(X))
        prior, att, res, Magg = torch.ones_like(x), net.init(x)[:, net.n_d:], 0.0, 0.0
        for s in range(net.steps):
            M = A2.sparsemax(net.abn[s](net.afc[s](att)) * prior)
            prior = prior * (net.gamma - M)
            o = net.ft[s](M * x)
            dcs = F.relu(o[:, :net.n_d])
            res = res + dcs
            Magg = Magg + M * dcs.sum(1, keepdim=True)
            att = o[:, net.n_d:]
        Magg = Magg / Magg.sum(1, keepdim=True).clamp_min(1e-12)
    return Magg.numpy().astype(np.float64), res.numpy().astype(np.float64)


def tn_stack(Xtr, ytr, Xte, seed, crossfit, quick):
    net = A2.fit_torch('tabnet', Xtr, ytr, Xte[:2], seed, quick)[3]
    Mte, Ete = tn_feats(net, Xte)
    if not crossfit:
        Mtr, Etr = tn_feats(net, Xtr)
        return Mtr, Etr, Mte, Ete
    Mtr, Etr = np.zeros((len(ytr), Xtr.shape[1])), np.zeros((len(ytr), Ete.shape[1]))
    for k, (i_fit, i_hold) in enumerate(StratifiedKFold(A2.K_CF, shuffle=True, random_state=seed).split(Xtr, ytr)):
        nk = A2.fit_torch('tabnet', Xtr[i_fit], ytr[i_fit], Xte[:2], seed * 31 + k + 1, quick)[3]
        Mtr[i_hold], Etr[i_hold] = tn_feats(nk, Xtr[i_hold])
    return Mtr, Etr, Mte, Ete


# ---------------------------------------------------------------- jobs
def _row(name, yte, p, thr, secs):
    r = dict(model=name, thr=thr, secs=secs)
    r.update(A2.tfree(yte, p))
    r.update(A2.tdep(yte, p, 0.5, '5'))
    r.update(A2.tdep(yte, p, thr, 't'))
    return r


def job_small(ds, rep, fold, C):
    """One outer fold on a small dataset: XGBoost and AAGB-IG (as in aagb2), TabPFN, tuned TabNet and
    FT-Transformer, and the TabNet-feature stacking leakage audit with fixed XGBoost hyperparameters."""
    A2._setup()
    torch.set_num_threads(1)
    q = C.get('quick', False)
    rows, out = [], dict(ds=ds, rep=rep, fold=fold)
    models = C.get('small_models', ['xgb', 'aagb_ig', 'tabpfn', 'tabnet_t', 'ftt_t'])
    old = [m for m in models if m in ('xgb', 'aagb_ig')]
    if old:
        r = A2.job_main(ds, 'main', rep, fold, old, dict(C, imb=False, explain=False))
        for z in r['rows']:
            rows.append({k: z[k] for k in ['model', 'thr', 'secs'] + [k for k in FMK if k in z and k not in ('thr', 'secs')]})
    D, tr, te, Xtr, ytr, Xte, yte, s0 = A2._fold_data(ds, 'main', rep, fold, C['R'])
    if 'tabpfn' in models:
        t0 = time.time()
        oof = np.zeros(len(ytr))
        for k, (a, b) in enumerate(StratifiedKFold(A2.K_IN, shuffle=True, random_state=s0).split(Xtr, ytr)):
            Xa, Xb = A2.prep(D, tr[a], tr[b])
            oof[b] = pfn_predict(Xa, ytr[a], [Xb], s0 + k, C['ctx_small'])[0]
        p = pfn_predict(Xtr, ytr, [Xte], s0, C['ctx_small'])[0]
        rows.append(_row('tabpfn', yte, p, A2.best_thr(ytr, oof), time.time() - t0))
    for m in ('tabnet_t', 'ftt_t'):
        if m in models:
            t0 = time.time()
            p, thr, bp = tune_deep(m.split('_')[0], D, tr, ytr, Xtr, Xte, s0, C)
            rows.append(_row(m, yte, p, thr, time.time() - t0))
    if C.get('leak', True):
        def run(name, Xa, Xb):
            p = A2.fp('xgb', A2.FIX, Xa, ytr, Xb, s0, quick=q)[0]
            r = _row(name, yte, p, 0.5, 0.0)
            rows.append(r)
        run('L:xgb', Xtr, Xte)
        for cf in (False, True):
            Mtr, Etr, Mte, Ete = tn_stack(Xtr, ytr, Xte, s0, cf, q)
            sfx = '_cf' if cf else '_ins'
            run('L:tn_mask' + sfx, np.hstack([Xtr, Mtr]), np.hstack([Xte, Mte]))
            run('L:tn_emb' + sfx, np.hstack([Xtr, Etr]), np.hstack([Xte, Ete]))
    out['rows'] = rows
    return out


def job_large(ds, rep, fold, C):
    """One outer fold on a larger dataset. Hyperparameters are tuned with the same TPE budget on one stratified
    20% inner hold-out (instead of 3 inner folds); the threshold is chosen on that hold-out."""
    A2._setup()
    torch.set_num_threads(1)
    q = C.get('quick', False)
    D = load_large(ds)
    y = D[1]
    tr, te = A2.outer_splits(y, C['R'])[rep * K_OUT + fold]
    Xtr, Xte = A2.prep(D, tr, te)
    ytr, yte, s0 = y[tr], y[te], 1000 * (rep + 1) + 10 * fold
    a, b = train_test_split(np.arange(len(ytr)), test_size=0.2, stratify=ytr, random_state=s0)
    Xa, Xb = A2.prep(D, tr[a], tr[b])
    inner = [(Xa, ytr[a], Xb, ytr[b], b)]
    rows = []
    for model in C.get('large_models', ['rf', 'xgb', 'lgbm', 'cat', 'mlp', 'tabpfn', 'aagb_rm', 'aagb_ig', 'aagb_ig_ins']):
        t0 = time.time()
        if model == 'tabpfn':
            pb = pfn_predict(Xa, ytr[a], [Xb], s0, C['ctx_large'])[0]
            p = pfn_predict(Xtr, ytr, [Xte], s0, C['ctx_large'])[0]
            rows.append(_row(model, yte, p, A2.best_thr(ytr[b], pb), time.time() - t0))
            continue
        if model.startswith('aagb'):
            g = dict(A2.GATE_CFG['aagb_rm' if model == 'aagb_rm' else 'aagb_ig'], quick=q)
            if model == 'aagb_ig_ins':
                g['crossfit'] = False
            inn, Xa_o, Xb_o, info = A2.aagb_inputs(inner, Xtr, ytr, Xte, g, s0)
            base = 'xgb'
        else:
            inn, Xa_o, Xb_o, base = inner, Xtr, Xte, model
        f = lambda P, Xa_, ya_, Xb_, sd, _b=base: A2.fp(_b, P, Xa_, ya_, Xb_, sd, quick=q)[0]
        bp, oof = A2.tune(base, inn, f, s0, C['trials'], len(ytr))
        p = A2.fp(base, bp, Xa_o, ytr, Xb_o, s0, quick=q)[0]
        rows.append(_row(model, yte, p, A2.best_thr(ytr[b], oof[b]), time.time() - t0))
    return dict(ds=ds, rep=rep, fold=fold, rows=rows)


# ---------------------------------------------------------------- orchestration
def emit_rows(res):
    for r in res['rows']:
        key = '%s:%d:%d' % (r['model'], res['rep'], res['fold'])
        if r['model'].startswith('L:'):  # leakage audit: threshold-free metrics only
            A2.emit(res['ds'], 'LK', key, [r['roc'], r['pr'], r['brier']])
        else:
            A2.emit(res['ds'], 'FM', key, [r.get(k, np.nan) for k in FMK[:-1]] + [r.get('secs', np.nan) / 1e4 * 10])


def run3(jobs, C):
    """jobs: list of (dataset, rep, fold). Results are printed as soon as each outer fold finishes."""
    import sklearn, platform, xgboost, lightgbm, catboost, imblearn
    try:
        import tabpfn
        tv = tabpfn.__version__
    except Exception as e:
        tv = 'NA(%s)' % type(e).__name__
    print('@@VER|py=%s;np=%s;pd=%s;sk=%s;xgb=%s;lgb=%s;cat=%s;torch=%s;optuna=%s;imblearn=%s;tabpfn=%s' % (
        platform.python_version(), np.__version__, pd.__version__, sklearn.__version__, xgboost.__version__,
        lightgbm.__version__, catboost.__version__, torch.__version__, optuna.__version__, imblearn.__version__, tv), flush=True)
    for f in ('aagb2.py', 'aagb3.py'):
        print('@@HASH|%s|%s' % (f, A2.module_hash(os.path.join(A2.OUT, f))), flush=True)
    for ds in dict.fromkeys(j[0] for j in jobs):
        D = load_any(ds)
        names, groups = A2.feat_groups(D)
        A2.emit(ds, 'INFO3', 'cfg', [C['R_large'] if ds in LARGE else C['R'], C['trials'], len(D[1]), int(D[1].sum()), len(names), len(set(groups))], 1)
        if ds in AUDIT:
            A2.emit(ds, 'AUD3', 'base', AUDIT[ds], 1)
            A2.emit(ds, 'VARS3', 'names', ';'.join(dict.fromkeys(groups)))
    T0 = time.time()
    fn = lambda ds, r, f: job_large(ds, r, f, dict(C, R=C['R_large'])) if ds in LARGE else job_small(ds, r, f, C)
    try:
        it = Parallel(n_jobs=C['n_jobs'], return_as='generator_unordered')(delayed(fn)(*j) for j in jobs)
    except Exception:
        it = Parallel(n_jobs=C['n_jobs'])(delayed(fn)(*j) for j in jobs)
    for i, res in enumerate(it):
        emit_rows(res)
        print('[%s] %d/%d %s r%d f%d %.0fs' % (time.strftime('%H:%M:%S'), i + 1, len(jobs), res['ds'], res['rep'], res['fold'], time.time() - T0), flush=True)
    print('@@ALLDONE3|%.0f' % (time.time() - T0), flush=True)


def time_deep(ds, C, kinds=('ftt', 'tabnet')):
    """Seconds per training epoch of a mid-sized configuration on the first outer fold (for planning only)."""
    A2._setup()
    D, tr, te, Xtr, ytr, Xte, yte, s0 = A2._fold_data(ds, 'main', 0, 0, C['R'])
    P = dict(ftt=dict(blocks=2, dt=128, ffr=4.0 / 3.0, adrop=0.2, fdrop=0.1, lr=1e-4, wd=1e-5),
             tabnet=dict(nd=16, steps=4, gamma=1.3, lr=2e-2, lam=1e-4))
    for k in kinds:
        t0 = time.time()
        fit_deep(k, P[k], Xtr, ytr, [Xte], s0, 3, 99)
        print('@@TIME|%s|%s|%.2f' % (ds, k, (time.time() - t0) / 3), flush=True)
