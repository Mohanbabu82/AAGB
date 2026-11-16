# AAGB v2 - supplementary experiments (imports aagb2.py): hidden-unit attention baseline in the style of
# AttnBoost, diagnostics for sparsity-penalised and row-normalised input gates, seed-only explanation stability.
import os, sys, time, itertools
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import StratifiedKFold
import aagb2 as A2

BASE_GATE = A2.GateNet
XG = {'ig_l1a': dict(gate='input', l1=0.01), 'ig_l1b': dict(gate='input', l1=0.05),
      'ig_sm': dict(gate='input', l1=0.0, lent=0.0, norm=True)}
XV = ['xgb', 'attnb_ins', 'attnb_cf', 'full_ig_l1a', 'full_ig_l1b', 'full_ig_sm']


class NormGate(A2.GateNet):
    """Input-level gate with row-normalised feature attention: a_f = d * softmax(W_f^T h + b_f), so the
    attention of one row has mean 1 and features compete for a fixed budget."""

    def attention(self, x):
        h = self.enc(x)
        A = x.shape[1] * torch.softmax(self.fa(h), 1)
        a = torch.sigmoid(self.sa(h)).squeeze(1) if self.use_sa else torch.ones(x.shape[0])
        return h, A, a


class HidNet(nn.Module):
    """Hidden-unit attention as described for AttnBoost: h = ReLU(W1 x + b1), alpha = sigmoid(Wa h + ba),
    h_att = alpha * h, logit = w2^T h_att + b2 (k = 128 hidden units)."""

    def __init__(self, d, k=128):
        super().__init__()
        self.l1, self.att, self.out = nn.Linear(d, k), nn.Linear(k, k), nn.Linear(k, 1)

    def rep(self, x):
        h = F.relu(self.l1(x))
        return torch.sigmoid(self.att(h)) * h

    def forward(self, x):
        return self.out(self.rep(x)).squeeze(1)


def train_hid(X, y, seed, quick=False):
    """Same optimiser, schedule and early stopping as the AAGB attention module; loss = BCE."""
    torch.manual_seed(seed)
    rs = np.random.RandomState(seed)
    net = HidNet(X.shape[1])
    itr, iva = A2._split(y, seed)
    Xa, ya, Xv, yv = A2._T(X[itr]), A2._T(y[itr]), A2._T(X[iva]), A2._T(y[iva])
    E = 5 if quick else 80
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
    best, state, bad, n = float('inf'), None, 0, len(itr)
    for ep in range(E):
        net.train()
        for b in A2._batches(rs, n, min(256, n)):
            bt = torch.from_numpy(b)
            loss = F.binary_cross_entropy_with_logits(net(Xa[bt]), ya[bt])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float(F.binary_cross_entropy_with_logits(net(Xv), yv))
        if vl < best - 1e-5:
            best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    return net


def hid_rep(net, X):
    with torch.no_grad():
        return net.rep(A2._T(X)).numpy().astype(np.float64)


def hid_feats(Xtr, ytr, Xte, seed, crossfit, quick=False):
    final = train_hid(Xtr, ytr, seed, quick)
    H_te = hid_rep(final, Xte)
    if not crossfit:
        return hid_rep(final, Xtr), H_te
    H_tr = np.zeros((len(ytr), H_te.shape[1]))
    for k, (i_fit, i_hold) in enumerate(StratifiedKFold(A2.K_CF, shuffle=True, random_state=seed).split(Xtr, ytr)):
        H_tr[i_hold] = hid_rep(train_hid(Xtr[i_fit], ytr[i_fit], seed * 31 + k + 1, quick), Xtr[i_hold])
    return H_tr, H_te


def gate_feats(Xtr, ytr, Xte, g, seed):
    """att_feats of the main pipeline; the row-normalised variant swaps the module class for the call."""
    if g.get('norm'):
        A2.GateNet = NormGate
    try:
        return A2.att_feats(Xtr, ytr, Xte, g, seed)
    finally:
        A2.GateNet = BASE_GATE


def job_extra(ds, rep, fold, C):
    A2._setup()
    q = C.get('quick', False)
    D, tr, te, Xtr, ytr, Xte, yte, s0 = A2._fold_data(ds, 'main', rep, fold, C['R'])
    G, gn = A2.group_matrix(A2.feat_groups(D)[1])
    d, mu, rs = Xtr.shape[1], Xtr.mean(0), np.random.RandomState(s0 + 5)
    rows, diag = [], {}

    def run(name, Xa, Xb):
        p, m = A2.fp('xgb', A2.FIX, Xa, ytr, Xb, s0, quick=q)
        r = dict(variant=name, rep=rep, fold=fold)
        r.update(A2.tfree(yte, p))
        r.update(A2.tdep(yte, p, 0.5, '5'))
        rows.append(r)
        return m

    run('xgb', Xtr, Xte)
    for key, cf in (('attnb_ins', False), ('attnb_cf', True)):
        H_tr, H_te = hid_feats(Xtr, ytr, Xte, s0, cf, q)
        run(key, np.hstack([Xtr, H_tr]), np.hstack([Xte, H_te]))
    for key, g in XG.items():
        A_tr, s_tr, A_te, s_te, net = gate_feats(Xtr, ytr, Xte, dict(g, quick=q), s0)
        pairs = A2.select_pairs(A_tr)
        Xb_aug = A2.build_aug(Xte, A_te, s_te, pairs)
        mdl = run('full_' + key, A2.build_aug(Xtr, A_tr, s_tr, pairs), Xb_aug)
        cm = A_te.mean(0)
        Ag = (A_te @ G.T) / G.sum(1)[None, :]
        pg = A2.shap_groups(mdl, Xb_aug, d, pairs, G)[1]
        dl = {k: A2.deletion(net, mdl, pairs, Xte, mu, G, sc)
              for k, sc in (('top', Ag), ('bot', -Ag), ('rnd', rs.rand(*Ag.shape)), ('shp', np.abs(pg)))}
        diag[key] = dict(att=(cm @ G.T) / G.sum(1), rng=float(cm.max() - cm.min()), mean=float(A_te.mean()),
                         sat=float((A_te > 0.95).mean()), wsd=float(A_te.std(1).mean()), asd=float(A_te.std(0).mean()), dl=dl)
    return dict(rows=rows, diag=diag)


def job_seed(ds, fold, C, seeds=(11, 22, 33, 44, 55)):
    """Seed-only stability on one outer fold of repetition 0: the data partition is fixed and only the random
    seed changes. Returns per-variable test attention for three gates and aggregated SHAP for two models."""
    A2._setup()
    q = C.get('quick', False)
    D, tr, te, Xtr, ytr, Xte, yte, s0 = A2._fold_data(ds, 'main', 0, fold, C['R'])
    G, gn = A2.group_matrix(A2.feat_groups(D)[1])
    d = Xtr.shape[1]
    out = {k: [] for k in ('ig', 'rm', 'pj', 'shap_xgb', 'shap_ig')}
    for s in seeds:
        sd = s0 + s
        for key in ('ig', 'rm', 'pj'):
            net = A2.train_gate(Xtr, ytr, dict(A2.GATES[key], quick=q), sd)
            out[key].append((A2.infer(net, Xte)[0].mean(0) @ G.T) / G.sum(1))
        m0 = A2.fp('xgb', A2.FIX, Xtr, ytr, Xte, sd, quick=q)[1]
        out['shap_xgb'].append(A2.shap_groups(m0, Xte, d, [], G, has_w=False)[0])
        A_tr, s_tr, A_te, s_te, net = A2.att_feats(Xtr, ytr, Xte, dict(gate='input', quick=q), sd)
        pairs = A2.select_pairs(A_tr)
        Xb = A2.build_aug(Xte, A_te, s_te, pairs)
        m1 = A2.fp('xgb', A2.FIX, A2.build_aug(Xtr, A_tr, s_tr, pairs), ytr, Xb, sd, quick=q)[1]
        out['shap_ig'].append(A2.shap_groups(m1, Xb, d, pairs, G)[0])
    return {k: A2.stability(np.vstack(v))[:2] for k, v in out.items()}


def hardware():
    model, mem = 'unknown', 0.0
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith('model name'):
                    model = line.split(':', 1)[1].strip()
                    break
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemTotal'):
                    mem = float(line.split()[1]) / 1048576.0
                    break
    except Exception:
        pass
    return 'cpus=%d;model=%s;mem_gb=%.1f' % (os.cpu_count() or 0, model, mem)


def run_extra(datasets, C):
    print('@@HW|' + hardware(), flush=True)
    print('@@HASH|' + A2.module_hash(os.path.join(A2.OUT, 'aagb2.py')), flush=True)
    print('@@HASHX|' + A2.module_hash(os.path.join(A2.OUT, 'aagb2x.py')), flush=True)
    for ds in datasets:
        T0 = time.time()
        od = os.path.join(A2.OUT, ds)
        os.makedirs(od, exist_ok=True)
        R = C['R']
        y = A2.load(ds)[1]
        folds = [(r, f) for r in range(R) for f in range(A2.K_OUT)]
        res = A2._par(job_extra, [(ds, r, f, C) for r, f in folds], C, ds + ' extra')
        da = pd.DataFrame([row for z in res for row in z['rows']])
        da.to_csv(os.path.join(od, 'fold_metrics_extra.csv'), index=False)
        piv = {k: A2._pivot(da, 'variant', k) for k in ('roc', 'pr', 'brier')}
        for v in XV:
            A2.emit(ds, 'XB', v, [np.nanmean(piv[k][v].values) for k in ('roc', 'pr', 'brier')])
            vals = []
            for k in ('roc', 'pr', 'brier'):
                dlt = (piv[k][v] - piv[k]['xgb']).values
                vals += [np.nanmean(dlt), np.nanstd(dlt, ddof=1) if len(dlt) > 1 else 0.0]
            A2.emit(ds, 'XBD', v, vals, 1e6)
        for g in XG:
            dg = [z['diag'][g] for z in res]
            A = np.vstack([e['att'] for e in dg])
            sp, jc, rk = A2.stability(A)
            A2.emit(ds, 'XGA', g, A.mean(0))
            A2.emit(ds, 'XGD', g, [np.mean([e[k] for e in dg]) for k in ('rng', 'mean', 'sat', 'wsd', 'asd')] + [sp, jc, A.max(1).mean() - A.min(1).mean()])
            dl = {k: np.array([e['dl'][k] for e in dg]) for k in ('top', 'bot', 'rnd', 'shp')}
            A2.emit(ds, 'XGDL', g, np.concatenate([dl[k].mean(0) for k in ('top', 'bot', 'rnd', 'shp')]))
            dd = dl['top'] - dl['rnd']
            A2.emit(ds, 'XGDT', g, np.column_stack([dd.mean(0), dd.std(0, ddof=1) if len(dd) > 1 else np.zeros(3)]).ravel(), 1e6)
            np.save(os.path.join(od, 'attention_by_variable_%s.npy' % g), A)
        sres = A2._par(job_seed, [(ds, f, C) for f in range(A2.K_OUT)], C, ds + ' seed')
        for k in ('ig', 'rm', 'pj', 'shap_xgb', 'shap_ig'):
            A2.emit(ds, 'SEED', k, [np.nanmean([z[k][0] for z in sres]), np.nanmean([z[k][1] for z in sres])])
        A2.emit(ds, 'XDONE', 'secs', [time.time() - T0], 1)
        with open(os.path.join(od, 'report_lines_extra.txt'), 'w') as fh:
            fh.write(chr(10).join(l for l in A2._LINES if l.startswith('@@' + ds + '|')) + chr(10))
    print('@@XALLDONE', flush=True)
