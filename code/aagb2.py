# AAGB v2 - revised experimental pipeline (part 1: data, preprocessing, metrics, statistics)
import os, sys, json, time, math, zlib, hashlib, itertools, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
import xgboost as xgb, lightgbm as lgb, optuna
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
optuna.logging.set_verbosity(optuna.logging.ERROR)

K_OUT, K_IN, K_CF, SEED = 5, 3, 5, 42
OUT = '/kaggle/working'
_CACHE = {}


def _find(name):
    for dn, _, fns in os.walk('/kaggle/input'):
        if name in fns:
            return os.path.join(dn, name)
    raise FileNotFoundError(name)


def load(ds, variant='main'):
    """Return (frame, y, numeric columns, categorical columns, category vocabularies)."""
    key = (ds, variant)
    if key in _CACHE:
        return _CACHE[key]
    cat = []
    if ds == 'heart':
        nominal = ['cp', 'restecg', 'slope', 'thal']
        if variant == 'main':  # curated Cleveland file, 297 complete records, condition = disease present
            df = pd.read_csv(_find('heart_cleveland_upload.csv'))
            y = df.pop('condition').astype(int).values
            cat = nominal
        else:  # redistributed heart.csv used in the first submission
            df = pd.read_csv(_find('heart.csv'))
            if variant == 'redist_fix':
                df = df.drop_duplicates().reset_index(drop=True)
                df['ca'] = df['ca'].astype(float).where(df['ca'] != 4, np.nan)
                df['thal'] = df['thal'].astype(float).where(df['thal'] != 0, np.nan)
                cat = nominal
            y = df.pop('target').astype(int).values
        for c in cat:
            df[c] = df[c].map(lambda v: np.nan if pd.isna(v) else str(int(v)))
    elif ds == 'pima':
        df = pd.read_csv(_find('diabetes.csv'))
        y = df.pop('Outcome').astype(int).values
        if variant == 'main':  # physiologically impossible zeros are treated as missing
            for c in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
                df[c] = df[c].astype(float).where(df[c] != 0, np.nan)
    elif ds == 'ckd':
        df = pd.read_csv(_find('kidney_disease.csv')).drop(columns=['id'])
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v).replace({'?': np.nan, '': np.nan})
        y = (df.pop('classification') == 'ckd').astype(int).values
        for c in ['pcv', 'wc', 'rc']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        cat = [c for c in df.columns if df[c].dtype == object]
        drop = {'no_sc': ['sc'], 'no_hemo': ['hemo'], 'no_pcv': ['pcv'], 'no_all3': ['sc', 'hemo', 'pcv']}.get(variant, [])
        df = df.drop(columns=drop)
    elif ds == 'stroke':
        df = pd.read_csv(_find('healthcare-dataset-stroke-data.csv')).drop(columns=['id'])
        y = df.pop('stroke').astype(int).values
        cat = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    else:
        raise ValueError(ds)
    num = [c for c in df.columns if c not in cat]
    for c in num:
        df[c] = df[c].astype(float)
    for c in cat:
        df[c] = df[c].astype(object)
    cats = {c: sorted(df[c].dropna().unique().tolist()) for c in cat}
    _CACHE[key] = (df, y, num, cat, cats)
    return _CACHE[key]


def feat_groups(D):
    """Names of the encoded columns and the source variable of each column."""
    df, y, num, cat, cats = D
    names, groups = list(num), list(num)
    for c in cat:
        if len(cats[c]) == 2:
            names.append(c + '=' + str(cats[c][1]))
            groups.append(c)
        else:
            names += [c + '=' + str(v) for v in cats[c]]
            groups += [c] * len(cats[c])
    return names, groups


def group_matrix(groups):
    gn = list(dict.fromkeys(groups))
    G = np.zeros((len(gn), len(groups)))
    for j, g in enumerate(groups):
        G[gn.index(g), j] = 1.0
    return G, gn


def prep(D, tr, te):
    """Imputation, scaling and encoding fitted on the training rows only."""
    df, y, num, cat, cats = D
    a, b = df.iloc[tr], df.iloc[te]
    ni = SimpleImputer(strategy='median').fit(a[num].values)
    sc = StandardScaler().fit(ni.transform(a[num].values))
    A, B = [sc.transform(ni.transform(a[num].values))], [sc.transform(ni.transform(b[num].values))]
    if cat:
        ci = SimpleImputer(strategy='most_frequent').fit(a[cat].values)
        oh = OneHotEncoder(categories=[cats[c] for c in cat], drop='if_binary', handle_unknown='ignore', sparse_output=False)
        oh.fit(ci.transform(a[cat].values))
        A.append(oh.transform(ci.transform(a[cat].values)))
        B.append(oh.transform(ci.transform(b[cat].values)))
    return np.hstack(A).astype(np.float64), np.hstack(B).astype(np.float64)


def outer_splits(y, R):
    return list(RepeatedStratifiedKFold(n_splits=K_OUT, n_repeats=R, random_state=SEED).split(np.zeros(len(y)), y))


def oversample(X, y, all_num, seed):
    """SMOTE for all-numeric data, random oversampling otherwise; skipped when classes are nearly balanced."""
    c = np.bincount(y, minlength=2)
    if c.min() / c.max() > 0.8 or c.min() < 2:
        return X, y
    if all_num:
        return SMOTE(random_state=seed, k_neighbors=int(min(5, c.min() - 1))).fit_resample(X, y)
    return RandomOverSampler(random_state=seed).fit_resample(X, y)


def ece(y, p, bins=10):
    order = np.argsort(p, kind='stable')
    e = 0.0
    for ch in np.array_split(order, bins):
        if len(ch):
            e += len(ch) / len(y) * abs(p[ch].mean() - y[ch].mean())
    return float(e)


def tfree(y, p):
    one = len(np.unique(y)) < 2
    return dict(roc=np.nan if one else float(roc_auc_score(y, p)), pr=np.nan if one else float(average_precision_score(y, p)),
                brier=float(np.mean((p - y) ** 2)), ece=ece(y, p))


def tdep(y, p, thr, sfx):
    pred = p >= thr
    pos = y == 1
    tp, fp_ = int((pred & pos).sum()), int((pred & ~pos).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & ~pos).sum())
    den = 2 * tp + fp_ + fn
    mden = math.sqrt(float(tp + fp_) * float(tp + fn) * float(tn + fp_) * float(tn + fn))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp_) if tn + fp_ else np.nan
    return {'f1' + sfx: 2 * tp / den if den else 0.0, 'mcc' + sfx: (tp * tn - fp_ * fn) / mden if mden else 0.0,
            'sens' + sfx: sens, 'spec' + sfx: spec, 'prec' + sfx: tp / (tp + fp_) if tp + fp_ else np.nan,
            'bacc' + sfx: 0.5 * (sens + spec)}


GRID = np.round(np.arange(0.02, 0.981, 0.01), 2)


def best_thr(y, p):
    """Threshold maximising F1 on out-of-fold training predictions (never on test data)."""
    P = p[None, :] >= GRID[:, None]
    pos = (y == 1)[None, :]
    tp = (P & pos).sum(1)
    fp_ = (P & ~pos).sum(1)
    fn = pos.sum() - tp
    return float(GRID[int(np.argmax(2 * tp / np.maximum(2 * tp + fp_ + fn, 1)))])


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt_fit(y, p):
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(_logit(p)[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_apply(p, ab):
    return 1.0 / (1.0 + np.exp(-(ab[0] * _logit(p) + ab[1])))


def csum(v):
    """Mean, SD and half-width of the corrected resampled 95% CI (Nadeau-Bengio variance, repeated k-fold)."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return (float(v.mean()) if n else np.nan, 0.0, 0.0)
    sd = float(v.std(ddof=1))
    return float(v.mean()), sd, float(stats.t.ppf(0.975, n - 1) * math.sqrt((1.0 / n + 1.0 / (K_OUT - 1)) * sd ** 2))


def stability(M, k=3):
    """Mean pairwise Spearman correlation, mean pairwise top-k Jaccard overlap, mean rank per variable."""
    M = np.asarray(M, float)
    R = np.vstack([stats.rankdata(-r) for r in M])
    tops = [set(np.argsort(-r, kind='stable')[:k].tolist()) for r in M]
    sp, jc = [], []
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            if R[i].std() > 0 and R[j].std() > 0:
                sp.append(np.corrcoef(R[i], R[j])[0, 1])
            jc.append(len(tops[i] & tops[j]) / len(tops[i] | tops[j]))
    return (float(np.mean(sp)) if sp else np.nan, float(np.mean(jc)) if jc else np.nan, R.mean(0))


_LINES = []


def emit(ds, block, key, vals, scale=1e4):
    """Print one checksummed result row: values are integers (value x scale); x marks a missing value."""
    if isinstance(vals, str):
        body = vals
    else:
        body = ','.join('x' if (v is None or not np.isfinite(float(v))) else str(int(round(float(v) * scale))) for v in vals)
    s = '%s|%s|%s|%s' % (ds, block, key, body)
    line = '@@%s|%08x' % (s, zlib.crc32(s.encode()) & 0xffffffff)
    _LINES.append(line)
    print(line, flush=True)


def audit(ds):
    """Data-integrity facts reported in the manuscript."""
    out = {}
    if ds == 'heart':
        a = pd.read_csv(_find('heart_cleveland_upload.csv'))
        b = pd.read_csv(_find('heart.csv'))
        out['curated'] = [len(a), int(a.duplicated().sum()), int((a.condition == 1).sum()), int(a.isna().sum().sum())]
        out['redist'] = [len(b), int(b.duplicated().sum()), int((b.target == 1).sum()), int((b.ca == 4).sum()), int((b.thal == 0).sum())]
        out['redist_dup_rows'] = b[b.duplicated(keep=False)].index.tolist()
        key = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak']
        m = a.drop_duplicates(key).merge(b.drop_duplicates(key), on=key, suffixes=('_a', '_b'))
        out['polarity'] = [len(m)] + [int(((m.condition == i) & (m.target == j)).sum()) for i in (0, 1) for j in (0, 1)]
    elif ds == 'pima':
        b = pd.read_csv(_find('diabetes.csv'))
        out['base'] = [len(b), int(b.duplicated().sum()), int(b.Outcome.sum())]
        out['zeros'] = [int((b[c] == 0).sum()) for c in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]
    elif ds == 'ckd':
        df, y = load('ckd')[:2]
        full = df.assign(_y=y)
        out['base'] = [len(df), int(full.duplicated().sum()), int(y.sum()), int(df.notna().all(axis=1).sum()),
                       int(df.isna().sum().max()), int((df.isna().sum() > 0).sum())]
    else:
        df, y = load('stroke')[:2]
        full = df.assign(_y=y)
        out['base'] = [len(df), int(full.duplicated().sum()), int(y.sum()), int(df.bmi.isna().sum()), int((df.gender == 'Other').sum())]
    return out


# ---------------------------------------------------------------- part 2: attention module and deep baselines
class GateNet(nn.Module):
    """Shared encoder with feature-level (A_f) and sample-level (a_s) sigmoid attention and a training-only head.
    gate = 'rowmean' : z = w_a^T (mean_j A_f,ij * a_s,i * h_i) + b_a            (original design)
    gate = 'proj'    : z = w_a^T (h_i * sigmoid(P^T a_f,i) * a_s,i) + b_a       (projection gate)
    gate = 'input'   : z = phi(x_i * a_f,i * a_s,i)                             (input-level gate)"""

    def __init__(self, d, m=64, dropout=0.15, gate='input', use_fa=True, use_sa=True):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, m), nn.BatchNorm1d(m), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(m, m), nn.BatchNorm1d(m), nn.ReLU(), nn.Dropout(dropout))
        self.fa, self.sa = nn.Linear(m, d), nn.Linear(m, 1)
        self.gate, self.use_fa, self.use_sa = gate, use_fa, use_sa
        if gate == 'input':
            self.pred = nn.Sequential(nn.Linear(d, m), nn.ReLU(), nn.Dropout(dropout), nn.Linear(m, 1))
        else:
            self.head = nn.Linear(m, 1)
            if gate == 'proj':
                self.P = nn.Linear(d, m, bias=False)

    def attention(self, x):
        h = self.enc(x)
        A = torch.sigmoid(self.fa(h)) if self.use_fa else torch.ones_like(x)
        a = torch.sigmoid(self.sa(h)).squeeze(1) if self.use_sa else torch.ones(x.shape[0])
        return h, A, a

    def forward(self, x):
        h, A, a = self.attention(x)
        if self.gate == 'rowmean':
            z = self.head(h * (A.mean(1) * a).unsqueeze(1))
        elif self.gate == 'proj':
            z = self.head(h * torch.sigmoid(self.P(A)) * a.unsqueeze(1))
        else:
            z = self.pred(x * A * a.unsqueeze(1))
        return z.squeeze(1), A, a


def _batches(rs, n, bs):
    perm = rs.permutation(n)
    out = [perm[i:i + bs] for i in range(0, n, bs)]
    if len(out) > 1 and len(out[-1]) < 8:
        out[-2] = np.concatenate([out[-2], out[-1]])
        out.pop()
    return out


def _split(y, seed):
    idx = np.arange(len(y))
    try:
        return train_test_split(idx, test_size=0.15, stratify=y, random_state=seed)
    except ValueError:
        return train_test_split(idx, test_size=0.15, random_state=seed)


def _T(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


def train_gate(X, y, cfg, seed):
    """Stage 1: train the attention module (Adam, cosine annealing, early stopping on an internal 15% split)."""
    torch.manual_seed(seed)
    rs = np.random.RandomState(seed)
    net = GateNet(X.shape[1], cfg.get('m', 64), cfg.get('dropout', 0.15), cfg.get('gate', 'input'),
                  cfg.get('use_fa', True), cfg.get('use_sa', True))
    if cfg.get('untrained'):
        net.eval()
        return net
    itr, iva = _split(y, seed)
    Xa, ya = X[itr], y[itr]
    wpos = float((ya == 0).sum() / max((ya == 1).sum(), 1)) if cfg.get('wbce') else 1.0
    if cfg.get('att_os'):
        Xa, ya = oversample(Xa, ya, cfg.get('all_num', False), seed)
    Xa_t, ya_t, Xv_t, yv_t, pw = _T(Xa), _T(ya), _T(X[iva]), _T(y[iva]), torch.tensor(wpos)
    l1, lent, lsa = cfg.get('l1', 1e-3), cfg.get('lent', 5e-3), cfg.get('lsa', 1e-3)

    def loss_fn(z, A, a, yy):
        Ac = A.clamp(1e-6, 1 - 1e-6)
        ent = -(Ac * Ac.log() + (1 - Ac) * (1 - Ac).log()).mean()
        return F.binary_cross_entropy_with_logits(z, yy, pos_weight=pw) + l1 * A.mean() - lent * ent + lsa * ((a - 0.5) ** 2).mean()

    E = 5 if cfg.get('quick') else cfg.get('epochs', 80)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get('lr', 1e-3))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
    best, state, bad, n = float('inf'), None, 0, len(ya)
    for ep in range(E):
        net.train()
        for b in _batches(rs, n, min(256, n)):
            bt = torch.from_numpy(b)
            z, A, a = net(Xa_t[bt])
            loss = loss_fn(z, A, a, ya_t[bt])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            z, A, a = net(Xv_t)
            vl = float(loss_fn(z, A, a, yv_t))
        if vl < best - 1e-5:
            best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.get('patience', 15):
                break
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    return net


def infer(net, X):
    with torch.no_grad():
        _, A, a = net.attention(_T(X))
    return A.numpy().astype(np.float64), a.numpy().astype(np.float64)


def att_feats(Xtr, ytr, Xte, cfg, seed):
    """Attention for training and test rows. With cross-fitting, the attention of every training row comes
    from a module that never saw that row (K_CF folds); test rows use the module fitted on all training rows."""
    final = train_gate(Xtr, ytr, cfg, seed)
    A_te, a_te = infer(final, Xte)
    if cfg.get('crossfit', True) and not cfg.get('untrained'):
        A_tr, a_tr = np.zeros_like(Xtr), np.zeros(len(ytr))
        for k, (i_fit, i_hold) in enumerate(StratifiedKFold(K_CF, shuffle=True, random_state=seed).split(Xtr, ytr)):
            net = train_gate(Xtr[i_fit], ytr[i_fit], cfg, seed * 31 + k + 1)
            A_tr[i_hold], a_tr[i_hold] = infer(net, Xtr[i_hold])
    else:
        A_tr, a_tr = infer(final, Xtr)
    return A_tr, a_tr, A_te, a_te, final


def select_pairs(A_tr, k=5):
    """All pairs among the k features with the highest mean training attention, in canonical (sorted) order."""
    top = np.argsort(-A_tr.mean(0), kind='stable')[:k]
    return sorted((int(min(p, q)), int(max(p, q))) for p, q in itertools.combinations(top, 2))


def build_aug(X, A, a, pairs, use_w=True, use_int=True):
    """Stage 2 input: [X | X * A_f * a_s | products of the selected pairs]."""
    parts = [X]
    if use_w:
        parts.append(X * A * a[:, None])
    if use_int and len(pairs):
        parts.append(np.column_stack([X[:, p] * X[:, q] for p, q in pairs]))
    return np.hstack(parts)


def grad_check(seed=0, n=6, d=5, m=8):
    """Largest within-row spread of dL_BCE/dA_f for the three gates (zero means no feature-specific gradient)."""
    g = torch.Generator().manual_seed(seed)

    def R(*s):
        return torch.randn(*s, generator=g, dtype=torch.float64)

    x, h, wa, P, W1, w2 = R(n, d), R(n, m), R(m), R(d, m), R(d, m), R(m)
    y = (R(n) > 0).double()
    out = {}
    for gate in ('rowmean', 'proj', 'input'):
        A = torch.sigmoid(R(n, d)).requires_grad_(True)
        a = torch.sigmoid(R(n))
        if gate == 'rowmean':
            z = (h * (A.mean(1) * a)[:, None]) @ wa
        elif gate == 'proj':
            z = (h * torch.sigmoid(A @ P) * a[:, None]) @ wa
        else:
            z = torch.tanh((x * A * a[:, None]) @ W1) @ w2
        gr, = torch.autograd.grad(F.binary_cross_entropy_with_logits(z, y), A)
        out[gate] = float((gr.max(1).values - gr.min(1).values).max())
    return out


def sparsemax(z):
    zs, _ = torch.sort(z, dim=-1, descending=True)
    k = torch.arange(1, z.shape[-1] + 1, dtype=z.dtype)
    cs = zs.cumsum(-1)
    ksup = ((1 + k * zs) > cs).sum(-1, keepdim=True)
    tau = (cs.gather(-1, ksup - 1) - 1) / ksup.to(z.dtype)
    return torch.clamp(z - tau, min=0)


class GBN(nn.Module):
    """Ghost batch normalisation."""

    def __init__(self, d, vbs=128, mom=0.02):
        super().__init__()
        self.bn, self.vbs = nn.BatchNorm1d(d, momentum=mom), vbs

    def forward(self, x):
        if not self.training or x.shape[0] <= self.vbs:
            return self.bn(x)
        return torch.cat([self.bn(c) for c in x.chunk(int(math.ceil(x.shape[0] / self.vbs)), 0)], 0)


class GLU(nn.Module):
    def __init__(self, i, o, fc=None):
        super().__init__()
        self.fc = fc if fc is not None else nn.Linear(i, 2 * o, bias=False)
        self.bn, self.o = GBN(2 * o), o

    def forward(self, x):
        x = self.bn(self.fc(x))
        return x[:, :self.o] * torch.sigmoid(x[:, self.o:])


class FeatT(nn.Module):
    def __init__(self, i, o, shared, n_ind=2):
        super().__init__()
        self.sh = nn.ModuleList([GLU(i if j == 0 else o, o, fc) for j, fc in enumerate(shared)])
        self.ind = nn.ModuleList([GLU(o, o) for _ in range(n_ind)])

    def forward(self, x):
        s = math.sqrt(0.5)
        x = self.sh[0](x)
        for g in list(self.sh)[1:] + list(self.ind):
            x = (x + g(x)) * s
        return x


class TabNet(nn.Module):
    """TabNet encoder and classification head following Arik and Pfister, with the reference default sizes
    (n_d = n_a = 8, three decision steps, gamma = 1.3, two shared and two step-specific GLU blocks)."""

    def __init__(self, d, n_d=8, n_a=8, steps=3, gamma=1.3):
        super().__init__()
        self.n_d, self.steps, self.gamma = n_d, steps, gamma
        self.bn = nn.BatchNorm1d(d, momentum=0.01)
        shared = [nn.Linear(d, 2 * (n_d + n_a), bias=False), nn.Linear(n_d + n_a, 2 * (n_d + n_a), bias=False)]
        self.init = FeatT(d, n_d + n_a, shared)
        self.ft = nn.ModuleList([FeatT(d, n_d + n_a, shared) for _ in range(steps)])
        self.afc = nn.ModuleList([nn.Linear(n_a, d, bias=False) for _ in range(steps)])
        self.abn = nn.ModuleList([GBN(d) for _ in range(steps)])
        self.out = nn.Linear(n_d, 1, bias=False)

    def forward(self, x):
        x = self.bn(x)
        prior, att, res, ent = torch.ones_like(x), self.init(x)[:, self.n_d:], 0.0, 0.0
        for s in range(self.steps):
            M = sparsemax(self.abn[s](self.afc[s](att)) * prior)
            ent = ent + (-M * torch.log(M + 1e-15)).sum(1).mean()
            prior = prior * (self.gamma - M)
            o = self.ft[s](M * x)
            res = res + F.relu(o[:, :self.n_d])
            att = o[:, self.n_d:]
        return self.out(res).squeeze(1), ent / self.steps


class FTT(nn.Module):
    """FT-Transformer following Gorishniy et al. (default configuration: 3 pre-norm blocks, token size 192,
    8 heads, ReGLU feed-forward of width 256, attention dropout 0.2, feed-forward dropout 0.1)."""

    def __init__(self, d, dt=192, blocks=3, heads=8, ffn=256, adrop=0.2, fdrop=0.1):
        super().__init__()
        self.w, self.b, self.cls = nn.Parameter(torch.empty(d, dt)), nn.Parameter(torch.empty(d, dt)), nn.Parameter(torch.empty(1, 1, dt))
        for p in (self.w, self.b, self.cls):
            nn.init.uniform_(p, -1 / math.sqrt(dt), 1 / math.sqrt(dt))
        self.att = nn.ModuleList([nn.MultiheadAttention(dt, heads, dropout=adrop, batch_first=True) for _ in range(blocks)])
        self.n1 = nn.ModuleList([nn.LayerNorm(dt) if i else nn.Identity() for i in range(blocks)])
        self.n2 = nn.ModuleList([nn.LayerNorm(dt) for _ in range(blocks)])
        self.f1 = nn.ModuleList([nn.Linear(dt, 2 * ffn) for _ in range(blocks)])
        self.f2 = nn.ModuleList([nn.Linear(ffn, dt) for _ in range(blocks)])
        self.fd, self.hn, self.head = nn.Dropout(fdrop), nn.LayerNorm(dt), nn.Linear(dt, 1)

    def forward(self, x):
        t = x.unsqueeze(-1) * self.w + self.b
        t = torch.cat([self.cls.expand(x.shape[0], -1, -1), t], 1)
        for i in range(len(self.att)):
            h = self.n1[i](t)
            t = t + self.att[i](h, h, h, need_weights=False)[0]
            u, g = self.f1[i](self.n2[i](t)).chunk(2, -1)
            t = t + self.f2[i](self.fd(u * F.relu(g)))
        return self.head(F.relu(self.hn(t[:, 0]))).squeeze(1)


def fit_torch(kind, Xtr, ytr, Xte, seed, quick=False):
    """Deep baselines at their default configuration; early stopping on an internal 15% validation split.
    Returns test probabilities, validation probabilities, validation labels and the fitted network."""
    torch.manual_seed(seed)
    rs = np.random.RandomState(seed)
    itr, iva = _split(ytr, seed)
    Xa, ya, Xv, yv = _T(Xtr[itr]), _T(ytr[itr]), _T(Xtr[iva]), _T(ytr[iva])
    d, n = Xtr.shape[1], len(itr)
    if kind == 'tabnet':
        net = TabNet(d)
        opt, bs, E, pat, lam = torch.optim.Adam(net.parameters(), lr=2e-2), min(256, n), 100, 15, 1e-3
    else:
        net = FTT(d)
        opt, bs, E, pat, lam = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5), (64 if n < 2000 else 256), 200, 16, 0.0
    if quick:
        E = 3

    def fwd(x):
        o = net(x)
        return o if isinstance(o, tuple) else (o, 0.0)

    best, state, bad = float('inf'), None, 0
    for ep in range(E):
        net.train()
        for b in _batches(rs, n, bs):
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
        p = torch.sigmoid(fwd(_T(Xte))[0]).numpy().astype(np.float64)
        pv = torch.sigmoid(fwd(Xv)[0]).numpy().astype(np.float64)
    return p, pv, ytr[iva], net


# ---------------------------------------------------------------- part 3: classifiers, nested tuning, main benchmark
FIX = dict(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
GATE_CFG = {'aagb_ig': dict(gate='input', crossfit=True), 'aagb_rm': dict(gate='rowmean', crossfit=False)}


def fp(base, P, Xa, ya, Xb, seed, spw=1.0, quick=False):
    """Fit one classifier and return (test probabilities, fitted model). Every model runs on a single thread."""
    P = dict(P)
    if quick:
        for k in ('n_estimators', 'iterations'):
            if k in P:
                P[k] = 40
    if base == 'rf':
        m = RandomForestClassifier(n_jobs=1, random_state=seed, **P)
    elif base == 'xgb':
        m = xgb.XGBClassifier(gamma=0.05, tree_method='hist', n_jobs=1, random_state=seed, eval_metric='logloss', scale_pos_weight=spw, **P)
    elif base == 'lgbm':
        m = lgb.LGBMClassifier(subsample_freq=1, n_jobs=1, random_state=seed, verbose=-1, deterministic=True, force_row_wise=True, **P)
    elif base == 'cat':
        m = CatBoostClassifier(bootstrap_type='Bernoulli', thread_count=1, random_seed=seed, verbose=0, allow_writing_files=False, **P)
    else:
        m = MLPClassifier(hidden_layer_sizes=(P['h1'], P['h2'], P['h3']), alpha=P['alpha'], learning_rate_init=P['lr'],
                          early_stopping=True, n_iter_no_change=15, max_iter=30 if quick else 200, random_state=seed)
    m.fit(Xa, ya)
    return m.predict_proba(Xb)[:, 1], m


def space(base, t):
    if base == 'rf':
        return dict(n_estimators=t.suggest_int('n_estimators', 200, 500, step=100), max_depth=t.suggest_int('max_depth', 3, 14),
                    min_samples_leaf=t.suggest_int('min_samples_leaf', 1, 8), max_features=t.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]))
    if base == 'xgb':
        return dict(n_estimators=t.suggest_int('n_estimators', 200, 600, step=50), max_depth=t.suggest_int('max_depth', 3, 8),
                    learning_rate=t.suggest_float('learning_rate', 0.01, 0.2, log=True), subsample=t.suggest_float('subsample', 0.6, 0.95),
                    colsample_bytree=t.suggest_float('colsample_bytree', 0.4, 0.95), min_child_weight=t.suggest_int('min_child_weight', 1, 5),
                    reg_alpha=t.suggest_float('reg_alpha', 1e-3, 2.0, log=True), reg_lambda=t.suggest_float('reg_lambda', 0.3, 5.0, log=True))
    if base == 'lgbm':
        return dict(n_estimators=t.suggest_int('n_estimators', 200, 600, step=50), num_leaves=t.suggest_int('num_leaves', 8, 64),
                    learning_rate=t.suggest_float('learning_rate', 0.01, 0.2, log=True), subsample=t.suggest_float('subsample', 0.6, 0.95),
                    colsample_bytree=t.suggest_float('colsample_bytree', 0.4, 0.95), min_child_samples=t.suggest_int('min_child_samples', 5, 40),
                    reg_alpha=t.suggest_float('reg_alpha', 1e-3, 2.0, log=True), reg_lambda=t.suggest_float('reg_lambda', 0.3, 5.0, log=True))
    if base == 'cat':
        return dict(iterations=t.suggest_int('iterations', 200, 500, step=50), depth=t.suggest_int('depth', 3, 7),
                    learning_rate=t.suggest_float('learning_rate', 0.02, 0.2, log=True), l2_leaf_reg=t.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
                    subsample=t.suggest_float('subsample', 0.6, 0.95))
    return dict(h1=t.suggest_categorical('h1', [64, 128, 256]), h2=t.suggest_categorical('h2', [32, 64, 128]),
                h3=t.suggest_categorical('h3', [16, 32, 64]), alpha=t.suggest_float('alpha', 1e-5, 1e-2, log=True),
                lr=t.suggest_float('lr', 5e-4, 5e-3, log=True))


def tune(base, inner, f, seed, n_trials, n):
    """Seeded TPE search with a fixed number of trials (no wall-clock limit); objective = mean inner-fold PR-AUC.
    Returns the best parameters and the out-of-fold training predictions of the best trial."""
    store = {}

    def obj(t):
        P = space(base, t)
        oof, sc = np.zeros(n), []
        for k, (Xa, ya, Xb, yb, ib) in enumerate(inner):
            pr = f(P, Xa, ya, Xb, seed + k)
            oof[ib] = pr
            sc.append(average_precision_score(yb, pr))
        store[t.number] = oof
        return float(np.mean(sc))

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials)
    return st.best_params, store[st.best_trial.number]


def _setup():
    torch.set_num_threads(1)
    warnings.filterwarnings('ignore')
    optuna.logging.set_verbosity(optuna.logging.ERROR)


def _fold_data(ds, variant, rep, fold, R):
    D = load(ds, variant)
    y = D[1]
    tr, te = outer_splits(y, R)[rep * K_OUT + fold]
    Xtr, Xte = prep(D, tr, te)
    return D, tr, te, Xtr, y[tr], Xte, y[te], 1000 * (rep + 1) + 10 * fold


def aagb_inputs(inner, Xtr, ytr, Xte, gcfg, seed):
    """Augmented matrices for the inner folds and the outer fold; the attention module is refitted in every
    training portion, so validation and test rows never influence it."""
    inn = []
    for k, (Xa, ya, Xb, yb, ib) in enumerate(inner):
        A_a, s_a, A_b, s_b, _ = att_feats(Xa, ya, Xb, gcfg, seed + 7 * (k + 1))
        pairs = select_pairs(A_a)
        inn.append((build_aug(Xa, A_a, s_a, pairs), ya, build_aug(Xb, A_b, s_b, pairs), yb, ib))
    A_tr, s_tr, A_te, s_te, net = att_feats(Xtr, ytr, Xte, gcfg, seed)
    pairs = select_pairs(A_tr)
    return inn, build_aug(Xtr, A_tr, s_tr, pairs), build_aug(Xte, A_te, s_te, pairs), dict(A_te=A_te, pairs=pairs)


def shap_groups(mdl, Xaug, d, pairs, G, has_w=True):
    """TreeSHAP on the augmented columns, aggregated to source variables: the original column and its
    attention-weighted copy are summed, and every product column is split equally between its two parents."""
    C = mdl.get_booster().predict(xgb.DMatrix(Xaug), pred_contribs=True)[:, :-1]
    W = np.zeros((Xaug.shape[1], d))
    W[np.arange(d), np.arange(d)] = 1.0
    off = d
    if has_w:
        W[d + np.arange(d), np.arange(d)] = 1.0
        off = 2 * d
    for i, (p, q) in enumerate(pairs):
        W[off + i, p] += 0.5
        W[off + i, q] += 0.5
    per_grp = (C @ W) @ G.T
    tot = max(float(np.abs(C).sum()), 1e-12)
    share = [float(np.abs(C[:, :d]).sum()) / tot, float(np.abs(C[:, d:off]).sum()) / tot, float(np.abs(C[:, off:]).sum()) / tot]
    return np.abs(per_grp).mean(0), per_grp, share


def job_main(ds, variant, rep, fold, models, C):
    """One outer fold of the nested benchmark for all requested models."""
    _setup()
    q = C.get('quick', False)
    D, tr, te, Xtr, ytr, Xte, yte, s0 = _fold_data(ds, variant, rep, fold, C['R'])
    all_num = len(D[3]) == 0
    G, gn = group_matrix(feat_groups(D)[1])
    d = Xtr.shape[1]
    cnt = np.bincount(ytr, minlength=2)
    inner = []
    for a, b in StratifiedKFold(K_IN, shuffle=True, random_state=s0).split(Xtr, ytr):
        Xa, Xb = prep(D, tr[a], tr[b])
        inner.append((Xa, ytr[a], Xb, ytr[b], b))
    res = dict(rep=rep, fold=fold, te=te, rows=[], probs={}, extra={})

    def add(name, p, thr, secs=0.0):
        r = dict(model=name, rep=rep, fold=fold, thr=thr, secs=secs)
        r.update(tfree(yte, p))
        r.update(tdep(yte, p, 0.5, '5'))
        r.update(tdep(yte, p, thr, 't'))
        res['rows'].append(r)
        res['probs'][name] = p.astype(np.float32)

    for model in models:
        t0 = time.time()
        if model in ('tabnet', 'ftt'):
            p, pv, yv, _ = fit_torch(model, Xtr, ytr, Xte, s0, q)
            add(model, p, best_thr(yv, pv), time.time() - t0)
            continue
        if model in GATE_CFG:
            inn, Xa_o, Xb_o, info = aagb_inputs(inner, Xtr, ytr, Xte, dict(GATE_CFG[model], quick=q), s0)
            base = 'xgb'
        else:
            inn, Xa_o, Xb_o, info, base = inner, Xtr, Xte, None, model
        f = lambda P, Xa, ya, Xb, sd, _b=base: fp(_b, P, Xa, ya, Xb, sd, quick=q)[0]
        bp, oof = tune(base, inn, f, s0, C['trials'], len(ytr))
        p, mdl = fp(base, bp, Xa_o, ytr, Xb_o, s0, quick=q)
        add(model, p, best_thr(ytr, oof), time.time() - t0)
        if C.get('imb', True) and variant == 'main' and model in ('xgb', 'aagb_ig') and cnt.min() / cnt.max() <= 0.8:
            ab = platt_fit(ytr, oof)
            add(model + '+cal', platt_apply(p, ab), best_thr(ytr, platt_apply(oof, ab)))
            for v in ('os', 'spw'):  # same hyperparameters, different imbalance strategy at the classifier
                oofv = np.zeros(len(ytr))
                for k, (Xa, ya, Xb, yb, ib) in enumerate(inn):
                    Xr, yr = oversample(Xa, ya, all_num, s0 + k) if v == 'os' else (Xa, ya)
                    w = float((ya == 0).sum() / max((ya == 1).sum(), 1)) if v == 'spw' else 1.0
                    oofv[ib] = fp('xgb', bp, Xr, yr, Xb, s0 + k, spw=w, quick=q)[0]
                Xr, yr = oversample(Xa_o, ytr, all_num, s0) if v == 'os' else (Xa_o, ytr)
                w = float(cnt[0] / max(cnt[1], 1)) if v == 'spw' else 1.0
                pv = fp('xgb', bp, Xr, yr, Xb_o, s0, spw=w, quick=q)[0]
                add(model + '+' + v, pv, best_thr(ytr, oofv))
                ab = platt_fit(ytr, oofv)
                add(model + '+' + v + '+cal', platt_apply(pv, ab), best_thr(ytr, platt_apply(oofv, ab)))
        if C.get('explain', True) and variant == 'main' and model in ('xgb', 'aagb_ig'):
            imp, _, share = shap_groups(mdl, Xb_o, d, info['pairs'] if info else [], G, has_w=info is not None)
            ex = dict(shap=imp, share=share)
            if info:
                ex['att'] = (info['A_te'].mean(0) @ G.T) / G.sum(1)
            res['extra'][model] = ex
    return res


# ---------------------------------------------------------------- part 4: ablation, gate comparison, factorial, timing
GATES = {'ig': dict(gate='input'), 'rm': dict(gate='rowmean'), 'pj': dict(gate='proj'),
         'ig_ins': dict(gate='input', crossfit=False), 'rm_ins': dict(gate='rowmean', crossfit=False),
         'ig_nosa': dict(gate='input', use_sa=False), 'ig_nofa': dict(gate='input', use_fa=False),
         'ig_untr': dict(gate='input', untrained=True), 'ig_l1a': dict(gate='input', l1=0.01), 'ig_l1b': dict(gate='input', l1=0.05)}


def deletion(net, mdl, pairs, Xte, mu, G, score, ks=(1, 2, 3)):
    """Faithfulness test: mean absolute change of the predicted probability when the k source variables with
    the highest per-row score are replaced by their training means and the whole pipeline is re-evaluated."""
    A, a = infer(net, Xte)
    p0 = mdl.predict_proba(build_aug(Xte, A, a, pairs))[:, 1]
    order = np.argsort(-score, axis=1, kind='stable')
    out = []
    for k in ks:
        M = np.zeros(score.shape)
        np.put_along_axis(M, order[:, :k], 1.0, axis=1)
        Xp = np.where((M @ G) > 0, mu[None, :], Xte)
        A2, a2 = infer(net, Xp)
        out.append(float(np.mean(np.abs(mdl.predict_proba(build_aug(Xp, A2, a2, pairs))[:, 1] - p0))))
    return out


def job_abl(ds, rep, fold, C):
    """Component ablation and gate comparison with fixed XGBoost hyperparameters on one outer fold."""
    _setup()
    q = C.get('quick', False)
    D, tr, te, Xtr, ytr, Xte, yte, s0 = _fold_data(ds, 'main', rep, fold, C['R'])
    G, gn = group_matrix(feat_groups(D)[1])
    d, mu, rs = Xtr.shape[1], Xtr.mean(0), np.random.RandomState(s0)
    rows, diag = [], {}

    def run(name, Xa, Xb):
        p, m = fp('xgb', FIX, Xa, ytr, Xb, s0, quick=q)
        r = dict(variant=name, rep=rep, fold=fold)
        r.update(tfree(yte, p))
        r.update(tdep(yte, p, 0.5, '5'))
        rows.append(r)
        return m

    m0 = run('xgb', Xtr, Xte)
    for key, g in GATES.items():
        A_tr, s_tr, A_te, s_te, net = att_feats(Xtr, ytr, Xte, dict(g, quick=q), s0)
        if key == 'ig_nofa':  # sample attention only
            run('nofa', np.hstack([Xtr, Xtr * s_tr[:, None]]), np.hstack([Xte, Xte * s_te[:, None]]))
            continue
        pairs = select_pairs(A_tr)
        Xb_aug = build_aug(Xte, A_te, s_te, pairs)
        mdl = run('full_' + key, build_aug(Xtr, A_tr, s_tr, pairs), Xb_aug)
        if key == 'ig':
            run('w_only', build_aug(Xtr, A_tr, s_tr, pairs, True, False), build_aug(Xte, A_te, s_te, pairs, True, False))
            run('int_only', build_aug(Xtr, A_tr, s_tr, pairs, False, True), build_aug(Xte, A_te, s_te, pairs, False, True))
        if key in ('ig', 'rm', 'pj'):
            cm = A_te.mean(0)
            Ag = (A_te @ G.T) / G.sum(1)[None, :]
            pg = shap_groups(mdl, Xb_aug, d, pairs, G)[1]
            dl = {k: deletion(net, mdl, pairs, Xte, mu, G, sc)
                  for k, sc in (('top', Ag), ('bot', -Ag), ('rnd', rs.rand(*Ag.shape)), ('shp', np.abs(pg)))}
            diag[key] = dict(att=(cm @ G.T) / G.sum(1), rng=float(cm.max() - cm.min()), mean=float(A_te.mean()),
                             sat=float((A_te > 0.95).mean()), wsd=float(A_te.std(1).mean()), asd=float(A_te.std(0).mean()), dl=dl)
            if key == 'rm':  # does w_a^T h agree in sign with the label (drives the gate towards 1)?
                with torch.no_grad():
                    h, A, a = net.attention(_T(Xtr))
                    s = (h @ net.head.weight.T).squeeze(1).numpy()
                    diag[key]['agree'] = float(np.mean((s > 0) == (ytr == 1)))
                    diag[key]['gprod'] = float((A.mean(1) * a).mean())
    for name, top in (('int_rand', rs.choice(d, min(5, d), replace=False)), ('int_gain', np.argsort(-m0.feature_importances_, kind='stable')[:5])):
        pr_ = sorted((int(min(p, q)), int(max(p, q))) for p, q in itertools.combinations(top, 2))
        run(name, build_aug(Xtr, None, None, pr_, False, True), build_aug(Xte, None, None, pr_, False, True))
    return dict(rows=rows, diag=diag)


def job_fact(ds, rep, fold, C):
    """2^4 factorial of imbalance handling for the input-gated, cross-fitted pipeline (fixed hyperparameters)."""
    _setup()
    q = C.get('quick', False)
    D, tr, te, Xtr, ytr, Xte, yte, s0 = _fold_data(ds, 'main', rep, fold, C['R'])
    all_num = len(D[3]) == 0
    cnt = np.bincount(ytr, minlength=2)
    feats, rows = {}, []
    for ao in (0, 1):
        for wb in (0, 1):
            feats[(ao, wb)] = att_feats(Xtr, ytr, Xte, dict(gate='input', att_os=bool(ao), wbce=bool(wb), all_num=all_num, quick=q), s0)
    for ao, co, wb, sp in itertools.product((0, 1), repeat=4):
        A_tr, s_tr, A_te, s_te, _ = feats[(ao, wb)]
        pairs = select_pairs(A_tr)
        Xa, ya, Xb = build_aug(Xtr, A_tr, s_tr, pairs), ytr, build_aug(Xte, A_te, s_te, pairs)
        if co:
            Xa, ya = oversample(Xa, ytr, all_num, s0)
        p = fp('xgb', FIX, Xa, ya, Xb, s0, spw=float(cnt[0] / max(cnt[1], 1)) if sp else 1.0, quick=q)[0]
        r = dict(rep=rep, fold=fold, A=ao, C=co, W=wb, S=sp)
        r.update(tfree(yte, p))
        r.update(tdep(yte, p, 0.5, '5'))
        rows.append(r)
    return rows


def timing(ds, C):
    """Single-thread wall-clock time of one fit and of inference (per 1000 rows) on the first outer fold."""
    _setup()
    q = C.get('quick', False)
    D, tr, te, Xtr, ytr, Xte, yte, s0 = _fold_data(ds, 'main', 0, 0, C['R'])
    DEF = dict(rf=dict(n_estimators=300, max_depth=8, min_samples_leaf=2, max_features='sqrt'), xgb=FIX,
               lgbm=dict(n_estimators=200, num_leaves=31, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0),
               cat=dict(iterations=300, depth=6, learning_rate=0.1, l2_leaf_reg=3.0, subsample=0.8),
               mlp=dict(h1=128, h2=64, h3=32, alpha=1e-4, lr=1e-3))
    out, k = {}, 1000.0 / len(yte)
    for m, P in DEF.items():
        t0 = time.perf_counter()
        mdl = fp(m, P, Xtr, ytr, Xte[:2], s0, quick=q)[1]
        t1 = time.perf_counter()
        mdl.predict_proba(Xte)
        out[m] = (t1 - t0, (time.perf_counter() - t1) * k)
    for m in ('tabnet', 'ftt'):
        t0 = time.perf_counter()
        net = fit_torch(m, Xtr, ytr, Xte[:2], s0, q)[3]
        t1 = time.perf_counter()
        with torch.no_grad():
            net(_T(Xte))
        out[m] = (t1 - t0, (time.perf_counter() - t1) * k)
    for m, g in GATE_CFG.items():
        t0 = time.perf_counter()
        A_tr, s_tr, A_te, s_te, net = att_feats(Xtr, ytr, Xte[:2], dict(g, quick=q), s0)
        pairs = select_pairs(A_tr)
        mdl = fp('xgb', FIX, build_aug(Xtr, A_tr, s_tr, pairs), ytr, build_aug(Xte[:2], A_te, s_te, pairs), s0, quick=q)[1]
        t1 = time.perf_counter()
        A, a = infer(net, Xte)
        mdl.predict_proba(build_aug(Xte, A, a, pairs))
        out[m] = (t1 - t0, (time.perf_counter() - t1) * k)
    return out


def determinism(ds, C):
    """Largest absolute difference between the test probabilities of repeated executions of the same fold
    (twice in the parent process, twice in separate worker processes). Zero means bitwise-reproducible."""
    C2 = dict(C, trials=min(C['trials'], 5), explain=False, imb=False)
    runs = [job_main(ds, 'main', 0, 0, ['aagb_ig'], C2) for _ in range(2)]
    runs += Parallel(n_jobs=2)(delayed(job_main)(ds, 'main', 0, 0, ['aagb_ig'], C2) for _ in range(2))
    P = [r['probs']['aagb_ig'].astype(np.float64) for r in runs]
    return max(float(np.max(np.abs(P[0] - x))) for x in P[1:])


# ---------------------------------------------------------------- part 5: orchestration and reporting
MODELS = ['rf', 'xgb', 'lgbm', 'cat', 'mlp', 'tabnet', 'ftt', 'aagb_rm', 'aagb_ig']
SENS = {'heart': ['redist', 'redist_fix'], 'pima': ['zeros'], 'ckd': ['no_sc', 'no_hemo', 'no_pcv', 'no_all3'], 'stroke': []}
MET = ['roc', 'pr', 'brier', 'ece', 'f15', 'f1t', 'mcct', 'senst', 'spect', 'prect', 'bacct', 'thr']
PT = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
ABL = ['xgb', 'full_ig', 'full_rm', 'full_pj', 'full_ig_ins', 'full_rm_ins', 'w_only', 'int_only', 'int_rand', 'int_gain',
       'full_ig_nosa', 'nofa', 'full_ig_untr', 'full_ig_l1a', 'full_ig_l1b']


def _par(fn, args, C, label):
    t, out = time.time(), []
    try:
        it = Parallel(n_jobs=C['n_jobs'], return_as='generator_unordered')(delayed(fn)(*a) for a in args)
    except Exception:
        it = Parallel(n_jobs=C['n_jobs'])(delayed(fn)(*a) for a in args)
    for i, r in enumerate(it):
        out.append(r)
        if (i + 1) % max(1, len(args) // 5) == 0:
            print('[%s] %s %d/%d %.0fs' % (label, time.strftime('%H:%M:%S'), i + 1, len(args), time.time() - t), flush=True)
    return out


def _pivot(df, col, val):
    return df.pivot_table(index=['rep', 'fold'], columns=col, values=val, aggfunc='first').sort_index()


def run_dataset(ds, C):
    T0 = time.time()
    od = os.path.join(OUT, ds)
    os.makedirs(od, exist_ok=True)
    R, stages = C['R'], C.get('stages', ['main', 'sens', 'abl', 'fact', 'time', 'det'])
    D = load(ds)
    y = D[1]
    names, groups = feat_groups(D)
    G, gn = group_matrix(groups)
    folds = [(r, f) for r in range(R) for f in range(K_OUT)]
    emit(ds, 'INFO', 'cfg', [R, C['trials'], len(y), int(y.sum()), len(names), len(gn), len(folds)], 1)
    emit(ds, 'VARS', 'names', ';'.join(gn))
    for k, v in audit(ds).items():
        emit(ds, 'AUD', k, v, 1)
    gc = grad_check()
    emit(ds, 'GC', 'spread', ';'.join('%s=%.3e' % (k, v) for k, v in gc.items()))

    if 'main' in stages:
        models = C.get('models', MODELS)
        res = sorted(_par(job_main, [(ds, 'main', r, f, models, C) for r, f in folds], C, ds + ' main'), key=lambda z: (z['rep'], z['fold']))
        df = pd.DataFrame([row for z in res for row in z['rows']])
        df.to_csv(os.path.join(od, 'fold_metrics_main.csv'), index=False)
        allm = list(dict.fromkeys(df.model))
        P = {m: np.full((R, len(y)), np.nan, dtype=np.float32) for m in allm}
        for z in res:
            for m, p in z['probs'].items():
                P[m][z['rep'], z['te']] = p
        np.savez_compressed(os.path.join(od, 'test_predictions_main.npz'), y=y, **{m.replace('+', '_'): v for m, v in P.items()})
        for m in allm:
            sub = df[df.model == m]
            vals = []
            for k in MET:
                mu, sd, ci = csum(sub[k].values)
                vals += [mu, ci]
            emit(ds, 'M', m, vals)
            emit(ds, 'SEC', m, [sub.secs.sum()], 1)
            cal = np.array([platt_fit(y, P[m][r].astype(np.float64)) for r in range(R)])
            citl = np.mean([P[m][r].mean() - y.mean() for r in range(R)])
            emit(ds, 'CAL', m, [cal[:, 0].mean(), cal[:, 1].mean(), citl])
            cc = np.zeros((R, 10, 2))
            for r in range(R):
                pr_ = P[m][r].astype(np.float64)
                for b, ch in enumerate(np.array_split(np.argsort(pr_, kind='stable'), 10)):
                    cc[r, b] = (pr_[ch].mean(), y[ch].mean())
            emit(ds, 'CC', m, cc.mean(0).ravel())
            nb = np.zeros((R, len(PT)))
            for r in range(R):
                for i, pt in enumerate(PT):
                    pred = P[m][r] >= pt
                    nb[r, i] = ((pred & (y == 1)).sum() - (pred & (y == 0)).sum() * pt / (1 - pt)) / len(y)
            emit(ds, 'DC', m, nb.mean(0))
        emit(ds, 'DC', 'treat_all', [y.mean() - (1 - y.mean()) * pt / (1 - pt) for pt in PT])
        for ref, others in (('aagb_ig', [m for m in allm if m != 'aagb_ig' and '+' not in m]), ('aagb_rm', ['xgb'])):
            if ref not in allm:
                continue
            for m in others:
                vals = []
                for k in ('roc', 'pr', 'brier', 'f1t', 'f15'):
                    pv = _pivot(df[df.model.isin([ref, m])], 'model', k)
                    dlt = (pv[ref] - pv[m]).values
                    vals += [np.nanmean(dlt), np.nanstd(dlt, ddof=1) if len(dlt) > 1 else 0.0]
                emit(ds, 'T', ref + '>' + m, vals, 1e6)
        ex = {m: [z['extra'][m] for z in res if m in z['extra']] for m in ('xgb', 'aagb_ig')}
        mean_vec = {}
        for m, lst in ex.items():
            if not lst:
                continue
            S = np.vstack([e['shap'] for e in lst])
            sp, jc, rk = stability(S)
            mean_vec['shap_' + m] = S.mean(0)
            emit(ds, 'SH', m, S.mean(0))
            emit(ds, 'SHK', m, rk, 100)
            emit(ds, 'SHS', m, [sp, jc] + np.mean([e['share'] for e in lst], 0).tolist())
            if 'att' in lst[0]:
                A = np.vstack([e['att'] for e in lst])
                sp, jc, rk = stability(A)
                mean_vec['att'] = A.mean(0)
                emit(ds, 'SA', m, A.mean(0))
                emit(ds, 'SAK', m, rk, 100)
                emit(ds, 'SAS', m, [sp, jc, A.std(0, ddof=1).mean() if len(A) > 1 else 0.0])
            np.save(os.path.join(od, 'shap_by_variable_%s.npy' % m), S)
        ks = sorted(mean_vec)
        emit(ds, 'AGR', ','.join(ks), [stats.spearmanr(mean_vec[a], mean_vec[b])[0] for a, b in itertools.combinations(ks, 2)])
        print('[%s] main done %.0fs' % (ds, time.time() - T0), flush=True)

    if 'sens' in stages:
        Cs = dict(C, explain=False, imb=False)
        for v in SENS[ds]:
            res = _par(job_main, [(ds, v, r, f, ['xgb', 'aagb_ig'], Cs) for r, f in folds], C, ds + ' sens ' + v)
            dv = pd.DataFrame([row for z in res for row in z['rows']])
            dv.to_csv(os.path.join(od, 'fold_metrics_%s.csv' % v), index=False)
            for m in ('xgb', 'aagb_ig'):
                vals = []
                for k in ('roc', 'pr', 'brier', 'f1t', 'f15'):
                    mu, sd, ci = csum(dv[dv.model == m][k].values)
                    vals += [mu, ci]
                emit(ds, 'S', v + ':' + m, vals)

    if 'abl' in stages:
        res = _par(job_abl, [(ds, r, f, C) for r, f in folds], C, ds + ' abl')
        da = pd.DataFrame([row for z in res for row in z['rows']])
        da.to_csv(os.path.join(od, 'fold_metrics_ablation.csv'), index=False)
        piv = {k: _pivot(da, 'variant', k) for k in ('roc', 'pr', 'brier')}
        for v in ABL:
            vals = [np.nanmean(piv[k][v].values) for k in ('roc', 'pr', 'brier')]
            emit(ds, 'AB', v, vals)
            vals = []
            for ref in ('xgb', 'full_ig'):
                for k in ('roc', 'pr', 'brier'):
                    dlt = (piv[k][v] - piv[k][ref]).values
                    vals += [np.nanmean(dlt), np.nanstd(dlt, ddof=1) if len(dlt) > 1 else 0.0]
            emit(ds, 'ABD', v, vals, 1e6)
        for g in ('ig', 'rm', 'pj'):
            dg = [z['diag'][g] for z in res]
            A = np.vstack([e['att'] for e in dg])
            sp, jc, rk = stability(A)
            emit(ds, 'GA', g, A.mean(0))
            emit(ds, 'GAK', g, rk, 100)
            emit(ds, 'GD', g, [np.mean([e[k] for e in dg]) for k in ('rng', 'mean', 'sat', 'wsd', 'asd')] + [sp, jc, A.max(1).mean() - A.min(1).mean()])
            dl = {k: np.array([e['dl'][k] for e in dg]) for k in ('top', 'bot', 'rnd', 'shp')}
            emit(ds, 'GDL', g, np.concatenate([dl[k].mean(0) for k in ('top', 'bot', 'rnd', 'shp')]))
            dd = dl['top'] - dl['rnd']
            emit(ds, 'GDT', g, np.column_stack([dd.mean(0), dd.std(0, ddof=1) if len(dd) > 1 else np.zeros(3)]).ravel(), 1e6)
            if g == 'rm':
                emit(ds, 'GRM', g, [np.mean([e['agree'] for e in dg]), np.mean([e['gprod'] for e in dg])])
            np.save(os.path.join(od, 'attention_by_variable_%s.npy' % g), A)
        print('[%s] abl done %.0fs' % (ds, time.time() - T0), flush=True)

    if 'fact' in stages:
        res = _par(job_fact, [(ds, r, f, C) for r, f in folds], C, ds + ' fact')
        dfc = pd.DataFrame([row for rows in res for row in rows])
        dfc.to_csv(os.path.join(od, 'fold_metrics_factorial.csv'), index=False)
        fs = ['A', 'C', 'W', 'S']
        for k in ('f15', 'pr', 'brier'):
            cell = dfc.pivot_table(index=['rep', 'fold'], columns=fs, values=k, aggfunc='first').sort_index()
            lv = np.array([list(c) for c in cell.columns])
            V = cell.values
            emit(ds, 'FC', k, np.nanmean(V, 0))
            vals = []
            for i in range(4):
                e = V[:, lv[:, i] == 1].mean(1) - V[:, lv[:, i] == 0].mean(1)
                vals += [np.nanmean(e), np.nanstd(e, ddof=1) if len(e) > 1 else 0.0]
            for i, j in itertools.combinations(range(4), 2):
                hi = V[:, (lv[:, i] == 1) & (lv[:, j] == 1)].mean(1) - V[:, (lv[:, i] == 0) & (lv[:, j] == 1)].mean(1)
                lo = V[:, (lv[:, i] == 1) & (lv[:, j] == 0)].mean(1) - V[:, (lv[:, i] == 0) & (lv[:, j] == 0)].mean(1)
                e = 0.5 * (hi - lo)
                vals += [np.nanmean(e), np.nanstd(e, ddof=1) if len(e) > 1 else 0.0]
            emit(ds, 'F', k, vals, 1e6)
        emit(ds, 'FCOL', 'order', ';'.join(''.join(str(int(x)) for x in c) for c in lv))
        print('[%s] fact done %.0fs' % (ds, time.time() - T0), flush=True)

    if 'time' in stages:
        for m, (a, b) in timing(ds, C).items():
            emit(ds, 'TM', m, [a * 1000, b * 1e6], 1)
    if 'det' in stages:
        emit(ds, 'DET', 'maxdiff', '%.3e' % determinism(ds, C))
    emit(ds, 'DONE', 'secs', [time.time() - T0], 1)
    with open(os.path.join(od, 'report_lines.txt'), 'w') as fh:
        fh.write(chr(10).join(l for l in _LINES if l.startswith('@@' + ds + '|')) + chr(10))


def module_hash(path):
    """Whitespace-insensitive SHA-256 of the pipeline source (printed in the log for provenance)."""
    with open(path) as fh:
        txt = chr(10).join(l.rstrip() for l in fh.read().splitlines() if l.strip())
    return hashlib.sha256(txt.encode()).hexdigest()


def run(datasets, C):
    import sklearn, catboost, imblearn, platform
    print('@@VER|py=%s;np=%s;pd=%s;sk=%s;xgb=%s;lgb=%s;cat=%s;torch=%s;optuna=%s;imblearn=%s' % (
        platform.python_version(), np.__version__, pd.__version__, sklearn.__version__, xgb.__version__, lgb.__version__,
        catboost.__version__, torch.__version__, optuna.__version__, imblearn.__version__), flush=True)
    print('@@HASH|' + module_hash(os.path.join(OUT, 'aagb2.py')), flush=True)
    for ds in datasets:
        run_dataset(ds, C)
    print('@@ALLDONE', flush=True)
