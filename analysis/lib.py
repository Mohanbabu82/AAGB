"""Local analysis helpers: load transferred result files, verify checksums, compute corrected statistics."""
import os, re, zlib, glob, math
import numpy as np, pandas as pd
from scipy import stats

K = 5                       # outer folds per repetition
RHO = 1.0 / (K - 1)         # n_test / n_train for k-fold CV
SCALE = {'T': 1e6, 'ABD': 1e6, 'F': 1e6, 'GDT': 1e6, 'XBD': 1e6, 'XGDT': 1e6, 'SHK': 100, 'SAK': 100, 'GAK': 100,
         'AUD': 1, 'INFO': 1, 'SEC': 1, 'TM': 1, 'DONE': 1}
TEXT = {'VARS', 'GC', 'DET', 'FCOL'}
DS = ['heart', 'pima', 'ckd', 'stroke']
DSN = {'heart': 'Heart Disease', 'pima': 'Pima Diabetes', 'ckd': 'CKD', 'stroke': 'Stroke'}
MODELS = ['rf', 'xgb', 'lgbm', 'cat', 'mlp', 'tabnet', 'ftt', 'aagb_rm', 'aagb_ig']
MN = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'cat': 'CatBoost', 'mlp': 'MLP', 'tabnet': 'TabNet',
      'ftt': 'FT-Transformer', 'aagb_rm': 'AAGB-RM (original)', 'aagb_ig': 'AAGB-IG (revised)'}


def parse_report(paths):
    """Parse checksummed '@@ds|block|key|body|crc' lines; return {ds: {block: {key: values}}} and the number of bad lines."""
    out, bad, seen = {}, 0, 0
    for path in paths:
        for line in open(path, encoding='utf8'):
            m = re.search(r'@@(.*)\|([0-9a-f]{8})\s*$', line)
            if not m:
                continue
            s, crc = m.group(1), m.group(2)
            seen += 1
            if '%08x' % (zlib.crc32(s.encode()) & 0xffffffff) != crc:
                bad += 1
                continue
            ds, block, key, body = s.split('|', 3)
            if block in TEXT:
                val = body
            else:
                sc = SCALE.get(block, 1e4)
                val = [np.nan if t == 'x' else int(t) / sc for t in body.split(',')] if body else []
            out.setdefault(ds, {}).setdefault(block, {})[key] = val
    return out, bad, seen


def load_npy_txt(path):
    with open(path) as fh:
        r, c = map(int, fh.readline().split())
        a = np.loadtxt(fh, ndmin=2)
    assert a.shape == (r, c), (path, a.shape, r, c)
    return a


class Results:
    """All result files of one dataset (possibly from several runs, which are concatenated)."""

    def __init__(self, root, ds):
        self.ds = ds
        dirs = sorted(d for d in glob.glob(os.path.join(root, '*', ds)) + glob.glob(os.path.join(root, ds)) if os.path.isdir(d))
        assert dirs, 'no results for ' + ds
        self.dirs = dirs

        def cat(name):
            fs = [os.path.join(d, name) for d in dirs if os.path.exists(os.path.join(d, name))]
            return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None

        self.main, self.abl, self.fact = cat('fold_metrics_main.csv'), cat('fold_metrics_ablation.csv'), cat('fold_metrics_factorial.csv')
        self.sens = {}
        for d in dirs:
            for f in glob.glob(os.path.join(d, 'fold_metrics_*.csv')):
                v = os.path.basename(f)[len('fold_metrics_'):-4]
                if v not in ('main', 'ablation', 'factorial'):
                    self.sens[v] = pd.read_csv(f)
        self.npy = {}
        for d in dirs:
            for f in glob.glob(os.path.join(d, '*.npy.txt')):
                self.npy[os.path.basename(f)[:-8]] = load_npy_txt(f)
        rep, self.bad, self.seen = parse_report([os.path.join(d, 'report_lines.txt') for d in dirs if os.path.exists(os.path.join(d, 'report_lines.txt'))])
        self.rep = rep.get(ds, {})
        self.vars = self.rep['VARS']['names'].split(';')
        self.nfold = int(self.main.groupby('model').size().max())

    def fold(self, model, metric, df=None):
        """Per-fold values of one metric for one model, ordered by (rep, fold)."""
        df = self.main if df is None else df
        col = 'model' if 'model' in df.columns else 'variant'
        return df[df[col] == model].sort_values(['rep', 'fold'])[metric].values.astype(float)

    def r(self, block, key):
        return self.rep[block][key]


def csum(v):
    """Mean, SD and corrected-resampled 95% CI half-width (Nadeau-Bengio variance for repeated k-fold CV)."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return (float(v.mean()) if n else np.nan), 0.0, 0.0
    sd = float(v.std(ddof=1))
    return float(v.mean()), sd, float(stats.t.ppf(0.975, n - 1) * math.sqrt((1.0 / n + RHO) * sd ** 2))


def paired(d, rope=0.01):
    """Corrected repeated k-fold t-test (Bouckaert-Frank) and Bayesian correlated t-test (Benavoli et al.) on fold differences."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = math.sqrt((1.0 / n + RHO) * sd ** 2)
    if se == 0:
        return dict(n=n, mean=m, sd=sd, lo=m, hi=m, t=0.0 if m == 0 else np.inf * np.sign(m), p=1.0 if m == 0 else 0.0,
                    p_left=float(m < -rope), p_rope=float(abs(m) <= rope), p_right=float(m > rope), dz=np.nan)
    t = m / se
    h = stats.t.ppf(0.975, n - 1) * se
    post = stats.t(n - 1, loc=m, scale=se)
    return dict(n=n, mean=m, sd=sd, lo=m - h, hi=m + h, t=t, p=float(2 * stats.t.sf(abs(t), n - 1)),
                p_left=float(post.cdf(-rope)), p_rope=float(post.cdf(rope) - post.cdf(-rope)), p_right=float(post.sf(rope)), dz=m / sd)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def min_detectable_dz(n, alpha=0.05):
    """Smallest |d_z| (mean / SD of fold differences) that reaches two-sided significance under the corrected test."""
    return float(stats.t.ppf(1 - alpha / 2, n - 1) * math.sqrt(1.0 / n + RHO))


def stability(M, k=3):
    M = np.asarray(M, float)
    R = np.vstack([stats.rankdata(-r) for r in M])
    tops = [set(np.argsort(-r, kind='stable')[:k].tolist()) for r in M]
    sp, jc = [], []
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            if R[i].std() > 0 and R[j].std() > 0:
                sp.append(np.corrcoef(R[i], R[j])[0, 1])
            jc.append(len(tops[i] & tops[j]) / len(tops[i] | tops[j]))
    return (float(np.mean(sp)) if sp else np.nan), (float(np.mean(jc)) if jc else np.nan), R.mean(0)


def f3(x, plus=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    if abs(x) < 0.0005:
        return '0.000'
    s = ('%+.3f' if plus else '%.3f') % x
    return s.replace('-', '−')


def fp(p):
    if not np.isfinite(p):
        return 'n/a'
    return '<0.001' if p < 0.001 else '%.3f' % p
