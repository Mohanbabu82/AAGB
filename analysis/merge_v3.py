# Merge numbers_v3.json (prefix "v3.") and the v3 tables into the build folder used by ms/build.py.
import json, sys, shutil, os, glob
out, v3 = sys.argv[1], sys.argv[2]
N = json.load(open(os.path.join(out, 'numbers.json')))
N = {k: v for k, v in N.items() if not k.startswith('v3.')}
M = json.load(open(os.path.join(v3, 'numbers_v3.json')))
def flat(prefix, v):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from flat(prefix + '.' + k, x)
    else:
        yield prefix, v
for k, v in M.items():
    if k in ('records', 'bad', 'meta'):
        continue
    for kk, vv in flat('v3.' + k, v):
        N[kk] = vv
N['v3.records'] = M['records']
json.dump(N, open(os.path.join(out, 'numbers.json'), 'w'), indent=1)
for f in glob.glob(os.path.join(v3, 'tables', 'v3_*.md')):
    shutil.copy(f, os.path.join(out, 'tables'))
print('merged', sum(1 for k in N if k.startswith('v3.')))
