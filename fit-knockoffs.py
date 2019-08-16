import numpy as np
from dask import array as da
import tables
import json

def knockoff_threshold(Wstat, q, offset):
    p = len(Wstat)
    Wabs = np.sort([a for a in map(abs, Wstat)])
    ok_ix = []
    for j in range(p):
        thresh = Wabs[j]
        numer = offset + np.sum([Wstat[i] <= -thresh for i in range(p)])
        denom = max(1.0, np.sum([Wstat[i] >= thresh for i in range(p)]))
        if numer / denom <= q:
            ok_ix.append(j)
    if len(ok_ix) > 0:
        return Wabs[ok_ix[0]]
    else:
        return float('Inf')

FDR = 0.1
xinfo = {}
with open('xcolnames.json') as jf:
    xinfo['xcolnames'] = json.load(jf)

with open('xtermslices.json') as jf:
    xinfo['xtermcols'] = json.load(jf)

h5read = tables.open_file('knockoff-data.h5', mode='r')
h5regression = tables.open_file('regression-data.h5', mode='r')
X = da.from_array(h5read.root.X)
pdim = X.shape[1]
Xtilde = da.from_array(h5read.root.Xtilde)
Y = da.from_array(h5regression.root.Y)
keepcols_svd = list(h5read.root.keepcols)
xcolnames_pdim = []
for k in xinfo['xcolnames']:
    if xinfo['xcolnames'][k] in keepcols_svd:
        xcolnames_pdim.append(k)

Xaug = da.hstack([X, Xtilde])
betahat_aug = da.linalg.solve(da.matmul(Xaug.T, Xaug), da.matmul(Xaug.T, Y)).compute()
Wstat = [abs(betahat_aug[i]) - abs(betahat_aug[i +pdim]) for i in range(pdim)]
threshold = knockoff_threshold(Wstat, FDR, offset=1)
sel = [Wstat[j] >= threshold for j in range(pdim)]

Xdrop = X[:, sel]
da.matmul(Xdrop.T, Xdrop)
betahat_final = da.linalg.solve(da.matmul(Xdrop.T, Xdrop), da.matmul(Xdrop.T, Y)).compute()
colnames_final = [i for i, j in zip(xcolnames_pdim, sel) if j]
colnames_dropped = [i for i, j in zip(xcolnames_pdim, sel) if not j]
print("desired FDR: ")
print(FDR)
print("\nKnockoff drops these columns:\n")
print(colnames_dropped)
h5read.close()
h5regression.close()
