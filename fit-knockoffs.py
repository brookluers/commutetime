import numpy as np
from dask import array as da
import tables
import json


xinfo = {}
with open('xcolnames.json') as jf:
    xinfo['xcolnames'] = json.load(jf)

with open('xtermslices.json') as jf:
    xinfo['xtermcols'] = json.load(jf)

h5read = tables.open_file('knockoff-data.h5', mode='r')
h5regression = tables.open_file('regression-data.h5', mode='r')
X = da.from_array(h5read.root.X)
Xtilde = da.from_array(h5read.root.Xtilde)
Y = da.from_array(h5regression.root.Y)
keepcols_svd = list(h5read.root.keepcols)
xcolnames_pdim = []
for k in xinfo['xcolnames']:
    if xinfo['xcolnames'][k] in keepcols_svd:
        xcolnames_pdim.append(k)

Xaug = da.hstack([X, Xtilde])
betahat_aug = da.linalg.solve(da.matmul(Xaug.T, Xaug), da.matmul(Xaug.T, Y))
