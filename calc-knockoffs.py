import numpy as np
from scipy.optimize import minimize
import scipy.linalg
from dask import array as da
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
from dask.diagnostics import visualize
import dask.config
import tables
import json

def get_ldetfun(Sigma, tol=1e-16):
    def f(svec):
        W = 2 * Sigma - np.diag(svec)
        Wev = np.linalg.eigvalsh(W)
        if any(Wev < tol):
            return -np.Inf
        else:
            return np.sum(np.log(svec)) + np.sum(np.log(Wev))
    return f

def get_ldetgrad(Sigma):
    pdim = Sigma.shape[1]
    def f(svec):
        W = 2 * Sigma - np.diag(svec)
        Winv = np.linalg.inv(W)
        return 1.0 / svec - np.diag(Winv)
    return f

def getknockoffs_qr(Xmat, Qx, Rx, G, svec):
    Utilde_raw = np.random.normal(size=Xmat.shape[0] * Xmat.shape[1]).reshape(Xmat.shape)
    Utilde_raw = Utilde_raw - np.matmul(Qx, np.matmul(Qx.T, Utilde_raw))
    Utilde, Ru = scipy.linalg.qr(Utilde_raw, mode='economic')
    Smat = np.diag(svec)
    Ginv_S = scipy.linalg.solve(G, Smat)
    CtC = 2 * Smat - np.matmul(Smat, Ginv_S)
    Cmat = scipy.linalg.cholesky(CtC)
    return Xmat - np.matmul(Xmat, Ginv_S) + np.matmul(Utilde, Cmat)


xinfo = {}
with open('xcolnames.json') as jf:
    xinfo['xcolnames'] = json.load(jf)

with open('xtermslices.json') as jf:
    xinfo['xtermcols'] = json.load(jf)

h5read = tables.open_file('regression-data.h5', mode='r')
h5write = tables.open_file('knockoff-data.h5', mode='w')
fatom = tables.Float64Atom()
filters = tables.Filters(complevel=1, complib='zlib')
Xmat = da.from_array(h5read.root.X, chunks = 200000)
Y = da.from_array(h5read.root.Y)
Wgt = da.from_array(h5read.root.wgt)
xmeans = da.mean(Xmat, axis=0)
print("Centering X columns")
Xmat = Xmat - xmeans
xnorms = da.linalg.norm(Xmat, axis=0)
xnorms, xmeans = da.compute(xnorms, xmeans)
keepcols = np.arange(Xmat.shape[1])[np.nonzero(xnorms)]
dropcols = np.arange(Xmat.shape[1])[xnorms==0]
print("Dropping column with norm zero:")
xcolnames = []
for colname in xinfo['xcolnames']:
    xcolnames.append(colname)
    for dropix in dropcols:
        if xinfo['xcolnames'][colname] == dropix:
            print(colname)

Xmat = Xmat[:, keepcols]
xnorms = xnorms[keepcols]
xmeans = xmeans[keepcols]
xcolnames_keep = np.array(xcolnames)[keepcols]
print("Standardizing X columns")
Xmat = Xmat / xnorms
tol = 1e-10
Qx, Rx, Px = scipy.linalg.qr(Xmat, mode='economic', pivoting=True)
dropcols_qr = Px[np.nonzero(abs(np.diag(Rx))<tol)]
keepcols_qr = Px[np.nonzero(abs(np.diag(Rx))>=tol)]
rank = np.sum(abs(np.diag(Rx)) >= tol)
Rx = Rx[0:rank, 0:rank]
Qx = Qx[:, 0:rank]
print("Dropping columns based on pivoted QR:")
print("\t" + "\n\t".join(xcolnames_keep[dropcols_qr]))
xnorms = xnorms[keepcols_qr]
xmeans = xmeans[keepcols_qr]
Xmat = Xmat[:, keepcols_qr]
xcolnames_keep = xcolnames_keep[keepcols_qr]
pdim = Xmat.shape[1]
print("design matrix has dimensions {:d} by {:d}".format(Xmat.shape[0], pdim))
# pseudo-inverse:  (X^t X)^(-1) X^t = R^(-1) Q^T
X_pseudo_inv = scipy.linalg.solve_triangular(Rx, Qx.T)
# condition number:  || X|| * ||X_pseudo_inv||
X_cnum = scipy.linalg.norm(Rx, ord=2) * scipy.linalg.norm(X_pseudo_inv, ord=2)

G = np.matmul(Rx.T, Rx)
ldetf = get_ldetfun(G)
ldetgrad = get_ldetgrad(G)
ldopt = minimize(lambda x: -ldetf(x),
        x0 = np.repeat(0.005, pdim),
        jac = lambda x: -ldetgrad(x),
        options={"maxiter": 25000},
        tol=1e-10,
        constraints = scipy.optimize.LinearConstraint(np.identity(pdim),lb=0,ub=1.0))
svec = ldopt.x

Xtilde = getknockoffs_qr(Xmat,Qx, Rx, G, svec)
np.matmul(Xtilde.T, Xmat)
Xtilde_store = h5write.create_carray(h5write.root, 'Xtilde', fatom,
                                     shape = Xtilde.shape,
                                     filters = filters)
Xmat_store = h5write.create_carray(h5write.root, 'X', fatom,
                                    shape = Xmat.shape,
                                    filters=filters)
keepcols_store = h5write.create_array(h5write.root, 'keepcols',
                                        keepcols)
d_store = h5write.create_carray(h5write.root,
                                    'x_singular_vals', fatom,
                                    shape=d.shape, filters=filters)
s_store = h5write.create_array(h5write.root, 'knockoff_svec', svec)
colnames_store = h5write.create_array(h5write.root, 'xcolnames', xcolnames)
#u_store = h5write.create_carray(h5write.root, 'Xsvd_u', fatom,
#                                shape = u.shape, filters=filters)
with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler() as cprof:
    da.store([Xmat, d], [Xmat_store, d_store])
    del Xmat
    da.store([Xtilde], [Xtilde_store])

visualize([prof, rprof, cprof], show=False)
h5write.close()
h5read.close()
