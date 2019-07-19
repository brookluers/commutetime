import numpy as np
from scipy.optimize import minimize
from dask import array as da
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
from dask.diagnostics import visualize
import dask.config
import tables
import json

def get_ldetfun(Sigma, tol=1e-12):
    def f(svec):
        W = 2 * Sigma - np.diag(svec)
        Wev = np.linalg.eigvalsh(W)
        if any(Wev < tol):
            return np.Inf
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

def getknockoffs_dask(Xmat, u, d, vT, svec, tol):
    # X = UDV^t
    # svec: vector of knockoff tuning parameters
    v = vT.T
    d_inv = 1 / d
    d_inv[d<tol] = 0
    pdim = Xmat.shape[1]
    VDi = v * d_inv # V * D^(-1)
    SVDi = (VDi.T * svec).T # S * V * D^(-1)
    Sigma_inv_S = da.matmul(VDi, SVDi.T) # Sigma^(-1) * S
    CtC = 2 * da.diag(svec) - da.matmul(SVDi, SVDi.T)
    # Cmat = da.linalg.cholesky(Sigma_sandwich)
    cU, cD, cVt = da.linalg.svd(CtC)
    # da.dot(cU * cD, cVt) == CtC
    cDsqrt = da.sqrt(cD)
    Cmat = da.dot(cU * cDsqrt, cVt)# p by p, memory-friendly
    # Sigma_inv_S, Cmat = da.compute(Sigma_inv_S, Cmat)
    ###
    zeroes_NxP = da.broadcast_to([0], Xmat.shape, Xmat.chunks)
    X_zeroes = da.concatenate((Xmat, zeroes_NxP),
                        axis=1).rechunk({0:200000, 1:-1})
    Q, R = da.linalg.tsqr(X_zeroes)
    del v
    del u
    del VDi
    del R
    del X_zeroes
    Utilde = Q[:, -pdim:]
    ## Version using projection matrix based on SVD
    #Zrand = da.random.random(Xmat.shape,chunks=Xmat.chunks)
    #Utilde, Rz = da.linalg.qr(Zrand - da.matmul(da.matmul(u, u.T), Zrand))
    #######
    #(X - X %*% Sigma_inv_S + Utilde %*% Cmat)
    Xtilde = Xmat - da.matmul(Xmat, Sigma_inv_S) + da.matmul(Utilde, Cmat)
    return Xtilde

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
keepcols = np.arange(pdim)[np.nonzero(xnorms)]
dropcols = np.arange(pdim)[xnorms==0]
print("Dropping column with norm zero:")
xcolnames = []
for colname in xinfo['xcolnames']:
    xcolnames.append(colname)
    for dropix in dropcols:
        if xinfo['xcolnames'][colname] == dropix:
            print(colname)

xcolnames_keep = np.array(xcolnames)[keepcols]
Xmat = Xmat[:, keepcols]
pdim = Xmat.shape[1]
xnorms = xnorms[keepcols]
xmeans = xmeans[keepcols]
print("Standardizing X columns")
Xmat = Xmat / xnorms
print("design matrix has dimensions {:d} by {:d}".format(Xmat.shape[0], pdim))

G = da.matmul(Xmat.T, Xmat).compute()
ldetf = get_ldetfun(G)
ldetgrad = get_ldetgrad(G)
ldopt = minimize(lambda x: -ldetf(x),
            x0 = np.repeat(0.0001, pdim),
            jac = lambda x: -ldetgrad(x),
            bounds = [(0.0, 1.0) for i in range(pdim)])
svec = ldopt.x
u, d, vT = da.linalg.tsqr(Xmat, compute_svd = True)
tol = 1e-08
print("Truncating X singular values at {:e}".format(tol))
Xtilde = getknockoffs_dask(Xmat, u, d, vT, svec, tol=1e-08)
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
