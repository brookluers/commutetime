import numpy as np
from dask import array as da
import tables

def getknockoffs_dask(Xmat, u, d, vT, svec):
    # X = UDV^t
    # svec: vector of knockoff tuning parameters
    v = vT.T
    pdim = Xmat.shape[1]
    VDi = v * (1/d) # V * D^(-1)
    SVDi = (VDi.T * svec).T # S * V * D^(-1)
    Sigma_inv_S = da.matmul(VDi, SVDi.T)
    CtC = 2 * da.diag(svec) - da.matmul(SVDi, SVDi.T)
    # Cmat = da.linalg.cholesky(Sigma_sandwich)
    cU, cD, cVt = da.linalg.svd(CtC)
    # da.dot(cU * cD, cVt) == CtC
    cDsqrt = da.sqrt(cD)
    Cmat = da.dot(cU * cDsqrt, cVt)
    ###
    zeroes_NxP = da.broadcast_to([0], Xmat.shape, Xmat.chunks)
    X_zeroes = da.concatenate((Xmat,zeroes_NxP),
                        axis=1).rechunk({0:'auto',1:-1})
    Q, R = da.linalg.qr(X_zeroes)
    Utilde = Q[:, -pdim:]
    ## Version using projection matrix based on SVD
    #Zrand = da.random.random(Xmat.shape,chunks=Xmat.chunks)
    #Utilde, Rz = da.linalg.qr(Zrand - da.matmul(da.matmul(u, u.T), Zrand))
    #######
    #(X - X %*% Sigma_inv_S + Utilde %*% Cmat)
    Xtilde = Xmat - da.matmul(Xmat, Sigma_inv_S) + da.matmul(Utilde, Cmat)
    return Xtilde


h5read = tables.open_file('regression-data.h5', mode='r')
Xmat = da.from_array(h5read.root.X)
Y = da.from_array(h5read.root.Y)
xnorms = da.linalg.norm(Xmat, axis=0)
Xmat = Xmat / xnorms
svec = np.repeat(0.001, Xmat.shape[1])
u, d, vT = da.linalg.svd(Xmat)
Xtilde = getknockoffs_dask(Xmat, u, d, vT, svec)
Xtilde.to_hdf5('knockoff-out.h5', '/Xtilde')
