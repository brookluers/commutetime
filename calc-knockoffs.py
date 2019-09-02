import numpy as np
from scipy.optimize import minimize
import scipy.linalg
from dask import array as da
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
from dask.diagnostics import visualize
import bigknockoff as bk
import pandas as pd
import tables
import json

def do_knockoff_seq(Xmat, test_ncols, Qx, Rx, fname):
    pdim = Xmat.shape[1]
    n = Xmat.shape[0]
    Gtest = [np.matmul(Rx[:, 0:j].T, Rx[:, 0:j]) for j in test_ncols]
    svec_test = [bk.get_svec_ldet(Gj) for Gj in Gtest]
    X_psudi_test = [scipy.linalg.solve_triangular(Rx[0:j, 0:j], Qx[:, 0:j].T) for j in test_ncols]
    cnum_xtest = [scipy.linalg.norm(Rx[0:j, 0:j], ord=2) * scipy.linalg.norm(xps, ord=2) for (j, xps) in zip(test_ncols, X_psudi_test)]
    Xtilde_test = [bk.getknockoffs_qr(Xmat[:, 0:j], Qx[:, 0:j], Rx[0:j, 0:j], Gj, svec_j) for (j, Gj, svec_j) in zip(test_ncols, Gtest, svec_test)]
    cnum_xaug_test = [np.linalg.cond(np.hstack([Xmat[:, 0:j], xtj])) for (j, xtj) in zip(test_ncols, Xtilde_test)]
    pd.DataFrame({'n': n, 'p': pdim, 'ncols': test_ncols, 'cnum_xaug': cnum_xaug_test, 'cnum_x': cnum_xtest}).to_csv(fname, index=False)
    return svec_test[-1], Xtilde_test[-1]

@profile
def scale_drop(Xmat, h5write):
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
    tol = 1e-8
    ## IF using scipy QR
    #Qx, Rx, Px = scipy.linalg.qr(Xmat, mode='economic', pivoting=True)
    #dropcols_qr = Px[np.nonzero(abs(np.diag(Rx))<tol)]
    #keepcols_qr = Px[np.nonzero(abs(np.diag(Rx))>=tol)]
    #rank = np.sum(abs(np.diag(Rx)) >= tol)
    ## USING BLOCKED QR
    Qx, Rx, PImat = bk.tsqr_pivot_seq(Xmat)
    #Rx = Rx[0:rank, 0:rank]
    #Qx = Qx[:, 0:rank]
    keepcols_qr = np.argmax(PImat, axis=0)
    dropmask = np.ones(Xmat.shape[1], dtype=bool)
    dropmask[keepcols_qr] = False
    dropcols_qr = np.arange(Xmat.shape[1])[dropmask]
    rank = keepcols_qr.shape[0]
    print("Dropping columns based on pivoted QR:")
    print("\t" + "\n\t".join(xcolnames_keep[dropcols_qr]))
    xnorms = xnorms[keepcols_qr]
    xmeans = xmeans[keepcols_qr]
    Xmat = Xmat[:, keepcols_qr]
    xcolnames_keep = xcolnames_keep[keepcols_qr]
    #keepcols_store = h5write.create_array(h5write.root, 'keepcols',
    #                                        keepcols)
    #cols_orig_store = h5write.create_array(h5write.root, 'xcolnames_all', xcolnames)
    #cols_keep_store = h5write.create_array(h5write.root, 'xcolnames_keep', xcolnames_keep)
    #da.store([xcolnames, xcolnames_keep], [cols_orig_store, cols_keep_store])
    return Xmat, Qx, Rx

@profile
def ko_s(Xmat, Qx, Rx, G):
    svec = bk.get_svec_ldet(G)
    Xtilde = bk.getknockoffs_qr(Xmat, Qx, Rx, G, svec)
    return Xtilde, svec

if __name__ == "__main__":
    xinfo = {}
    with open("fpaths.json") as fpj:
        FPATHS = json.load(fpj)
    with open(FPATHS['xcolnames_json']) as jf:
        xinfo['xcolnames'] = json.load(jf)
    with open(FPATHS['xtermslices_json']) as jf:
        xinfo['xtermcols'] = json.load(jf)
    h5read = tables.open_file(FPATHS['designmat_h5'], mode='r')
    h5write = tables.open_file(FPATHS['knockoff_h5'], mode='w')
    fatom = tables.Float64Atom()
    filters = tables.Filters(complevel=1, complib='zlib')
    Xmat = da.from_array(h5read.root.X)
    Y = da.from_array(h5read.root.Y)
    Wgt = da.from_array(h5read.root.wgt)
    Xmat, Qx, Rx = scale_drop(Xmat, h5write)
    pdim = Xmat.shape[1]
    print("design matrix has dimensions {:d} by {:d}".format(Xmat.shape[0], pdim))
    # pseudo-inverse:  (X^t X)^(-1) X^t = R^(-1) Q^T
    # X_pseudo_inv = scipy.linalg.solve_triangular(Rx, Qx.T)
    # condition number:  || X|| * ||X_pseudo_inv||
    #X_cnum = scipy.linalg.norm(Rx, ord=2) * scipy.linalg.norm(X_pseudo_inv, ord=2)
    G = np.matmul(Rx.T, Rx)
    #test_ncols = np.concatenate([[60], np.arange(90, Xmat.shape[1] - 10, 10), [Xmat.shape[1]]])
    #svec, Xtilde = do_knockoff_seq(Xmat, test_ncols, Qx, Rx, 'Xaug-condition.csv')
    Xtilde, svec = ko_s(Xmat, Qx, Rx, G)
    #s_store = h5write.create_array(h5write.root, 'knockoff_svec', svec)
    # ((G - np.diag(svec)) - np.matmul(Xtilde.T, Xmat)).compute()
    # np.matmul(Xtilde.T,Xtilde).compute() - G
    #Xtilde_store = h5write.create_carray(h5write.root, 'Xtilde', fatom,
    #                                     shape = Xtilde.shape,
    #                                     filters = filters)
    #Xmat_store = h5write.create_carray(h5write.root, 'X', fatom,
    #                                    shape = Xmat.shape,
    #                                    filters=filters)
    #with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler() as cprof:
    #    da.store([Xmat, Xtilde], [Xmat_store, Xtilde_store])
    #da.store([svec], [s_store])
    #visualize([prof, rprof, cprof], show=False)
    h5write.close()
    h5read.close()
