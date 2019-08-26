import numpy as np
from scipy.optimize import minimize
import scipy.linalg
from dask import array as da

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

def get_svec_equi(G):
    evs = np.linalg.eigvalsh(G)
    pdim = G.shape[1]
    svec = np.repeat(min([1.0, 2 * evs.min()]), pdim)
    return svec

def get_svec_ldet(G):
    ldetf = get_ldetfun(G)
    ldetgrad = get_ldetgrad(G)
    pdim = G.shape[0]
    print("Maximizing log-determinant of augmented Gram matrix")
    print("\tInitial steps without using gradient...")
    init_opt = minimize(lambda x: -ldetf(x),
                x0=np.random.uniform(0.0,0.003,size=pdim),
                constraints=scipy.optimize.LinearConstraint(np.identity(pdim),lb=0,ub=1.0),
                options = {'maxiter': 10})
    #print("\tGradient-based maximization with starting value\n\t")
    #print(init_opt.x)
    ldopt = minimize(lambda x: -ldetf(x),
            x0 = init_opt.x,
            jac = lambda x: -ldetgrad(x),
            options={"maxiter": 25000},
            tol = 1e-10,
            constraints = scipy.optimize.LinearConstraint(np.identity(pdim),lb=0,ub=1.0))
    print("Final steps using gradient...")
    print("\tresult: \n\t" +
            ldopt.message + "\n\tfunciton value: " + str(ldopt.fun) + "\n\tnit = " + str(ldopt.nit) + "\n\tsolution: " + str(ldopt.x))
    svec = ldopt.x
    return svec


def getknockoffs_qr(Xmat, Qx, Rx, G, svec, tol=1e-10):
    Utilde_raw = np.random.normal(size=Xmat.shape[0] * Xmat.shape[1]).reshape(Xmat.shape)
    Utilde_raw = Utilde_raw - np.matmul(Qx, np.matmul(Qx.T, Utilde_raw))
    Utilde, Ru = scipy.linalg.qr(Utilde_raw, mode='economic')
    Smat = np.diag(svec)
    Ginv_S = scipy.linalg.solve(G, Smat)
    CtC = 2 * Smat - np.matmul(Smat, Ginv_S)
    w, v = scipy.linalg.eigh(CtC)
    w[abs(w) < tol] = 0
    Cmat = np.diag(np.sqrt(w)).dot(v.T)
    return Xmat - np.matmul(Xmat, Ginv_S) + np.matmul(Utilde, Cmat)

if __name__ == "__main__":
    np.random.seed(1)
    ptest = 70
    pgrid = np.indices((ptest, ptest))
    rowix = pgrid[0]
    colix = pgrid[1]
    rho = 0.9
    armat = rho**abs(rowix - colix)
    print("Testing det(G) optimization with AR1(" + str(rho) + ") sample covariance matrix, dimension = " + str(ptest))
    N = 1000
    Xmat = np.random.multivariate_normal(mean = np.zeros(ptest), cov = armat, size=N)
    xmeans = np.mean(Xmat, axis=0)
    Xmat = Xmat - xmeans
    xnorms = da.linalg.norm(Xmat, axis=0)
    Xmat = Xmat / xnorms
    G = Xmat.T.dot(Xmat)
    sv1 = get_svec_ldet(G)
    sv2 = get_svec_ldet(G)
    sv3 = get_svec_ldet(G)
    np.set_printoptions(precision=4, suppress=True)
    print("Solutions from 3 random starting values:\n")
    print(sv1)
    print(sv2)
    print(sv3)
    ngtest = 100
    gldf = get_ldetfun(G)
    gldg = get_ldetgrad(G)
    rel_errs = [0.0] * ngtest
    for i in np.arange(ngtest):
        checkval = np.random.uniform(0.0, 0.003, size=ptest)
        gradnorm = np.linalg.norm(gldg(checkval))
        chgrad = scipy.optimize.check_grad(gldf, gldg, checkval)
        rel_errs[i] = chgrad / gradnorm
    print("Mean relative error (" + str(ngtest) + " trials) between numerical appx. of gradient and by-hand gradient function: " + str(np.mean(rel_errs)))
