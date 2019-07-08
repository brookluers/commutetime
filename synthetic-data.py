import numpy as np
import pandas as pd
import tables
from patsy import dmatrix
from patsy.contrasts import Treatment

def newchunk(chunksize, measvars, catvars, ncat):
    d = pd.DataFrame(np.random.randn(chunksize, len(measvars)),
                     columns = measvars)
    for var in catvars:
        d[var] = np.random.randint(0, ncat, chunksize)
    return d



h5fname = 'design-test.h5'
h5file = tables.open_file(h5fname, mode='w')
fatom = tables.Float64Atom()

chunksize = 17000
nchunks = 20
measvars = list('abcd')
catvars = ['cat1'] #, 'cat2']
ncat = 5
lnames = np.arange(ncat )
formula = "0 + a + b + c + d + d:c + a:b + a:c + C(cat1, Treatment, levels = lnames)"
pdim = dmatrix(formula, newchunk(1, measvars,  catvars, ncat)).shape[1]

beta = np.arange(0, 1, 1.0 / pdim)# .reshape(pdim,1)

Xarray = h5file.create_earray(h5file.root, 'X', fatom,
                              shape = (0,pdim),
                              filters = tables.Filters(complevel=1, complib='zlib'),
                              expectedrows = nchunks * chunksize)
Yarray = h5file.create_earray(h5file.root, 'Y', fatom,
                              shape = (0,),
                              filters = tables.Filters(complevel=1,
                              complib='zlib'),
                              expectedrows = nchunks * chunksize)

for i in range(nchunks):
    cdata = newchunk(chunksize, measvars, catvars, ncat)
    xmat = dmatrix(formula, cdata)
    Yarray.append(xmat.dot(beta) + np.random.randn(chunksize))
    Xarray.append(xmat)

h5file.close()
