import numpy as np
import pandas as pd
import tables
from patsy import (dmatrix, ModelDesc, Term, EvalFactor, LookupFactor)
from patsy.contrasts import Treatment
from itertools import combinations
import json
import acs_config as cfg

def doHist(chunk, Yvar, Yhistbins,
            Yhist_gpvars, wgtvar):
    chist = chunk[Yhist_gpvars + [Yvar, wgtvar]].assign(HISTBIN = lambda xi: pd.cut(xi[Yvar], Yhistbins, right=False, labels=False))
    # 3-column dataframe: grouping variable, histbin, sum(wgt)
    chist = chist[Yhist_gpvars + ['HISTBIN', wgtvar]].groupby(Yhist_gpvars + ['HISTBIN'],as_index=False).aggregate(np.sum)
    return chist

if __name__ == "__main__":
    ## Build the regression formula
    catvars = list(cfg.flevels.keys())
    with open("fpaths.json") as fpj:
        FPATHS = json.load(fpj)
    numvar_evals = ["I(YEAR - 2000)", "INCTOT99"]
    catvar_evals = ["C(" + cv + ", Treatment, levels=cfg.flevels['" + cv + "'])" for cv in catvars]
    desc = ModelDesc([],[Term([EvalFactor(v)]) for v in numvar_evals])
    desc.rhs_termlist += [Term([EvalFactor(v)]) for v in catvar_evals]
    # Interactions
    interact_order = 2
    catvar_interact = ['SEX','AGECAT','RACE']
    print("Including all order-" + str(interact_order) + " interactions of the following variables:\n\t" + ", ".join(catvar_interact + numvar_evals))
    interact_evals = numvar_evals + [catvar_evals[i] for i in [catvars.index(v) for v in catvar_interact]]
    desc.rhs_termlist += [Term([EvalFactor(v) for v in list(comb)]) for comb in combinations(interact_evals, interact_order)]
    # 'implied decimals'
    #    mentioned in the data dictionary were already
    #    taken care of in the csv file
    gpvars = list(cfg.flevels.keys()) + ['YEAR']
    Yhistbins = list(np.arange(0,300,5)) + [600]
    Yhist_gpvars = ['TRANWORK', 'YEAR']
    Yhist_temp = []
    measvars = ['INCTOT99', 'TRANTIME']
    gpdatlist = []
    nrows = 0
    nkept = 0
    h5file = tables.open_file(FPATHS["designmat_h5"], mode='w')
    fatom = tables.Float64Atom()
    temp = pd.read_csv(FPATHS["raw_csv"], usecols=list(cfg.rawvars_dtypes.keys()),
                        dtype = cfg.rawvars_dtypes, nrows = 200)
    tempdesign = dmatrix(desc,
                    temp[cfg.getfilter(temp, cfg.Yvar, cfg.Y_navalue)].assign(**cfg.derivevars))
    pdim = tempdesign.shape[1]
    with open(FPATHS["xcolnames_json"], "w") as xj:
        xj.write(json.dumps(tempdesign.design_info.column_name_indexes))
    tnameslices = tempdesign.design_info.term_name_slices
    tnamedict = {}
    xindices = list(range(pdim))
    for tn in tnameslices:
        tnamedict[tn]  = xindices[tnameslices[tn]]
    with open(FPATHS["xtermslices_json"], "w") as xj:
        xj.write(json.dumps(tnamedict))

    Xarray = h5file.create_earray(h5file.root, 'X', fatom,
                                  shape = (0,pdim),
                                  filters = tables.Filters(complevel=1, complib='zlib'),
                                  expectedrows = 18000000)
    Yarray = h5file.create_earray(h5file.root, 'Y', fatom,
                                  shape = (0,),
                                  filters = tables.Filters(complevel=1,
                                  complib='zlib'),
                                  expectedrows = 18000000)
    Warray = h5file.create_earray(h5file.root, 'wgt', fatom,
                                  shape = (0,),
                                  filters = tables.Filters(complevel=1,complib='zlib'),
                                  expectedrows = 18000000)
    for chunk in pd.read_csv(FPATHS["raw_csv"],
                                usecols = list(cfg.rawvars_dtypes.keys()),
                                dtype = cfg.rawvars_dtypes,
                                nrows = 250000,
                                chunksize = 75000):
        nrows += len(chunk.index)
        print("{:d} rows so far...".format(nrows))
        curfilter = cfg.getfilter(chunk, cfg.Yvar, cfg.Y_navalue)
        nkept += sum(curfilter)
        chunk = chunk[curfilter].assign(**cfg.derivevars)
        xmat = dmatrix(desc, chunk)
        Xarray.append(xmat)
        Yarray.append(chunk[cfg.Yvar].values)
        Warray.append(chunk[cfg.wgtvar].values)
        # Compute histograms BEFORE
        # weighting the quantitative variables
        Yhist_temp.append(doHist(chunk, cfg.Yvar,
                                    Yhistbins, Yhist_gpvars, cfg.wgtvar))
        # Aggregation/summary
        # multiply quantitative variables by the person weights
        chunk[measvars] = chunk[measvars].multiply(chunk[cfg.wgtvar], axis=0)
        # sum the quantitative variables within each grouping cell
        csummary = chunk[measvars + gpvars + [cfg.wgtvar]].groupby(gpvars, as_index=False).aggregate(np.sum)
        gpdatlist.append(csummary)

    Yhist = pd.concat(Yhist_temp)
    gpdat = pd.concat(gpdatlist)
    # eliminate duplicate grouping cells due to chunking
    gpdat = gpdat.groupby(gpvars, as_index=False).aggregate(np.sum)
    #  ... do the same for the grouped histograms
    Yhist = Yhist.groupby(Yhist_gpvars + ['HISTBIN'], as_index=False).aggregate(np.sum)
    Yhist.to_csv(FPATHS["Yhist_csv"], index=False)
    pd.DataFrame(Yhistbins, index=range(len(Yhistbins)), columns=['binval']).to_csv(FPATHS["Yhist_bins_csv"], index=True)
    # label the integer-valued grouping variables
    for k in cfg.flevels.keys():
        gpdat[k] = [cfg.flabels[k][cfg.flevels[k].index(i)] for i in gpdat[k]]
    gpdat.info()
    gpdat.to_csv(FPATHS["summary_out_csv"], index=False)
    # Save the factor levels and labels
    with open(FPATHS["factor_vars_json"], "w") as jf:
        jf.write(json.dumps([cfg.flevels, cfg.flabels]))
    print("\nRead {:d} rows, discarded {:d} where WRKLSTWK != 2 or worked from home \n".format(nrows, nrows - nkept))
    h5file.close()
