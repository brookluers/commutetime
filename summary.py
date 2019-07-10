import numpy as np
import pandas as pd
import tables
from patsy import dmatrix
from patsy.contrasts import Treatment
import json

def getfilter(chunk, Yvar, navalue):
    yesworked = chunk['WRKLSTWK'] == 2 # worked last week
    noworkhome = chunk['TRANWORK'] != 70 # did not work from home
    Y_notmissing = chunk[Yvar] != navalue # missing response
    return yesworked & noworkhome & Y_notmissing

dtypes = {
    'YEAR': 'uint16',
    'CPI99': 'float32',
    'PERWT': 'float64',
    'EDUC': 'uint8',
    'SEX': 'uint8',
    'AGE': 'uint8',
    'INCTOT': 'float32', # total pre-tax income
    'TRANTIME': 'float64',
    'TRANWORK': 'uint16',
    'WRKLSTWK': 'uint8',
    'MARST': 'uint8',
    'RACE': 'uint8',
    'BEDROOMS': 'uint8',
    'OCC2010': 'uint16',
    'OWNERSHP': 'uint8',
    # 'HCOVPRIV': 'float16', # missing values
    # 'HCOVANY': 'uint8', # missing values!
    #'METRO': 'float16', # metropolitan status of household (missing values)
    'NCHILD': 'uint8',
    'DEPARTS': 'float16'
}

## Occupational category, 2010 classification
occ10groups = [10, 500, 800, 1000, 1300, 1550, 1600, 2000,
                2100, 2200, 2600, 3000, 3600, 3700, 4000,
                4200, 4300, 4700, 5000, 6005,
                6200, 6800, 7000, 7700, 9000, 9800, 9920, 9921]
occ10labels = [
"Management, Business, Science, and Arts",#  = 10-430
"Business Operations Specialists", # = 500-730
"Financial Specialists", # = 800-950
"Computer and Mathematical", # = 1000-1240
"Architecture and Engineering", # = 1300-1540
"Technicians", # = 1550-1560
"Life, Physical, and Social Science", # = 1600-1980
"Community and Social Services", # = 2000-2060
"Legal", # = 2100-2150
"Education, Training, and Library", # = 2200-2550
"Arts, Design, Entertainment, Sports, and Media", # = 2600-2920
"Healthcare Practitioners and Technicians", # = 3000-3540
"Healthcare Support", # = 3600-3650
"Protective Service", # = 3700-3950
"Food Preparation and Serving", # = 4000-4150
"Building and Grounds Cleaning and Maintenance", #= 4200-4250
"Personal Care and Service", # = 4300-4650
"Sales and Related", # = 4700-4965
"Office and Administrative Support", # = 5000-5940
"Farming, Fishing, and Forestry", # = 6005-6130
"Construction", #  = 6200-6765
"Extraction", # = 6800-6940
"Installation, Maintenance, and Repair", # = 7000-7630
"Production", # = 7700-8965
"Transportation and Material Moving", # = 9000-9750
"Military Specific", #= 9800-9830
"Unemployed (no occupation for 5+ years) or Never Worked"] #= 9920

agegroups = [16, 25, 35, 45, 55, 65, 200]
agelabels = ['16 - 24', '25 - 34', '35 - 44', '45 - 54', '55  64', '65 or older']

departgroups = [0, 1] + list(np.arange(200,2500,200))
departlabels = ['NA', '12 - 2 a.m.', '2 - 4 a.m.',
                '4 - 6 a.m.', '6 - 8 a.m.', '8 - 10 a.m.',
                '10 a.m. - 12 p.m.', '12 - 2 p.m.',
                '2 - 4 p.m.', '4 - 6 p.m.' ,'6 - 8 p.m.',
                '8 - 10 p.m.' ,'10 p.m. - 12 a.m.']

flevels = {
    'SEX': [1,2],
    'AGECAT': [i for i in range(len(agelabels))],
    'OCC10GP': [i for i in range(len(occ10labels))],
    'NCHILD': [i for i in range(10)],
    'EDUC': [i for i in range(12)],
    'OWNERSHP': [0,1,2],
    'TRANWORK': [0, 10, 11, 12, 13, 14, 15, 20, 30, 31,
                32, 33, 34, 35, 36, 40, 50, 60],
    'MARST': [1,2,3,4,5,6],
    'RACE': [1,2,3,4,5,6,7,8,9],
    #'HCOVANY': [1,2],
    'DEPARTCAT':  [i for i in range(len(departlabels))],
    #'HCOVPRIV': [1,2],
    'BEDCAT': [i for i in range(7)]
}

flabels = {
    'SEX': ['Male','Female'],
    'AGECAT': agelabels,
    'OCC10GP' : occ10labels,
    'DEPARTCAT': departlabels,
    'OWNERSHP': ['NA', 'Owned or being bought', 'Rented'],
    'NCHILD' : ['0','1','2','3','4','5','6','7','8','9 or more'],
    'EDUC': ['N/A or no schooling', 'Nursery school to grade 4',
             'Grade 5, 6, 7, or 8', 'Grade 9','Grade 10', 'Grade 11',
             'Grade 12','1 year of college','2 years of college',
             '3 years of college', '4 years of college', '5+ years of college'],
    'TRANWORK': ['N/A', 'Auto, truck, or van','Auto','Driver',
                 'Passenger', 'Truck', 'Van', 'Motorcycle','Bus or streetcar',
                 'Bus or trolley bus','Streetcar or trolley car',
                 'Subway or elevated', 'Railroad',
                 'Taxicab','Ferryboat','Bicycle', 'Walked only','Other'],
    'MARST': ['Married, spouse present', 'Married, spouse absent',
              'Separated', 'Divorced', 'Widowed', 'Never married/single'],
    'RACE' : ['White', 'Black/African American/Negro',
               'American Indian or Alaska Native', 'Chinese',
               'Japanese', 'Other Asian or Pacific Islander',
               'Other race', 'Two major races', 'Three or more major races'],
    #'HCOVANY': ['No health insurance','With health insurance'],
    #'HCOVPRIV': ['Without private health insurance',
    #            'With private health insurance'],
    'BEDCAT': ['N/A/', 'No bedrooms','1','2','3','4','5 or more']
}

olsformula = "0 + YEAR + C(SEX, levels = flevels['SEX'])\
+ C(AGECAT, Treatment, levels=flevels['AGECAT'])\
+ C(RACE, Treatment, levels=flevels['RACE'])\
+ C(MARST, Treatment, levels=flevels['MARST'])\
+ C(BEDCAT, Treatment, levels=flevels['BEDCAT'])\
+ C(NCHILD, Treatment, levels=flevels['NCHILD'])\
+ C(EDUC, Treatment, levels=flevels['EDUC'])\
+ C(OCC10GP, Treatment, levels=flevels['OCC10GP'])\
 + C(TRANWORK, Treatment, levels=flevels['TRANWORK'])\
 + C(OWNERSHP, Treatment, levels=flevels['OWNERSHP'])\
  + INCTOT99"

derivevars = {
    'INCTOT99' : lambda xi: xi['INCTOT'] * xi['CPI99'],
    'AGECAT' : lambda xi:  pd.cut(xi['AGE'], agegroups,
                                  right = False, labels=False),
    'OCC10GP': lambda xi: pd.cut(xi['OCC2010'], occ10groups,
                                 right = False, labels = False),
    'BEDCAT' : lambda xi: xi['BEDROOMS'].where(xi['BEDROOMS'] < 6, 6),
    'DEPARTCAT': lambda xi: pd.cut(xi['DEPARTS'], departgroups,
                                    right = False, labels = False)
}


# 'implied decimals'
#    mentioned in the data dictionary were already
#    taken care of in the csv file

wgtvar = 'PERWT'
gpvars = list(flevels.keys()) + ['YEAR']
Yvar = 'TRANTIME'
Y_navalue = 0
Yhistbins = list(np.arange(0,300,2.5)) + [600]
Yhist_gpvars = ['TRANWORK', 'YEAR']
Yhist_temp = []
measvars = ['INCTOT99', 'TRANTIME']
gpdatlist = []
nrows = 0
nkept = 0

h5fname = 'regression-data.h5'
h5file = tables.open_file(h5fname, mode='w')
fatom = tables.Float64Atom()
csvfname = '~/Downloads/usa_00008.csv'
temp = pd.read_csv(csvfname, usecols=list(dtypes.keys()),
                    dtype = dtypes, nrows = 200)

pdim = dmatrix(olsformula,
                temp[getfilter(temp, Yvar, Y_navalue)].assign(**derivevars)).shape[1]
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

def doHist(chunk, Yvar, Yhistbins,
            Yhist_gpvars, wgtvar):
    chist = chunk[Yhist_gpvars + [Yvar, wgtvar]].assign(HISTBIN = lambda xi: pd.cut(xi[Yvar], Yhistbins, right=False, labels=False))
    # 3-column dataframe: grouping variable, histbin, sum(wgt)
    chist = chist[Yhist_gpvars + ['HISTBIN', wgtvar]].groupby(Yhist_gpvars + ['HISTBIN'],
                        as_index=False).aggregate(np.sum)
    return chist

for chunk in pd.read_csv(csvfname,
                            usecols = list(dtypes.keys()),
                            dtype = dtypes,
                            nrows = 200000,
                            chunksize = 50000):
    nrows += len(chunk.index)
    print("{:d} rows so far...".format(nrows))
    curfilter = getfilter(chunk, Yvar, Y_navalue)
    nkept += sum(curfilter)
    chunk = chunk[curfilter].assign(**derivevars)
    xmat = dmatrix(olsformula, chunk)
    Xarray.append(xmat)
    Yarray.append(chunk[Yvar].values)
    Warray.append(chunk[wgtvar].values)
    # Compute histograms BEFORE
    # weighting the quantitative variables
    Yhist_temp.append(doHist(chunk, Yvar,
                                Yhistbins, Yhist_gpvars, wgtvar))
    # Aggregation/summary
    # multiply quantitative variables by the person weights
    chunk[measvars] = chunk[measvars].multiply(chunk[wgtvar], axis=0)
    # sum the quantitative variables within each grouping cell
    csummary = chunk[measvars + gpvars + [wgtvar]].groupby(gpvars, as_index=False).aggregate(np.sum)
    gpdatlist.append(csummary)


Yhist = pd.concat(Yhist_temp)
gpdat = pd.concat(gpdatlist)
# eliminate duplicate grouping cells due to chunking
gpdat = gpdat.groupby(gpvars, as_index=False).aggregate(np.sum)
#  ... do the same for the grouped histograms
Yhist = Yhist.groupby(Yhist_gpvars + ['HISTBIN'], as_index=False).aggregate(np.sum)
Yhist.to_csv("Yhist.csv", index=False)
pd.DataFrame(Yhistbins, index=range(len(Yhistbins)), columns=['binval']).to_csv("Yhistbins.csv", index=True)
# label the integer-valued grouping variables
for k in flevels.keys():
    gpdat[k] = [flabels[k][flevels[k].index(i)] for i in gpdat[k]]
gpdat.info()
gpdat.to_csv("gpsummary.csv", index=False)
# Save the factor levels and labels
with open("factors.json", "w") as jf:
    jf.write(json.dumps([flevels, flabels]))
print("\nRead {:d} rows, discarded {:d} where WRKLSTWK != 2 or worked from home \n".format(nrows, nrows - nkept))
h5file.close()
