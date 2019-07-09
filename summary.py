import numpy as np
import pandas as pd
import tables
from patsy import dmatrix
from patsy.contrasts import Treatment
import json

def getfilter(chunk):
    # worked last week, not from home
    yesworked = chunk['WRKLSTWK'] == 2
    noworkhome = chunk['TRANWORK'] != 70 # did not work from home
    return yesworked & noworkhome

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
    # 'HCOVPRIV': 'float16',
    #'METRO': 'float16', # metropolitan status of household
    'NCHILD': 'uint8'
    # 'DEPARTS': 'float16'
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

flevels = {
    'SEX': [1,2],
    'AGECAT': [i for i in range(len(agelabels))],
    'OCC10GP': [i for i in range(len(occ10labels))],
    'NCHILD': [i for i in range(10)],
    'EDUC': [i for i in range(12)],
    'TRANWORK': [0, 10, 11, 12, 13, 14, 15, 20, 30, 31,
                32, 33, 34, 35, 36, 40, 50, 60],
    'MARST': [1,2,3,4,5,6],
    'RACE': [1,2,3,4,5,6,7,8,9],
    #'HCOVPRIV': [1,2],
    'BEDCAT': [i for i in range(7)]
}

flabels = {
    'SEX': ['Male','Female'],
    'AGECAT': agelabels,
    'OCC10GP' : occ10labels,
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
  + INCTOT99"


derivevars = {
    'INCTOT99' : lambda xi: xi['INCTOT'] * xi['CPI99'],
    'AGECAT' : lambda xi:  pd.cut(xi['AGE'], agegroups,
                                  right = False, labels=False),
    'OCC10GP': lambda xi: pd.cut(xi['OCC2010'], occ10groups,
                                 right = False, labels = False),
    'BEDCAT' : lambda xi: xi['BEDROOMS'].where(xi['BEDROOMS'] < 6, 6)
}


h5fname = 'regression-data.h5'
h5file = tables.open_file(h5fname, mode='w')
fatom = tables.Float64Atom()

csvfname = '~/Downloads/usa_00008.csv'
temp = pd.read_csv(csvfname, usecols=list(dtypes.keys()),
                    dtype = dtypes, nrows = 200)
pdim = dmatrix(olsformula, temp[getfilter(temp)].assign(**derivevars)).shape[1]
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
# 'implied decimals' were already accounted for in CPI99 and PERWT


wgtvar = 'PERWT'
gpvars = list(flevels.keys()) + ['YEAR']
Yvar = 'TRANTIME'
measvars = ['INCTOT99', 'TRANTIME']
gpdatlist = []
nrows = 0
nkept = 0

for chunk in pd.read_csv(csvfname,
                            usecols = list(dtypes.keys()),
                            dtype = dtypes,
                            nrows = 90000,
                            chunksize = 20000):
    nrows += len(chunk.index)
    print("{:d} rows so far...".format(nrows))
    curfilter = getfilter(chunk)
    nkept += sum(curfilter)
    chunk = chunk[curfilter].assign(**derivevars)
    xmat = dmatrix(olsformula, chunk)
    Xarray.append(xmat)
    Yarray.append(chunk[Yvar].values)
    Warray.append(chunk[wgtvar].values)
    # Aggregation/summary
    # multiply quantitative variables by the person weights
    chunk[measvars] = chunk[measvars].multiply(chunk[wgtvar], axis=0)
    csummary = chunk[measvars + gpvars + [wgtvar]].groupby(gpvars, as_index=False).aggregate(np.sum)
    gpdatlist.append(csummary)

gpdat = pd.concat(gpdatlist)
# eliminate duplicate groupby combinations due to chunking
gpdat = gpdat.groupby(gpvars, as_index=False).aggregate(np.sum)

# label the integer-valued grouping variables
for k in flevels.keys():
    gpdat[k] = [flabels[k][flevels[k].index(i)] for i in gpdat[k]]

gpdat.info()
with open("factors.json", "w") as jf:
    jf.write(json.dumps(flevels))
    jf.write(json.dumps(flabels))
print("\nRead {:d} rows, discarded {:d} where WRKLSTWK != 2 or worked from home \n".format(nrows, nrows - nkept))
gpdat.to_csv("gpsummary.csv", index=False)

h5file.close()
