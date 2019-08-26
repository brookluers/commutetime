import pandas as pd

rawvars_dtypes = {
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
    # 'METRO': 'uint8', # metropolitan status of household (missing values)
    'REGION': 'uint16', # census region and divison
    #'INCINVST': 'float32', # 6-digit investment income/loss
    'CARPOOL': 'uint16',
    'NCHILD': 'uint8',
    'NCOUPLES': 'uint8',
    'DEPARTS': 'float16',
    'CLASSWKRD': 'uint16',
    'FOODSTMP': 'uint8',
    'PROPTX99': 'uint16'
    #'PWTYPE' : 'uint8' # # entirely missing??
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

departgroups = [0, 1] + list(range(200,2500,200))
departlabels = ['NA', '12 - 2 a.m.', '2 - 4 a.m.',
                '4 - 6 a.m.', '6 - 8 a.m.', '8 - 10 a.m.',
                '10 a.m. - 12 p.m.', '12 - 2 p.m.',
                '2 - 4 p.m.', '4 - 6 p.m.' ,'6 - 8 p.m.',
                '8 - 10 p.m.' ,'10 p.m. - 12 a.m.']

division_groups = [11, 12, 13, 21, 22, 23, 31, 32, 33, 34, 41, 42, 43, 91, 92, 97, 99]
division_labels = ["New England Division", "Middle Atlantic Division",
    "Mixed Northeast Divisions (1970 Metro)", "East North Central Div.",
    "West North Central Div.", "Mixed Midwest Divisions (1970 Metro)",
    "South Atlantic Division", "East South Central Div.", "West South Central Div.",
    "Mixed Southern Divisions (1970 Metro)", "Mountain Division",
    "Pacific Division", "Mixed Western Divisions (1970 Metro)", "Military/Military reservations",
    "PUMA boundaries cross state lines-1% sample", "State not identified", "Not identified"]
classwkrd_levels = [0, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
classwkrd_labels = ['N/A', 'Self-employed', 'Employer', 'Working on own account',
    'Self-employed, not incorporated', 'Self-employed, incorporated',
    'Works for wages', 'Works on salary (1920)', 'Wage/salary, private',
    'Wage/salary at non-profit', 'Wage/salary, government', 'Federal govt employee',
    'Armed forces', 'State govt employee', 'Local govt employee', 'Unpaid family worker']
flevels = {
    'SEX': [1,2],
    'AGECAT': [i for i in range(len(agelabels))],
    'OCC10GP': [i for i in range(len(occ10labels))],
    'NCHILD': [i for i in range(10)],
    'NCOUPLES': [i for i in range(10)],
    'EDUC': [i for i in range(12)],
    'OWNERSHP': [0,1,2],
    'TRANWORK': [0, 10, 11, 12, 13, 14, 15, 20, 30, 31,
                32, 33, 34, 35, 36, 40, 50, 60],
    'MARST': [1,2,3,4,5,6],
    'RACE': [1,2,3,4,5,6,7,8,9],
    #'HCOVANY': [1,2],
    'DEPARTCAT':  [i for i in range(len(departlabels))],
    #'HCOVPRIV': [1,2],
    'BEDCAT': [i for i in range(7)],
    'REGION': division_groups,
    'CARPOOL': [0,1,2,3,4,5],
    'CLASSWKRD': classwkrd_levels,
    'FOODSTMP': [0,1,2],
    'PROPTXPAID': [0, 1]
}

flabels = {
    'SEX': ['Male','Female'],
    'AGECAT': agelabels,
    'OCC10GP' : occ10labels,
    'DEPARTCAT': departlabels,
    'OWNERSHP': ['NA', 'Owned or being bought', 'Rented'],
    'NCHILD' : ['0','1','2','3','4','5','6','7','8','9 or more'],
    'NCOUPLES': ['0 or N/A', '1','2','3','4','5','6','7','8','9'],
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
    'BEDCAT': ['N/A', 'No bedrooms','1','2','3','4','5 or more'],
    'REGION': division_labels,
    'CARPOOL': ['N/A/', 'Drives alone','Carpools',
        'Shares driving', 'Drives others only', 'Passenger only'],
    'CLASSWKRD': classwkrd_labels,
    'FOODSTMP': ['N/A', 'No', 'Yes'],
    'PROPTXPAID': ['NA or zero real estate taxes','Paid real estate taxes']
}

derivevars = {
    'INCTOT99' : lambda xi: xi['INCTOT'] * xi['CPI99'],
    'AGECAT' : lambda xi:  pd.cut(xi['AGE'], agegroups,
                                  right = False, labels=False),
    'OCC10GP': lambda xi: pd.cut(xi['OCC2010'], occ10groups,
                                 right = False, labels = False),
    'BEDCAT' : lambda xi: xi['BEDROOMS'].where(xi['BEDROOMS'] < 6, 6),
    'DEPARTCAT': lambda xi: pd.cut(xi['DEPARTS'], departgroups,
                                    right = False, labels = False),
    'PROPTXPAID': lambda xi: 1 * (xi['PROPTX99'] >= 2)
}

wgtvar = 'PERWT'
Yvar = 'TRANTIME'
Y_navalue = 0

def getfilter(chunk, Yvar, navalue):
    yesworked = chunk['WRKLSTWK'] == 2 # worked last week
    noworkhome = chunk['TRANWORK'] != 70 # did not work from home
    Y_notmissing = chunk[Yvar] != navalue # missing response
    return yesworked & noworkhome & Y_notmissing
