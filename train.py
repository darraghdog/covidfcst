'''
Reference code : https://www.kaggle.com/nkiith/covid19
Date pull : 
    Pull data from https://github.com/nytimes/covid-19-data
    git clone https://github.com/nytimes/covid-19-data
    git pull covid-19-data
'''

import os
import sys
import optparse
from datetime import date

import warnings
warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import tqdm as tqdm

pd.options.display.max_rows = 5000
pd.options.display.max_columns = 50
pd.options.display.width = 500

'''
Params
'''
parser = optparse.OptionParser()
parser.add_option('-a', '--outpath', action="store", dest="outpath", help="Path to dump objects", default="out")
parser.add_option('-b', '--datapath', action="store", dest="datapath", help="Fold for split", default="data/covid-19-data/")
parser.add_option('-c', '--rootpath', action="store", dest="rootpath", help="Rootpath", default=".")
parser.add_option('-d', '--preddays', action="store", dest="preddays", help="Test length", default="2")
parser.add_option('-e', '--test', action="store", dest="test", help="Test history", default="T")
parser.add_option('-f', '--seed', action="store", dest="seed", help="Training seed", default="42")
parser.add_option('-g', '--valsize', action="store", dest="valsize", help="Validation size - default 0", default="0.0")

options, args = parser.parse_args()

print('Load params')
for (k,v) in options.__dict__.items():
    print('{}{}'.format(k.ljust(20), v))
    
ROOTPATH = options.rootpath
DATAPATH = options.datapath
OUTPATH = options.outpath
TEST_DURATION = int(options.preddays)
TEST = 't' in options.test.lower()
randomState = int(options.seed) # For train test split
valSize = float(options.valsize) # 90:10 ratio >> for final testing

'''
ROOTPATH = '/Users/dhanley2/Documents/rework/covid/jhopkins'
sys.path.append(ROOTPATH)
DATAPATH = os.path.join(ROOTPATH, "data/covid-19-data/")
OUTPATH = os.path.join(ROOTPATH, "out")
TEST_DURATION = 2 # days
TEST = False#True
randomState = 42
valSize = 0.0 # 0.1
'''

sys.path.append(ROOTPATH)
from utils.utils import geoTemplate, dateFeatures

dtypes = {'fips':str, 
          'county':str, 
          'state':str, 
          'cases':int, 
          'deaths':int}
casesdf = pd.read_csv(os.path.join(DATAPATH, "us-counties.csv"), \
                      dtype = dtypes, 
                      parse_dates = ['date'])
geocols = ['state', 'county', 'fips']
timecols = ['month', 'day']

'''
# Fill in missing dates
'''
geodf = casesdf[geocols].drop_duplicates()
templatedf =  geoTemplate(casesdf, geodf)
casesdf = pd.merge(casesdf, templatedf, 
                   on = templatedf.columns.tolist(), 
                   how = 'outer')
casesdf['cases'] = casesdf['cases'].fillna(-1)
casesdf['deaths'] = casesdf['deaths'].fillna(-1)
casesdf = casesdf.sort_values(geocols+['date'])
casesdf['new_cases'] = casesdf.groupby(geocols)['cases'].diff(1).fillna(-1)

'''
Train test time based split
'''
if TEST:
    trainmax = casesdf.date.max() - pd.DateOffset(TEST_DURATION)
    trndf = casesdf.query('date <= @trainmax')
    tstdf = casesdf.query('date > @trainmax')
else:
    # Create a test data set for the new dates
    trainmax = casesdf.date.max() - pd.DateOffset(TEST_DURATION)
    trndf = casesdf.copy()
    tstdf = casesdf.query('date > @trainmax')
    tstdf['date'] = tstdf['date'] + pd.DateOffset(days=TEST_DURATION)
    tstdf['cases'] = tstdf['new_cases'] = tstdf['deaths'] = np.nan
    
'''
# Join up and create features
'''
trndf['type'] = 'train'
tstdf['type'] = 'test'
alldf = pd.concat([trndf, tstdf], 0).copy()
alldf = dateFeatures(alldf)
alldf['fcstidx'] = range(alldf.shape[0])
y_tst = alldf[['fcstidx', 'cases', 'deaths']].loc[alldf.type=='test']
alldf.loc[alldf.type=='test', ['cases', 'deaths']] = np.nan

'''
Offset days
'''
alldf = alldf.sort_values(geocols + timecols + ['fcstidx']).reset_index(drop=True)

for offset in range(0,8):
    offset += (TEST_DURATION)
    alldf[f'c_diff{offset}'] = (alldf.groupby(geocols)['cases'].shift(offset) - \
                                alldf.groupby(geocols)['cases'].shift(offset+1)) \
                                 .fillna(-1)
    alldf[f'c_off{offset}'] = alldf.groupby(geocols)['cases'].shift(offset).fillna(-1)

alldf['new_cases'] = np.nan
idx = alldf['type'] == "train"
# alldf.sort_values(['county', 'month', 'day'])[alldf.columns[:10].tolist()+['new_cases']].head(100)
# alldf.sort_values(['county', 'month', 'day']).head(80).tail(10).transpose()

'''
Re split datasets
'''
trndf = alldf.query('type == "train"')
tstdf = alldf.query('type == "test"').set_index('fcstidx').loc[y_tst.fcstidx]
ytrndf = trndf[['cases', 'deaths']]

# Final validation
X_train, X_val, y_train, y_val = train_test_split(trndf, ytrndf, \
                                                    test_size=valSize, 
                                                    random_state=randomState)
filtcols = alldf.filter(regex='c_off|c_diff').columns.tolist()

X_train = X_train[filtcols]
X_val   = X_val[filtcols]
X_test  = tstdf[filtcols]

'''
# Adaboost Regressor >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
'''
treeDepth = 10 # Fixed
mdl = DecisionTreeRegressor(max_depth=treeDepth) # This is fixed
param_grid = {
    'n_estimators': [100],#, 250, 500],
    'learning_rate': [0.01]#, 0.001]
                }
regrMdl = AdaBoostRegressor(base_estimator=mdl)
clf = RandomizedSearchCV(estimator = regrMdl, 
                         param_distributions = param_grid,
                         n_iter = 1, 
                         cv = 3, 
                         verbose=1, 
                         random_state=randomState, 
                         n_jobs = -1)
clf.fit(X_train, y_train.cases)
if valSize>0:
    y_predval = clf.predict(X_val)
y_predtst = clf.predict(X_test)

if TEST:
    if valSize>0:
        print(f'Val MAE   : {mae(y_val.cases.values, y_predval):.4f}')
        print(f'Val RMSE  : {rmse(y_val.cases.values, y_predval):.4f}')
    print(f'Test MAE  : {mae(y_tst.cases.values, y_predtst):.4f}')
    print(f'Test RMSE : {rmse(y_tst.cases.values, y_predtst):.4f}')
else:
    tstdf['cases'] = y_predtst
    outcols = geocols + ['date', 'cases', 'type']
    preddf = pd.concat([trndf, tstdf], 0)[outcols]
    preddf['cases'] = preddf['cases'].clip(0, 999999999).astype(np.int32)
    preddf = preddf.sort_values(geocols+['date']).reset_index(drop=True)
    today = date.today().strftime("%m%d%Y")
    fname = os.path.join(OUTPATH, f'forecast_{TEST_DURATION}day_{today}.csv')
    preddf.to_csv(fname, index = False)
    
    

'''
tstdf['pred_cases'] = y_predtst
tstdf['cases'] = y_tst.set_index('fcstidx').loc[tstdf.index].cases

tstdf.query('county == "Westchester"')['pred_cases']

tstdf.query('county == "Westchester"').transpose()
tstdf.query('county == "Calhoun"').sort_values('state').transpose()

pd.options.display.width = 500
pd.options.display.float_format = '{:,.0f}'.format

pd.concat([tstdf.filter(regex='state|county|month|day'),
          tstdf.filter(regex='c_off').astype(np.int),
          tstdf.filter(regex='cases')],1) \
    .query('state == "New York"') \
    .set_index('county') \
    .sort_values(['county', 'day'])
'''