
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def pickle_write(obj, name):
    ''' pickle_write'''
    with open(name, 'wb') as handle:
        pk.dump(obj, handle, protocol=pk.HIGHEST_PROTOCOL)

def geoTemplate(df, geodf):
    idx = pd.date_range(df.date.min(), df.date.max())
    tileidx = idx.tolist() * geodf.shape[0]
    geodf = pd.concat([geodf]*len(idx))
    geodf.index = sorted(tileidx)
    geodf = geodf.reset_index().rename(columns={'index': "date"})
    return geodf

def dateFeatures(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day  
    # df = df.drop(['date'], axis=1)
    fillcols = ['county','state','fips']
    df[fillcols] = df[fillcols].fillna('dummy')
    return df

# Finding RMSE
def ErrorCalc(mdl, ref, tag):
    relError = np.abs(mdl - ref)/ np.abs(ref+1)
    MeanErrorV = np.mean(relError)
    print(tag + ': Mean Rel Error in %: ', MeanErrorV * 100)
    return MeanErrorV

# Since cumulative prediction >> This script is not used for Kaggle dataset
def AdjustingErrorsOutliers(tempPred, df) :
    tempPred = np.round(tempPred)
    tempPrev = df['day5'].to_numpy() # Next cumulative prediction must be more than or equal to previous
    for i in range(len(tempPred)):
        if tempPred[i] < tempPrev[i] : # Since cumulative prediction
            tempPred[i] = tempPrev[i]
    return tempPred
