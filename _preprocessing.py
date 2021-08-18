# -*- coding: utf-8 -*-
'''
Script for data wrangling and preprocessing.
'''

from    sklearn  import preprocessing
import  pandas   as     pd
import  numpy    as     np

# Number of missing values:
def check_na(dataframe):
    return dataframe.isna().sum()

# Checking if we didn't miss anything (na and not na add up to total)
def Assert(dataframe):
    assert dataframe['Close'].isna().sum() + dataframe['Close'].notna().sum()\
        == len(dataframe)

# Drop adj close column
def Drop_Adj(dataframe):
    return dataframe.drop('Adj Close', axis=1)


# Interpolating using Akima splines:
def akima(dataframe):
    return dataframe.interpolate(method='akima')

# Copy the original returns to dt before processing so we keep them for rewards:
#dt['DoD return'] = df['DoD return']

def Normalize(df):
    # Normalize "Volume" feature:
    column_names = df.columns # store column names to restore them after processing
    
    norm_volume = pd.DataFrame(preprocessing.normalize(np.array(df['Volume']).reshape(1,-1)))
    df['Volume'] = np.array(norm_volume).reshape(-1,1)
    df['Volume'] = df['Volume']*10
    df.columns = column_names
    df = df.iloc[:,1:len(df.columns)]