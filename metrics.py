# -*- coding: utf-8 -*-
"""
This script is used to generate some input features used as predictors.

They're based on multiple financial indicators.
"""


import ta
import numpy as np
import pandas as pd

# n-th percentile price:
def nth_percentile(df):
    df['Percentile Price'] = ''
    n = 10 # 4th percentile price: changeable parameter
    for i in range(n, len(df)-n):
        n_list = df['Close'][i-n:i] # Gives the last n values for every i-th you input
        pr = df['Close'][i] #price
        counter = 0
        
        for o in range(i-n, i):
            if pr > n_list[o]:
                counter += 1
            else:
                pass
        percentile_price = counter/n
        df['Percentile Price'][i+1] = percentile_price
    
    
    # Fill in the gaps:
    df['Percentile Price'][0:n+1] = 0.9
    df['Percentile Price'][len(df)-n+1:len(df)]=0.1



# Difference between closes:
def close_delta(df):
    df['Close diff'] = ''
    for i in range(1,len(df)):
        df['Close diff'][i] = df['Close'][i] - df['Close'][i-1]
    df['Close diff'][0] = 0


# Initialize VWAP object
def VWAP(df):
    VWAP = ta.volume.VolumeWeightedAveragePrice(high   = df['High'],
                                                low    = df['Low'],
                                                close  = df['Close'],
                                                volume = df['Volume'],
                                                window = 14, fillna = True)
    # Add it to dataframe
    df['VWAP'] = VWAP.volume_weighted_average_price()



# Bollinger bands:
def Bollinger(df):
    # Initialize Bollinger Bands Indicator
    bb = ta.volatility.BollingerBands(close=df["Close"],
                                      window =20, window_dev=1, fillna = True)
    # Add Bollinger Bands features
    df['bb_ma'] = bb.bollinger_mavg()
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    
    # Add Bollinger Band high/ low indicator: 1 if price is higher and 0 if below:
    df['bb_hi'] = bb.bollinger_hband_indicator()
    df['bb_li'] = bb.bollinger_lband_indicator()



# Add Day-over-Day return: (for rewards)
def DoD(df):
    df['DoD return']= ''
    
    # Calculate the daily return for all days:
    for i in range(1, len(df)):
       df['DoD return'][i] = float(((df['Close'][i]-df['Close'][i-1]) / df['Close'][i-1])*100 )
    
    # Fill the initial empty observation with the mean of the next 3:
    df['DoD return'][0] = np.mean(df['DoD return'][1:4])
    
    # converting new column to numeric
    df['DoD return'] = pd.to_numeric(df['DoD return'])                                                     