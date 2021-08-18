# -*- coding: utf-8 -*-
"""
Utility script for descriptive statistics.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Pull dependencies:
dt = globals()['df']
cumul_reward = globals()['cum_reward']

# Plot price:
x = range(0, len(dt)) 
y = dt['Close']

fig, ax = plt.subplots()
ax.plot(x,y)
ax.set(xlabel='Days', ylabel="Stock price",
       title='Evolution of the stock price')
ax.grid()
plt.show()

# Plot price over cumulative rewards:
x = range(0,len(cumul_reward))
fig, ax = plt.subplots()
ax.plot(x, pd.Series(cumul_reward)/5.4)

x = range(0, len(dt)) 
y = dt['Close']
ax.plot(x,y)

plt.title('Rewards against actual price')
plt.xlabel('Days')
plt.ylabel('Rewards v.s. Price')
plt.legend(['Rewards', 'Price'])
plt.show()

# Descriptive stats
dt.describe()

# Plot histogram of returns:
mu = np.mean(dt['DoD return'])
sigma = np.std(dt['DoD return'])

num_bins = 70
fig, ax = plt.subplots()
n, bins, patches = ax.hist(dt['DoD return'], num_bins, density=1)
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')

ax.set_xlabel('Daily returns (%)')
ax.set_ylabel('Density')
ax.set_title(r'Distribution of Daily Returns')
plt.show()

# Auto-correlation:
def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))
autocorr(dt['Close'])