# -*- coding: utf-8 -*-
"""
Settings & Hyper-parameters to change in order to try different configurations.
"""

# Path to the used CSV data:
data_path = r'C:\Users\Acer\Desktop\Data Projects\Python\Deep Q-Learning\project\master\data\stock_prices.csv'

# Equity the Agent starts with:
starting_equity = 10000

# Percentage of equity to invest (default 10%)
position_size = 0.1

# Exploration probability: (starts at 100% then decays towards "min_epsilon")
epsilon = 1

# Factor by which to decay epsilon with each timestep
   # => Close to 0: decays fast;  => close to 1: decays slow
decay = 0.9999

# Decay epsilon to the limit of 10% (so it doesn't go all the way to 0)
min_epsilon = 0.1 

# Discount rate
gamma = 0.8 

# Learning rate
alpha = 0.9

# Reward threshold to stop training once reached:
threshold = 500

# Maximum epochs to run training before stopping (whether optimized or not)
epoch_limit = 2

# How many time-steps to update the Target DQN:
timesteps = 100

# Values that will be updated by the algorithm:
_current_equity = 10_000
old_equity = 10_000
fraction_invested = 0.1


##############################################################################
# DON'T CHANGE unless for testing
last_action = 4   # initially we start with being in Cash (action #4)
##############################################################################