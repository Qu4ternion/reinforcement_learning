# -*- coding: utf-8 -*-
"""
Script for backtesting the performance of the Agent after training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
###########
Backtesting
###########
'''

# Collect necessary global variables:
env = globals()['env']
orders = globals()['orders']
model = globals()['model']
df = env.df

# Transform the numbers in "Orders" list to their respective order type for
# interpretability:
for i in range(len(orders)):
    if orders[i] == 0:
        orders[i] = "Long"
    
    elif orders[i] == 1:
        orders[i] = "Short"
        
    elif orders[i] == 2:
        orders[i] = "Hold"
    
    elif orders[i] == 3:
        orders[i] = "Close"
        
    elif orders[i] == 4:
        orders[i] = "Cash"

# % of time spent Holding, long, short, cash:
Hold = []
Long = []
Short = []
Cash = []
Close = []

for i in range(len(orders)):
    if orders[i] == 'Hold':
        Hold.append(orders[i])
        
    if orders[i] == 'Long':
        Long.append(orders[i])
    
    if orders[i] == 'Short':
        Short.append(orders[i])
    
    if orders[i] == 'Cash':
        Cash.append(orders[i])
        
    if orders[i] == 'Close':
        Close.append(orders[i])
    
print("- % of periods spent Holding:" , len(Hold)/len(orders)*100)
print("- % of periods spent Long:"    , len(Long)/len(orders)*100)
print("- % of periods spent Short:"   , len(Short)/len(orders)*100)
print("- % of periods spent in Cash:" , len(Cash)/len(orders)*100)
print("- % of periods spent Closing:" , len(Close)/len(orders)*100)
print("")
print("- Number of Buys:", orders.count('Long'))
print("- Number of Sells:",orders.count('Short'))
print("- Total trades:", orders.count('Long') + orders.count('Short'))

# Original OHLC data:
ohlc = df['Close'].reset_index(drop= True)

# orders in LSCH format: (lsch = Long, Short, Cash, Hold.)
lsch = pd.DataFrame(orders)

# Check if same size:
assert len(ohlc) == len(lsch)

# Combine the two dataframes:
bt_df = pd.concat([ohlc, lsch], axis=1)

# Capital curve:
starting_capital = 10_000
percent_risked = 0.1
order_size = starting_capital*percent_risked
capital_curve = [starting_capital]

# Closes the very last positon if left open:
if bt_df[0][len(bt_df)-1] not in ['Close']:
    bt_df[0][len(bt_df)-1] = 'Close'
    

# Backtester:
for i in range(len(bt_df)):
    if bt_df[0][i] == 'Long':
        entry = bt_df['Close'][i]                   # Entry price
        
        for o in range(i, len(bt_df)):
            if bt_df[0][o] == 'Close':
                _exit = bt_df['Close'][o]           # Store _exit price
                p_l = (_exit - entry)*order_size   # profit/loss
                starting_capital += p_l
                capital_curve.append(starting_capital)
                break

    elif bt_df[0][i] == 'Short':
        entry = bt_df['Close'][i]                   # store entry price
        
        for o in range(i, len(bt_df)):               # Look for Close
            if bt_df[0][o] == 'Close':
                _exit = bt_df['Close'][o] 
                p_l = (entry - _exit)*order_size    # Profit/Perte
                starting_capital += p_l
                capital_curve.append(starting_capital)
                break
        
# Visualize final Capital Curve:
x = range(0,len(capital_curve))
fig, ax = plt.subplots()
ax.plot(x, capital_curve)
ax.set_xlabel('Transactions')
ax.set_ylabel('Capital')
ax.set_title("Capital evolution after each transaction")
plt.show()

'''
###################
Performance results
###################
'''
# Strategy performance:
print("- ROI of strategy (%):",
      ((capital_curve[len(capital_curve)-1] / capital_curve[0])-1)*100)


# Buy & Hold performance:
BH_profit = (bt_df['Close'][len(bt_df['Close'])-1] -
             bt_df['Close'][0])*10000*percent_risked
print("- ROI of Buy & Hold (%):",(((BH_profit+10000)/10000)-1)*100)


'''
##########
Validation
##########
'''
# Making predictions on unseen data (i.e. validation set):
last_action = 4   # cash
pred_orders = []

for i in range(9595, len(df)):
    state = np.matrix(env.current_state(i))
    pred = np.array(model(state))
    
    # reduce the set to available actions via set difference (complement) and
    #make it list so it is iterable:
    for m in list({0,1,2,3,4} - set(env.action_space(last_action))): 
        pred[0][m] = 0
    
    for _ in range(5):
            if pred[0][_] == max(pred[0]):
                pred_action = _
                
    pred_orders.append(pred_action)
    last_action = pred_action
    
for i in range(len(pred_orders)):
    if pred_orders[i] == 0:
        pred_orders[i] = "Long"
    
    elif pred_orders[i] == 1:
        pred_orders[i] = "Short"
        
    elif pred_orders[i] == 2:
        pred_orders[i] = "Hold"
    
    elif pred_orders[i] == 3:
        pred_orders[i] = "Close"
        
    elif pred_orders[i] == 4:
        pred_orders[i] = "Cash"