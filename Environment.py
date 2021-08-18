# -*- coding: utf-8 -*-
"""
Environment class that represents the financial market environment.
"""
from parameters import starting_equity, position_size, last_action

# Class modelling the financial environment:
class Environment:
    
    def __init__(self, df, orders = [last_action, last_action],
                 equity = starting_equity, position_size = position_size):

        self.equity = equity
        self.position_size = position_size  # 1% of account
        self.df = df
        self.orders = orders
        self.last_action = last_action
        
    # Return feasible actions given the last action taken:
    @staticmethod
    def action_space(action) -> list:  
        '''

        Parameters
        ----------
        action : int
            Long (0) ; Short (1) ; Hold (2) ; Close (3) ; Cash (4).

        Returns
        -------
        List
            Will return list of feasible actions given the prior one taken.

        '''
        return [2,3] if action <= 2 else [0,1,4]
    
    # Return current state of the Environment:
    @staticmethod
    def current_state(current_state, df) -> list:
        return list(df.iloc[current_state, 0:len(df.columns)-1])
                                # we subset so we exclude the 
                                # last 'returns' column

    # Return next state given current state:
    @staticmethod
    def next_state(current_state, df) -> list:
        return list(df.iloc[current_state + 1, 0:len(df.columns)-1])
    
    # Get daily return of a given a state:
    @staticmethod
    def get_return(state_index, df) -> int:
        return df['DoD return'][state_index+1]
    
    # Get reward of an action:
    def reward(self, action, state_index) -> int : 
        '''
        Parameters
        ----------
        action : int
            represents the action taken by the Agent.
        state_index : int
            index of the current time-step.

        Returns
        -------
        int
            Reward received by the Agent (can be positive or negative).
        '''
        # cash
        if action == 4: 
            return -0.1  # negative reward (i.e. penalty) for staying in cash
        
        # long
        elif action == 0: 
            return self.get_return(state_index, self.df) # reward / penalty
  
        # short
        elif action == 1: 
            if self.get_return(state_index, self.df) < 0:
                return self.get_return(state_index, self.df)  # reward
            else:
                return -self.get_return(state_index, self.df) # penalty
        
        # hold: we first need to know what type of hold? buy or sell
        elif action == 2:
            if self.last_action == 0: # hold-buy
                return self.get_return(state_index, self.df) # reward / penalty
                    
            elif self.last_action == 1: # hold sell
                if self.get_return(state_index, self.df) < 0:
                    return self.get_return(state_index, self.df) # positive reward
                else:
                    return -self.get_return(state_index, self.df) # penalty
                
            elif self.last_action == 2: # Holding prior state and still holding now
                for _ in range(len(self.orders)-1, -1, -1): # Holding buy or sell?
                
                    if self.orders[_] == 0: # Holding a buy position:
                        return self.get_return(state_index, self.df) # reward / penalty
                        break

                    elif self.orders[_] == 1: # Holding sell position:
                        if self.get_return(state_index, self.df) < 0:
                            return self.get_return(state_index, self.df) #reward
    
                        else:
                            return -self.get_return(state_index, self.df) #penalty
                        break

            
        # close: we need to know if we're closing buy or sell position
        elif action == 3:
            if self.orders[len(self.orders)-2] == 0: # closing buy position
                if self.get_return(state_index, self.df) > 0:
                    return -self.get_return(state_index, self.df)
                    # negative reward if closed buy while next day was going
                    # to be profitable (i.e. missed opportunity)
    
                else:
                    return self.get_return(state_index, self.df) # reward
            
            elif self.orders[len(self.orders)-2] == 1: # closing sell position
                if self.get_return(state_index, self.df) < 0:
                    return -self.get_return(state_index, self.df)
                # negative reward (missed opportunity)
    
                else:
                    return self.get_return(state_index, self.df)
            
            elif self.orders[len(self.orders)-2] == 2: # Hold: we gotta know
                                                       # what type of hold:
                
                for _ in range(len(self.orders)-1, -1, -1):
                    if self.orders[_] == 0:            # Closing a buy-hold
                        if self.get_return(state_index, self.df) > 0:
                            return -self.get_return(state_index, self.df) # negative reward if closed while next day

                        else:
                            return self.get_return(state_index, self.df)
                        break
                    
                    elif self.orders[_] == 1:          # Closing a short-hold
                        if self.get_return(state_index, self.df) < 0:
                            return -self.get_return(state_index, self.df) # negative reward if closed while next day
    
                        else:
                            return self.get_return(state_index, self.df) 
                        break

'''
##############################################################################
# Some Variations of the Reward Function: portfolio value, sparse reward, etc.
##############################################################################

    # Function that takes in the action, state index and current equity and outputs the current portfolio value                
    @staticmethod
    def reward2(action, state_id):
        
        global current_equity, old_equity
        
        if action in [0,1]: # long/short won't make a difference immediately after entry so equity stays the same
            return 0
        
        elif action in [2,3]: # hold, close
            for _ in range(state_id,-1, -1): # finding the entry price
                
                if orders[_] == 0:  # hold buy / buy entry
                    entry_price = dt['Close'][_] # store entry price
                    current_price = dt['Close'][state_id]
                    p_l = (current_price - entry_price)*current_equity*fraction_invested
                    old_equity = current_equity
                    current_equity += p_l
                    return (current_equity - old_equity)/old_equity
                
                    break
                
                elif orders[_] == 1:  # hold sell / sell entry
                    entry_price = dt['Close'][_] # store entry price
                    current_price = dt['Close'][state_id]
                    p_l = (entry_price - current_price)*current_equity*fraction_invested
                    old_equity = current_equity
                    current_equity += p_l
                    return (current_equity - old_equity)/old_equity
                
                    break
                
        # cash
        elif action == 4: 
            return -0.5 # equity stays unchanged as we stay in cash
        
    
    @staticmethod
    def sparse(action, state_id):
            
            global current_equity, old_equity
            
            if action in [0,1,2]:
                return 1
            
            elif action == 4:
                return 0
            
            elif action == 3:
                for _ in range(state_id,-1, -1): # finding the entry price
                    
                    if orders[_] == 0:  # hold buy / buy entry
                        entry_price = dt['Close'][_] # store entry price
                        current_price = dt['Close'][state_id]
                        p_l = (current_price - entry_price)*current_equity*fraction_invested
                        old_equity = current_equity
                        current_equity += p_l
                        return (current_equity - old_equity)/old_equity
                    
                        break
                    
                    elif orders[_] == 1:  # hold sell / sell entry
                        entry_price = dt['Close'][_] # store entry price
                        current_price = dt['Close'][state_id]
                        p_l = (entry_price - current_price)*current_equity*fraction_invested
                        old_equity = current_equity
                        current_equity += p_l
                        return (current_equity - old_equity)/old_equity
                    
                        break
            
            
                    
    # The P&L reward function:
    @staticmethod
    def reward3(action, state_id):
        global current_equity
        if action in [0,1]: # long/short won't make a difference immediately after entry so equity stays the same
            return 0
        
        elif action in [2,3]: # hold, close
            for _ in range(state_id, -1, -1): # finding the entry price
                
                if orders[_] == 0:  # hold buy / buy entry
                    entry_price = dt['Close'][_] # store entry price
                    current_price = dt['Close'][state_id]
                    p_l = (current_price - entry_price)
                    
                    return p_l
                
                    break
                
                elif orders[_] == 1:  # hold sell / sell entry
                    entry_price = dt['Close'][_] # store entry price
                    current_price = dt['Close'][state_id]
                    p_l = (entry_price - current_price)
                    
                    return p_l
                
                    break
        
        elif action == 4: # cash
            return 0
'''