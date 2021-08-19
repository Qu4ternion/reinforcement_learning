# -*- coding: utf-8 -*-

"""
@author: Reda Gossony

Main function for the Algorithmic trading bot.
"""
import keras
import math
import tensorflow             as tf
import numpy                  as np
import matplotlib.pyplot      as plt
from   keras import backend   as K

from   Environment            import Environment
from   DQN                    import DQN
from   load_data              import load_data
from   _preprocessing         import check_na, Assert, Drop_Adj, akima, Normalize
from   metrics                import (nth_percentile, close_delta, VWAP,
                                      Bollinger, DoD)
from   parameters             import (threshold, epsilon, min_epsilon, alpha,
                                      gamma, decay, data_path, _current_equity,
                                      timesteps, epoch_limit)
#tf.enable_eager_execution()

def main():

    # Import data:
    df = load_data(data_path)
    
    # Data cleaning:
    print('Cleaning data...')
    try:
        missing = check_na(df)  # check for missing observations
        Assert(df)              # sanity check
        df = Drop_Adj(df)       # Drop "Adjusted close" column
        df = akima(df)          # Interpolate missing values using Akima Splines
        Normalize(df)           # Normalize "Volume" column (high values)
     
    except:
        return df
    
    # Add input features:
    print('Creating input features...')
    nth_percentile(df)          # n-th percentile price variation
    close_delta(df)             # day-to-day price difference
    VWAP(df)                    # Volume Waighted Average Price
    Bollinger(df)               # Relative Bollinger Bands
    DoD(df)                     # Daily Rate of Return
    
    # Drop the absolute bands and only leave the relative bands:
    df = df.drop(['bb_ma', 'bb_high', 'bb_low','Date'], axis = 1)    
    
    # Instantiate "Environment" and feed it our dataframe:
    global env
    env = Environment(df)
    
    # Instantiate Double Deep Q-Neural Networks (DQN + Target DQN):
    global model
    model, target_dqn = DQN(), DQN()
    
    # Instantiate Loss function:
    loss_function = keras.losses.Huber()
    
    # Instantiate Optimizer:
    optimizer = tf.keras.optimizers.SGD(learning_rate = alpha)
                 # keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    
    # Initialize storage for states, actions & rewards here:
    cumulative_reward = 0
    curr_equity = []
    epoch = 0
    K.clear_session()
    
    # Run loop until it's solved solved (as many epochs as necessary):
    while cumulative_reward < threshold  and  epoch < epoch_limit:
        last_action = 4  # initial state: being in cash
        cumulative_reward = 0
        current_equity = _current_equity
        states_history = []
        actions_history = []
        rewards_history = []
        Losses = []
        orders = []
        overtime_rewards = []
        cum_reward = []
        cum_buy_q = []
        
        # increment epoch
        epoch += 1
        
        # iterate over all time-steps (i.e. states)
        for state_index in range(0, len(df)-1): 
            
            # get current state from the Environment:
            state = np.matrix(env.current_state(state_index, df))
            
            # epsilon-greedy: take an action given the state
            global epsilon
            # decay epsilon to the limit of 'min_epsilon'
            epsilon = max(epsilon, min_epsilon) 
            
            if epsilon >= np.random.rand(1)[0]:
                # pick a random action from action space depending on our last
                # action taken.
                action = np.random.choice(env.action_space(last_action))  
                
            # else take Max Q-value action (best action):
            else:
                # A tensor with the same data as input, with an additional
                # dimension inserted at the index (position) specified by "axis"
                state_tensor = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
                #state_tensor = tf.expand_dims(state_tensor, axis = 0)
                
                # input state into DQN to make a Q value prediction
                action_probs = model(state_tensor[0], training=False) 
               
                # Take best action (highest Q-value):
                # loop to find action that maximizes the estimated Q value
                for i in env.action_space(last_action):
                    reduced_space =[np.array(action_probs)[0][i] \
                                    for i in env.action_space(last_action)]
                
                # Make an array that has only the available actions given last
                # action
                # Since it's a reduced space, the indices of this list do not
                # represent our original actions
                # So we write these 2 loops to find the original index which
                # gives us the max action
                for i in range(len(reduced_space)): 
                    if reduced_space[i] == max(reduced_space):
                        Max = reduced_space[i]
                
                for i in range(5):
                    if np.array(action_probs)[0][i] == Max:
                        action = i
                     
            # Log orders:
            orders.append(action)
            env.orders = orders
            
            # Portfolio reward:
            action_reward = env.reward(action, state_index)
           #action_reward = env.reward2(action, state_index)
            
            # Next state given current state index:
            next_state = np.array(env.next_state(state_index, df))
         
            # Store states, actions and rewards:
            states_history.append(state)
            actions_history.append(action)
            rewards_history.append(action_reward)
            
            '''
            Single DQN framework:
                
            # Estimated Q-value (future rewards) of next state s' [second pass]:
            future_rewards = model.predict(tf.expand_dims(next_state, axis=0))
            
            # Subset so we can only take the max over the available actions
            instead of the whole action space:
            reduced_rewards = []
            Reduced_rewards = [reduced_rewards.append(future_rewards[0][r])
                               for r in Environment.action_space(action)]
            
            future_rewards = max(reduced_rewards)
            '''
    
            # Double DQN framework:
            future_rewards = target_dqn(tf.expand_dims(next_state, axis=0))
            
            # Updating target Q-value: Q(s,a) = reward + gamma * max Q(s',a')
            target_q_value = action_reward + gamma * future_rewards
           #target_q_value = action_reward + gamma * tf.reduce_max(future_rewards, axis=1)
           
           # "tf.reduce_max" computes the maximum of elements across dimensions
           # of a tensor.
            
            
            # Create one-hot encoding filter with 5 columns (one for each action)
            # each row of 0s and 1s representing which action was taken
            # in each state.
            Filter = tf.one_hot(actions_history, depth=5, dtype='float64')
            
            # Loss:
            with tf.GradientTape() as tape:
                # NN-predicted Q-value (that we want to get to the target Q value)
                pred_q_values = model(states_history)
        
                # Multiplying the predicted Q-values to the mask will give us
                # a matrix that has the Q-value for each action taken
                q_action = tf.reduce_sum(tf.multiply(pred_q_values, Filter))
                
                # Calculate loss between target Q-value and predicted Q-value
                loss = loss_function(target_q_value, q_action)  
                
            # Stop training if Loss diverges to infinity:
            if np.array(loss) == math.inf:
                    print('Loss diverged on iteration:', state_index,
                          'on epoch:', epoch)
                    break
            
            # Backpropagation:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # Update the weights:
            weights = model.get_weights()
            model.set_weights(weights)
            
            # Update the Target Network every 10 iterations:
            if state_index % timesteps == 0:
                target_dqn.set_weights(model.get_weights())
            
            # Update last action taken
            last_action = action
            env.last_action = action
            
            # Decay epsilon:
            epsilon = epsilon*decay
            
            # Update running reward:
            cumulative_reward += action_reward
            
            # Empty history lists
            states_history = []
            actions_history = []
            
            # Log loss:
            Losses.append(loss)
            
            # Log rewards over-time:
            overtime_rewards.append(action_reward)
            
            # Log cumulative rewards:
            cum_reward.append(cumulative_reward)
            
            # Log Q-value for Buying:
            if action == 0:
                cum_buy_q.append(target_q_value)
            else:
                pass
            
            # Log current equity:
            curr_equity.append(current_equity)
            
            # Show iteration, loss and exploration rate:
            print('--------------------------------------')
            print(f'- Epoch: {epoch}')
            print(f'- Iteration number: {state_index}')
            print(f'- Loss is equal to: {np.array(loss)}')
            print(f'- Exploration probability is: {round(epsilon, 3)}')
            print(f'- Cumulative reward is: {round(cum_reward[-1], 2)}')

        x = range(0,len(cum_reward))
        fig, ax = plt.subplots()
        ax.plot(x, cum_reward)
        plt.title(f'Cumulative rewards for Epoch: {epoch}')
        plt.show()
    
    # Mean of Errors:
    np.mean(Losses)
    
    # Plotting the losses:
    x = range(0,len(Losses))
    fig, ax = plt.subplots()
    ax.plot(x, Losses)
    plt.title('Neural network Loss')
    plt.show()
    
    # Plot rewards overtime:
    x = range(0,len(overtime_rewards))
    fig, ax = plt.subplots()
    ax.plot(x, overtime_rewards)
    plt.title('Reward variation overtime')
    plt.xlabel('Days')
    plt.ylabel('Rewards')
    plt.show()
    
    # Plot cumulative reward:
    x = range(0,len(cum_reward))
    fig, ax = plt.subplots()
    ax.plot(x, cum_reward)
    plt.title('Rewards collected overtime')
    plt.xlabel('Days')
    plt.ylabel('Rewards')
    plt.show()
    
# =============================================================================
#     # Plot cumulative Buy Q Value:
#     q = []
#     for i in range(2,orders.count(0)):
#         q_ = cum_buy_q[i][0]
#         q.append(q_)
#     q = np.array(q)
#     
#     x = range(len(q))
#     fig, ax = plt.subplots()
#     ax.plot(x, q)
#     plt.show()
# =============================================================================

if __name__ == '__main__':
    main()