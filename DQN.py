# -*- coding: utf-8 -*-
"""
Class that instantiates the Deep Q-Networks
"""

import tensorflow as tf
import keras
from parameters import alpha

# Building the DQN architecture:
def DQN():
    # Defining layers: inputs, hidden and output
    init = tf.keras.initializers.glorot_normal() # Xavier weight initializer
    inputs = tf.keras.layers.Input(shape=(10,),dtype='float64') # input layer: 17 neurons for the 17 indicators we're using
  
    # 1st hidden layer with 10 neurons  
    hidd_layer1 = tf.keras.layers.Dense(10, activation=tf.nn.tanh,
                                        kernel_initializer = init,
                                        dtype='float64', use_bias=True )(inputs) 
    
    # 2nd hidden layer with 10 neurons
    hidd_layer2 = tf.keras.layers.Dense(10, activation=tf.nn.tanh,
                                        kernel_initializer = init,
                                        dtype='float64',
                                        use_bias=True)(hidd_layer1) 
    
    # Output layer with 5 outputs for each of the 5 possible actions:
    outputs = tf.keras.layers.Dense(5, activation = None,
                                    kernel_initializer = 'zeros',
                                    bias_initializer = 'zeros',
                                    dtype='float64',
                                    use_bias=True)(hidd_layer2) 

    # Model object groups the layers together:
    return tf.keras.Model(inputs=inputs, outputs=outputs)