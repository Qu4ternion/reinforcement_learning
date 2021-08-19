# -*- coding: utf-8 -*-
"""
Class that instantiates the Deep Q-Networks
"""

import tensorflow as tf

# Building the DQN architecture:
def DQN():
    
    # Defining layers: inputs, hidden and output:
    
    # Xavier weight initializer
    init = tf.keras.initializers.glorot_normal() 
    
    # input layer: 10 neurons for the 10 indicators we're using
    inputs = tf.keras.layers.Input(shape=(10,), dtype='float64') 
  
    # 1st hidden layer with 20 neurons  
    hidd_layer1 = tf.keras.layers.Dense(20,
                                        activation=tf.nn.tanh,
                                        kernel_initializer = init,
                                        dtype='float64',
                                        use_bias=True)(inputs) 
    
    # 3nd hidden layer with 20 neurons
    hidd_layer2 = tf.keras.layers.Dense(20,
                                        activation=tf.nn.tanh,
                                        kernel_initializer = init,
                                        dtype='float64',
                                        use_bias=True)(hidd_layer1) 
    
    # Output layer with 5 outputs for each of the 5 possible actions:
    outputs = tf.keras.layers.Dense(5,
                                    activation = tf.nn.softmax,
                                    kernel_initializer = init,
                                    bias_initializer = init,
                                    dtype='float64',
                                    use_bias=True)(hidd_layer2) 

    # Model object groups the layers together:
    return tf.keras.Model(inputs=inputs, outputs=outputs)