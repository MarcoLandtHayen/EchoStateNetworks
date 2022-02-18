#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:03:04 2022

@author: mlandt-hayen
"""

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow.keras.initializers as tfi
from tensorflow.keras.utils import plot_model



### Set up custom layer
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, n_res, W_in_lim=1, leak_rate=0.5, verbose=0):
        super(CustomLayer, self).__init__()
        
        # Setup reservoir units
        self.n_res = n_res
        self.W_in_lim = W_in_lim
        self.leak_rate = leak_rate
        self.res_units_init = tf.keras.layers.Dense(units=n_res, 
                                                    activation=None, 
                                                    use_bias=True,
                                                    kernel_initializer=tfi.RandomUniform(minval=-W_in_lim, maxval=W_in_lim, seed=None),
                                                    bias_initializer=tfi.RandomUniform(minval=-W_in_lim, maxval=W_in_lim, seed=None))
        self.res_units = tf.keras.layers.Dense(units=n_res, 
                                               activation=None, 
                                               use_bias=True,
                                               kernel_initializer=tfi.RandomUniform(minval=-W_in_lim, maxval=W_in_lim, seed=None),
                                               bias_initializer=tfi.RandomUniform(minval=-W_in_lim, maxval=W_in_lim, seed=None))
        self.verbose= verbose
    
    
    
    
    def call(self, inputs):
    
        # Connect inputs to reservoir units to get initial reservoir state (t=1), called x_prev, since
        # it will be used as "previous" state when calculating further reservoir states.
        x_prev = self.leak_rate * tf.tanh(self.res_units_init(inputs[:,0:1,:])) # x(1) = leak_rate * tanh(W_in * u(1))
        print("\ninitial x_prev: ", x_prev)
        
        # Initialize storage X for all reservoir states (samples, timesteps, n_res):
        # Store x_prev as x_1 in X
        X = x_prev
        
        # Now loop over remaining time steps t = 2..T
        T = inputs.shape[1]
        print("\nT: ", T)
        
        for t in range(1,T):
            x_t = (1 - self.leak_rate) * x_prev + self.leak_rate * tf.tanh(self.res_units_init(inputs[:,t:t+1,:]) + self.res_units(x_prev))
    
            # Store x_t in X
            X = tf.concat([X, x_t], axis=1)
            
            # x_t becomes x_prev for next timestep
            x_prev = x_t
        
        print("\nX shape: ", X.shape)
        return X
        
            
        
        
    
    ## Use X to store all n_res final reservoir states after T input steps
    ## for all n_samples train inputs, use specified activation function.
    
    # initialize X (n_res x n_samples)
    X = np.zeros((n_res,n_samples))
    
    # Loop over n_samples    
    for i in range(n_samples):
        
        # Loop over timesteps in current sample
        for j in range(T):
            
            # If desired activation function is 'tanh':
            if activation=='tanh':
                # first input timestep needs special treatment, since reservoir state is not yet initialized
                if j == 0:
                    X[:,i:i+1] = np.tanh(W_in * train_input[i,j])
                elif j > 0:
                    X[:,i:i+1] = np.tanh(W_in * train_input[i,j] + np.reshape(np.matmul(W_res, X[:,i:i+1]),(n_res,1)))
            # If desired activation function is 'sigmoid'
            elif activation=='sigmoid':
                # first input timestep needs special treatment, since reservoir state is not yet initialized
                if j == 0:
                    X[:,i:i+1] = expit(W_in * train_input[i,j])
                elif j > 0:
                    X[:,i:i+1] = expit(W_in * train_input[i,j] + np.reshape(np.matmul(W_res, X[:,i:i+1]),(n_res,1)))
            
            
            
            


### Set up model WITH custom layer

# Set (hyper-)parameters:
n_res = 20 # Number of reservoir units
W_in_lim = 1 # Initialize input weights from random uniform distribution in [- W_in_lim, + W_in_lim]
leak_rate = 0.5 # Leak rate for transition of reservoir states
spec_radius = 1.2 # Spectral radius, becomes largest Eigenvalue of reservoir weight matrix
sparsity = 0.2 # Sparsity of reservoir weight matrix

# Input layer
model_inputs = Input(shape=(10,2)) # (timesteps, features)
print(model_inputs)

# Use custom layer
custom_layer = CustomLayer(n_res=n_res, W_in_lim=W_in_lim, leak_rate=leak_rate)(model_inputs) # 20 reservoir units
print(custom_layer)


# Output unit
output = Dense(1, name='output')(custom_layer)
print(output)

# Define model
model = Model(model_inputs, output, name='model_2')
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)


## Modify reservoir weights W_res using spectral radius:

# Get model weights for ALL layers
model_weights = np.array(model.get_weights())
print("\model_weights shape: ", model_weights.shape)

# Extract reservoir weights
W_res = model_weights[2]
print("\nW_res: ", W_res)

# Check shape, expect (n_res, n_res)
print("\nW_res shape: ", W_res.shape)

# Need temporary matrix W_temp to implement sparsity manually
W_temp = np.random.uniform(low=0, high=1, size=(n_res,n_res))
W_sparse = W_temp <= sparsity

# Now apply sparsity to initial W_res
W = W_sparse * W_res

# Get largest Eigenvalue of W
ev_max = np.max(np.real(np.linalg.eigvals(W)))

# Finally set up W_res
W_res = spec_radius * W / ev_max

# Integrate modified reservoir weights into model weights
model_weights[2] = W_res
model.set_weights(model_weights)







# Print input weights and biases, connecting model_inputs to reservoir_units
print("\nmodel layers: ", model.layers)
print("\n\nW_in weights: ", model.layers[1].weights[0])
print("\nW_in biases: ", model.layers[1].weights[1])
print("\n\nW_res weights: ", model.layers[1].weights[2])
print("\nW_res biases: ", model.layers[1].weights[3])

## Check distribution of input and reservoir weights after initialization

# Flatten input and reservoir weights
W_in = np.reshape(model.layers[1].weights[0], (model.layers[1].weights[0].shape[0] * model.layers[1].weights[0].shape[1], 1))
W_res = np.reshape(model.layers[1].weights[2], (model.layers[1].weights[2].shape[0] * model.layers[1].weights[2].shape[1], 1))

# Plot histogram for input weights
nBins = 100
fig, axes = plt.subplots(1, 1, figsize=(10,5))
axes.hist(W_in[:,0], nBins, color="red")
axes.set_xlabel("weight value")
axes.set_ylabel("counts")
axes.set_title("Histogram of initial input weights W_in")
plt.show()

# Plot histogram for input weights
nBins = 100
fig, axes = plt.subplots(1, 1, figsize=(10,5))
axes.hist(W_res[:,0], nBins, color="blue")
axes.set_xlabel("weight value")
axes.set_ylabel("counts")
axes.set_title("Histogram of initial reservoir weights W_res")
plt.show()




#opt = Adam(lr=0.01, clipnorm=1)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
model.summary()

# plot model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)




# encoder LSTM
# 'elu' activation means exponential linear unit
n_hidden = 16

encoder_lstm = LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2, return_sequences=True, return_state=True)
encoder_all_h, encoder_last_h, encoder_last_c = encoder_lstm(encoder_inputs)
print(encoder_all_h)
print(encoder_last_h)
print(encoder_last_c)

# add batch normalization
encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

# initial states of the encoder
states = [encoder_last_h, encoder_last_c]
print(states)

### Set up the temporal attention layer
attention=TemporalAttention(n_hidden, verbose=0)
#attention=TemporalAttentionPaper(n_hidden, verbose=0)

# Set up the decoder layers
decoder_inputs = Input(shape=(decoder_train_input.shape[1], 1), name='decoder_inputs')
decoder_lstm = LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2, return_state=True, return_sequences=False, name='decoder_lstm')
decoder_dense = Dense(1, name='decoder_dense')
print(decoder_inputs)

### pay "initial" attention
### create the context vector by applying attention to 
### last encoder hidden state + encoder_outputs (all encoder hidden states)
context_vector, attention_weights=attention(encoder_last_h, encoder_all_h)
#context_vector, attention_weights=attention(encoder_last_h, encoder_last_c, encoder_all_h)
context_vector = tf.expand_dims(context_vector, 1)
print(context_vector)

# initial decoder input
inputs = tf.concat([context_vector, decoder_inputs[:,0:1,0:1]], axis=-1)
print(inputs)

# decoder will only process one timestep at a time.
# have (T-1) targets to process. The first step is already used for initial decoder input.
# so have only (T-2) remaining steps.
for i in range(decoder_train_input.shape[1] - 1):    
    
    # Run the decoder on one timestep
    decoder_outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
    
    ### apply attention on decoder hidden state and encoder hidden states
    context_vector, attention_weights=attention(state_h, encoder_all_h)    
    #context_vector, attention_weights=attention(state_h, state_c, encoder_all_h)    
    context_vector = tf.expand_dims(context_vector, 1) 
    #print(context_vector)    

    # create next decoder input and update states
    inputs = tf.concat([context_vector, decoder_inputs[:,(i+1):(i+2),0:1]], axis=-1)    
    states = [state_h, state_c]
    #print(inputs)

decoder_output = decoder_dense(state_h)
print(decoder_output)


# Define and compile model 
model = Model([encoder_inputs, decoder_inputs], decoder_output, name='model_encoder_decoder')

#opt = Adam(lr=0.01, clipnorm=1)
opt = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
model.summary()

# plot model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)





### set up custom class for temporal attention, according to tutorial
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units, verbose=0):
        super(TemporalAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.verbose= verbose
    
    def call(self, query, values):
        if self.verbose:
            print('\n******* Bahdanau Attention STARTS******')
            print('query (decoder hidden state): (batch_size, hidden size) ', query.shape)
            print('values (encoder all hidden state): (batch_size, max_len, hidden size) ', values.shape)
            
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)
        
        if self.verbose:
            print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)
            
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
        if self.verbose:
            print('score: (batch_size, max_length, 1) ',score.shape)
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        if self.verbose:
            print('attention_weights: (batch_size, max_length, 1) ',attention_weights.shape)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        if self.verbose:
            print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ',context_vector.shape)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        if self.verbose:
            print('context_vector after reduce_sum: (batch_size, hidden_size) ',context_vector.shape)
            print('\n******* Bahdanau Attention ENDS******')
        return context_vector, attention_weights
    
    

### set up custom class for temporal attention, according to paper on Dual-Stage attention
class TemporalAttentionPaper(tf.keras.layers.Layer):
    def __init__(self, units, verbose=0):
        super(TemporalAttentionPaper, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.verbose= verbose
    
    def call(self, query, cellstate, values):
        if self.verbose:
            print('\n******* Bahdanau Attention STARTS******')
            print('query (decoder hidden state): (batch_size, hidden size) ', query.shape)
            print('values (encoder all hidden state): (batch_size, max_len, hidden size) ', values.shape)
            
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)
        cellstate_with_time_axis = tf.expand_dims(cellstate, 1)
        
        if self.verbose:
            print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)
            
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
        self.W1(tf.concat([query_with_time_axis, cellstate_with_time_axis], axis=-1)) + self.W2(values)))
        if self.verbose:
            print('score: (batch_size, max_length, 1) ',score.shape)
            print('concat query and cellstate: (batch_size, 1, 2 x hidden size) ', tf.concat([query_with_time_axis, cellstate_with_time_axis], axis=-1).shape)
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        if self.verbose:
            print('attention_weights: (batch_size, max_length, 1) ',attention_weights.shape)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        if self.verbose:
            print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ',context_vector.shape)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        if self.verbose:
            print('context_vector after reduce_sum: (batch_size, hidden_size) ',context_vector.shape)
            print('\n******* Bahdanau Attention ENDS******')
        return context_vector, attention_weights