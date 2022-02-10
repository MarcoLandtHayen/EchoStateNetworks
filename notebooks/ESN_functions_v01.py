#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content: Useful functions for EchoStateNetworks
Version: 01 (2022-02-09)

@author: mlandt-hayen
"""

import numpy as np
from scipy.special import expit  # vectorized logistig function
import matplotlib.pyplot as plt

### Define function to split data:

## Function input: 
# Feed timeseries with absolute values.

## Parameters:
# input_length (int): timesteps per input sample
# target_length (int): number of steps ahead to predict
# time_lag (int): Necessary when creating inputs for multi-reservoir ESN, having timelag 0,1,2,... days between
#                 input and target.
# train_val_split ([0,1]): rel. amount of input samples used for training
# val_samples_from ('end', 'start_end'): Determines where to take validation samples from. Usually take
#                                        samples from series 'end' as default, but can also take equal
#                                        number of samples from beginning AND end ('start_end').
# abs_to_rel_YN (True/False): if True, convert absolute values to relative change values
# scaled_YN (True/False):if True, scale inputs and targets to [0,1] applying min/max scaling 
#                        according to min/max obtained from only train inputs
# verbose (True/False)): if True, print shapes of function input and output

## Function output:
# Returns arrays (samples, timesteps, timelag) train_input and val_input as abs. or rel. changes.
# Returns series (samples) train_target and val_target as abs. or rel. changes.
# Returns train_min and train_max, used for scaling, set to ZERO, if scaled_YN=False.

def split_data(data_abs, input_length, target_length, time_lag=0, 
               train_val_split=0.8, val_samples_from='end', abs_to_rel_YN=True, binary_YN=False, scaled_YN=False,
               verbose=True):
    
    # Optionally convert input data from absolute values to rel. change values.
    if abs_to_rel_YN:
        data = (data_abs[1:] - data_abs[:-1]) / data_abs[:-1]
    else:
        data = data_abs
     
    # Split data according to desired input_length, save series in list X.
    # End up with (len(data) - input_length + 1) samples.
    X = list()
    for i in range(len(data) - input_length + 1):
        sample = data[i:i+input_length]
        X.append(sample)
    
    # Convert X to np.array.
    X = np.array(X)
    
    # Cut the last target_length samples to make sure, we have a target for each input sample.
    X = X[:-target_length]
        
    # Create targets, optionally as rel. change values from abs. values. 
    # Take desired target_length into account. Store in Y.
    if abs_to_rel_YN:
        Y = (data_abs[target_length:] - data_abs[:-target_length]) / data_abs[:-target_length]
    else:
        Y = data_abs[target_length:]
    
    # Cut the first input_length targets, since we don't have input samples for these targets.
    # And cut another time_lag targets upfront, to have identical dimensions for each time_lag slice later on.
    # Note: If working with absolute targets, leave one more target, to have correct dimensions.
    if abs_to_rel_YN:
        Y = Y[input_length + time_lag:]
    else:
        Y = Y[input_length + time_lag - 1:]
    
    # Now take care of desired time_lag.
    # lag=0 means to keep original input samples, but still need to reshape to have lag as third dimension.
    # And leave out the first time_lag input samples to end up with the same dimensions for all lags.    
    X_out = np.reshape(X[time_lag:], (X.shape[0] - time_lag, X.shape[1], 1))
    
    # lag>0 (loop over j) means to leave out the first (time_lag - j) input series and last j input series, 
    # but keep the target fixed! And reshape to have lag as third dimension.
    # Note: Loop starts with j=0, hence need j+1
    for j in range(time_lag):
        X_temp = np.reshape(X[time_lag-(j+1):-(j+1)], (X.shape[0] - time_lag, X.shape[1], 1))
        X_out = np.concatenate((X_out, X_temp), axis=2)
    
    # re-assign obtained X_out to X
    X = X_out    
    
    # Split inputs and targets into train and validation data according to train_val_split and
    # desired val_samples_from attribute.
    
    # If validation samples are supposed to be from 'end' of series:
    if val_samples_from == 'end':
        n_train = int(len(X) * train_val_split)
    
        train_input = X[:n_train]
        val_input = X[n_train:]
        train_target = Y[:n_train]
        val_target = Y[n_train:]
    
    # If validation samples are supposed to be from 'beginning' and 'end':
    else:
        n_train = int(len(X) * (1-train_val_split)/2)
        
        train_input = X[n_train:-n_train]
        train_target = Y[n_train:-n_train]
        val_input = np.concatenate((X[:n_train], X[-n_train:]))
        val_target = np.concatenate((Y[:n_train], Y[-n_train:]))
  
    # Optionally scale train and validation inputs and targets to [0,1] 
    # according to min/max obtained from train inputs.
    # Initialize train_min and _max to zero, to have return values even if no scaling is desired.
    train_min = 0
    train_max = 0
    if scaled_YN:
        train_min = np.min(train_input)
        train_max = np.max(train_input)

        # scale input values and targets, according to min/min of ONLY train inputs:
        # substract min and divide by (max - min)
        train_input = (train_input - train_min) / (train_max - train_min)
        val_input = (val_input - train_min) / (train_max - train_min)
        train_target = (train_target - train_min) / (train_max - train_min)
        val_target = (val_target - train_min) / (train_max - train_min)

    # Optionally print dimensions
    if verbose:
        print("raw data shape: ", data.shape)
        print("train_input shape: ", train_input.shape)
        print("val_input shape: ", val_input.shape)
        print("train_target shape: ", train_target.shape)
        print("val_target shape: ", val_target.shape)
        print("train_min: ", train_min)
        print("train_max: ", train_max)
        
    #return values
    return train_input, val_input, train_target, val_target, train_min, train_max    


### Define function to train basic ESN:

## Function input: 
# Feed timeseries with relative change values: train_input and train_target with
# dimensions (num samples, num timesteps) and (num samples), respectively.

## Parameters:
# n_res (int): number of reservoir units
# sparsity ([0,1]): Sparsity of connections in reservoir
# spectral_rad (real>0): Spectral radius for initializing reservoir weights.
# w_in_lim (real>0): Parameter for initializing input weights.
# activation ('tanh', 'sigmoid'): Specify function for transition of reservoir states.
# verbose (True/False)): if True, print shapes of function input and output

## Function output:
# Returns array containing input weights W_in (n_res, 1).
# Returns array containing reservoir weights W_res (n_res, n_res).
# Returns array containing trained output weights W_out (1, n_res).

def trainESN(train_input, train_target, n_res, sparsity=0.2, spectral_rad=1.2, w_in_lim=1.0, 
             activation='tanh', verbose=True):
    
    # Get number of samples (n_samples) and input length (T) from train_input
    n_samples = train_input.shape[0]
    T = train_input.shape[1]
        
    ## initialize W_in from uniform distribution in [-w_in_lim, w_in_lim]
    W_in = np.random.uniform(low=-w_in_lim, high=w_in_lim, size=(n_res,1))
  
    ## initialize W_res

    # Need temporary matrix W_temp to implement sparsity manually
    W_temp = np.random.uniform(low=0, high=1, size=(n_res,n_res))
    W_sparse = W_temp <= sparsity

    # Then initialize W_full from uniform distribution in [-1,1]
    W_full = np.random.uniform(low=-1.0, high=1.0, size=(n_res,n_res))

    # Now apply sparsity to W_full
    W = W_sparse * W_full

    # get largest Eigenvalue of W
    ev_max = np.max(np.real(np.linalg.eigvals(W)))

    # finally set up W_res
    W_res = spectral_rad * W / ev_max

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
            
    ## Get output weights W_out from closed form solution

    # First need pseudo-inverse of X, since X is usually not a square matrix
    X_inv = np.linalg.pinv(X)

    # then need to reshape train_target
    train_target = np.reshape(train_target, (1,n_samples))

    # then get output weights
    W_out = np.matmul(train_target, X_inv)

    # Optionally print dimensions
    if verbose:
        print("train_input shape: ", train_input.shape)
        print("train_target shape: ", train_target.shape)        
        print("W_in shape: ", W_in.shape)
        print("W_out shape: ", W_out.shape)
        print("W_res shape: ", W_res.shape)
        print("W_res max: ", np.round(np.max(W_res),3))
        print("W_res sparsity: ", np.round(sum(sum(W_res != 0)) / (n_res**2), 3))
    
    # return values
    return W_in, W_res, W_out


### Define function to get ESN predictions and evaluation metrics on validation data:

## Function input: 
# Feed weight matrices W_in, W_res and W_out from traines ESN.
# Feed validation data with absolute or relative change values: val_input and val_target with
# dimensions (num samples, num timesteps) and (num samples), respectively.

## Parameters:
# activation ('tanh', 'sigmoid'): Specify function for transition of reservoir states. (Shoud be the same as in training!)
# abs_values_YN (True/False): Flags if we have absolute values as function input. This is important for
#                             correctly calculating accuracy. And determines the output as either absolute
#                             or relative change values.
# scaled_YN (True/False): Flags if we have scaled inputs and targets. This is important to know, since  
#                         calculating the accuracy requires un-scaled rel. change values.
# train_min / train_max: If we have scaled inputs, we need min and max values used for scaling.
# verbose (True/False)): if True, print shapes of function input and output plus evaluation metrics

## Function output:
# Returns array containing true targets val_target (num samples), as un-scaled absolute or
# rel. change values, depending on attribute abs_values_YN: If True return absolute values, else rel. changes.
# Returns array containing predictions val_pred (num samples), as un-scaled absolute or
# rel. change values, depending on attribute abs_values_YN: If True return absolute values, else rel. changes.
# Returns accuracy as amount of correctly predicted up/down movements.
# Returns mean-absolute-error for deviation of predicted values from true targets.

def predESN(W_in, W_res, W_out, val_input, val_target, activation='tanh', abs_values_YN=False, scaled_YN=False, 
            train_min=0.0, train_max=0.0, verbose=True):
    
    # Get number of reservoir units (n_res), number of samples (n_samples) and input length (T) from inputs
    n_res = len(W_res)
    n_samples = val_input.shape[0]
    T = val_input.shape[1]
    
    ## Use X to store all n_res final reservoir states after T input steps
    ## for all n_samples validation inputs, use specified activation function.
    
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
                    X[:,i:i+1] = np.tanh(W_in * val_input[i,j])
                elif j > 0:
                    X[:,i:i+1] = np.tanh(W_in * val_input[i,j] + np.reshape(np.matmul(W_res, X[:,i:i+1]),(n_res,1)))
            
            # If desired activation function is 'sigmoid':
            elif activation=='sigmoid':
                # first input timestep needs special treatment, since reservoir state is not yet initialized
                if j == 0:
                    X[:,i:i+1] = expit(W_in * val_input[i,j])
                elif j > 0:
                    X[:,i:i+1] = expit(W_in * val_input[i,j] + np.reshape(np.matmul(W_res, X[:,i:i+1]),(n_res,1)))
                
    # Get predicted output from trained W_out and final reservoir states for all validation inputs.
    # Note: val_pred comes with shape (1, num samples), need to get rid of first dimension.
    val_pred = np.matmul(W_out, X)[0]
    
    # Optionally re-scale target and prediction before calculating accuracy on un-scaled rel. change values.
    if scaled_YN:
        val_target = val_target * (train_max - train_min) + train_min
        val_pred = val_pred  * (train_max - train_min) + train_min
    
             
    ## Evaluate ESN prediction
    
    # Temporarily create rel. change target and prediction, if absolute values are given as function input,
    # because calculating balance and accuracy requires un-scaled rel. change values.
    
    if abs_values_YN:
                       
        # Create rel. change targets
        val_target_temp = (val_target[1:] - val_target[:-1]) / val_target[:-1]
        val_pred_temp = (val_pred[1:] - val_pred[:-1]) / val_pred[:-1]  
    
        # check balance in up/down movements of validation data
        val_balance = np.round(sum(val_target_temp>=0) / len(val_target_temp), 3)
        
        # get prediction accuracy from rel. change prediction
        accuracy = np.round(sum(np.sign(val_target_temp) == np.sign(val_pred_temp)) / len(val_target_temp), 3)

    # If function input is already given as rel. change values, can directly calculate balance and accuracy
    else:
        # check balance in up/down movements of validation data
        val_balance = np.round(sum(val_target>=0) / len(val_target), 3)
        
        # get prediction accuracy from rel. change prediction
        accuracy = np.round(sum(np.sign(val_target) == np.sign(val_pred)) / len(val_target), 3)

    # get mean-absolute-error for deviation of predicted values from true targets
    mae = np.round(sum(np.abs(val_target - val_pred)) / len(val_target), 4)

        
    # Optionally print dimensions and metrics
    if verbose:
        print("val_input shape: ", val_input.shape)
        print("val_target shape: ", val_target.shape)
        print("val_pred shape: ", val_pred.shape)        
        print("W_in shape: ", W_in.shape)
        print("W_out shape: ", W_out.shape)
        print("W_res shape: ", W_res.shape)
        print("up movements percentage in val_target: ", val_balance)
        print("ESN pred. accuracy: ", accuracy)
        print("ESN mean abs. error: ", mae)

    
    # return values
    return val_target, val_pred, accuracy, mae


### Define function to get CNN/LSTM predictions and evaluation metrics on validation data:

## Function input: 
# Feed validation data with relative change values: val_input and val_target with
# dimensions (num samples, num timesteps, num features) and (num samples), respectively.

## Parameters:
# model: trained CNN/LSTM model
# target_length (int): Necessary information to reconstruct absolute values from relativ change values.
# abs_values_YN (True/False): Flags if we have absolute values as function input. This is important for
#                             correctly calculating accuracy. Function output series are absolute values
#                             in any case.
# scaled_YN (True/False): Flags if we have scaled inputs and targets. This is important to know, since  
#                         calculating the accuracy requires un-scaled rel. change values.
# train_min / train_max: If we have scaled inputs, we need min and max values used for scaling.
# abs_base: Optionally input initial absolute value to hook on, if omitted, set default: 1.
# verbose (True/False)): if True, print shapes of function input and output plus evaluation metrics

## Function output:
# Returns array containing true targets val_target (num samples), as un-scaled absolute or
# rel. change values, depending on attribute abs_values_YN: If True return absolute values, else rel. changes.
# Returns array containing predictions val_pred (num samples), as un-scaled absolute or
# rel. change values, depending on attribute abs_values_YN: If True return absolute values, else rel. changes.
# Returns accuracy as amount of correctly predicted up/down movements.
# Returns mean-absolute-error mae_rel_chg and mae_abs for deviation of predicted values from true targets,
# as rel. change values and absolute values, respectively.

def predCNNLSTM(val_input, val_target, model, target_length=1, abs_values_YN=False,
                scaled_YN=False, train_min=0.0, train_max=0.0, abs_base=1.0, verbose=True):
                       
    # Get predicted output from trained model for all validation inputs.
    # Note: val_pred comes with shape (num samples, 1), need to get rid of second dimension.
    val_pred = model.predict(val_input)[:,0]
    
    # Optionally re-scale target and prediction before calculating accuracy on un-scaled rel. change values.
    if scaled_YN:
        val_target = val_target * (train_max - train_min) + train_min
        val_pred = val_pred  * (train_max - train_min) + train_min
    
             
    ## Evaluate ESN prediction
    
    # Function input was given as rel. change or absolute values and is now un-scaled.    
    # Temporarily create rel. change target and prediction, if absolute values are given as function input,
    # because calculating balance and accuracy requires un-scaled rel. change values.
    
    if abs_values_YN:
                       
        # Create rel. change targets
        val_target_temp = (val_target[1:] - val_target[:-1]) / val_target[:-1]
        val_pred_temp = (val_pred[1:] - val_pred[:-1]) / val_pred[:-1]  
    
        # check balance in up/down movements of validation data
        val_balance = np.round(sum(val_target_temp>=0) / len(val_target_temp), 3)
        
        # get prediction accuracy from rel. change prediction
        accuracy = np.round(sum(np.sign(val_target_temp) == np.sign(val_pred_temp)) / len(val_target_temp), 3)

    # If function input is already given as rel. change values, can directly calculate balance and accuracy
    else:
        # check balance in up/down movements of validation data
        val_balance = np.round(sum(val_target>=0) / len(val_target), 3)
        
        # get prediction accuracy from rel. change prediction
        accuracy = np.round(sum(np.sign(val_target) == np.sign(val_pred)) / len(val_target), 3)

    # get mean-absolute-error for deviation of predicted values from true targets
    mae = np.round(sum(np.abs(val_target - val_pred)) / len(val_target), 4)
         
    
    # Optionally print dimensions and plot true vs. predicted absolute values
    if verbose:
        
        # Fidelity check: Plot true vs. predicted target values
        plt.figure(figsize=(16,8))
        plt.plot(range(len(val_target)),val_target,'b',label="true data", alpha=0.3)
        plt.plot(range(len(val_pred)),val_pred,'k',  alpha=0.8, label='pred data')
        plt.legend()
        plt.show()
        
               
    # Optionally print dimensions and metrics
    if verbose:
        print("val_input shape: ", val_input.shape)
        print("val_target shape: ", val_target.shape)
        print("val_pred shape: ", val_pred.shape)     
        print("up movements percentage in val_target: ", val_balance)
        print("ESN pred. accuracy: ", accuracy)
        print("ESN mean abs. error: ", mae)
        
    
    # return values
    return val_target, val_pred, accuracy, mae




### Define function to revert relative change values to absolute values.

## Function input: 
# Feed two time series with relative change values: prediction and true values.

## Parameters:
# abs_base: Optionally input initial absolute value to hook on, if omitted, set default: 1.
# target_length (int): Necessary information to reconstruct absolute values from relativ change values.
# verbose (True/False)): if True, print shapes of function input and output plus plot absolute time series.

## Function output:
# Returns two time series with absolute values: prediction and true values.

def rel_to_abs(true_values, pred_values, target_length=1, abs_base=1.0, verbose=True):
    
    # Initialize storage for time series with absolute values.
    true_values_abs = np.zeros(len(true_values))
    pred_values_abs = np.zeros(len(true_values))
    
    # Loop over input series, optionally hook on base value
    for i in range(len(true_values)):
        
        # First target_length values are hooked on the same base value (or default: 1), for simplicity.
        # To be more accurate, one needed target_length base values to hook on.
        if i < target_length:
            true_values_abs[i] = abs_base * (1 + true_values[i])
            pred_values_abs[i] = abs_base * (1 + pred_values[i])
        # Here: One-step prediction with "teacher-forcing", hence hook on true absolute values.
        # Note: Tage target_length into account, to hook on correct true value: target_length steps back.
        elif i >= target_length:
            true_values_abs[i] = true_values_abs[i-target_length] * (1 + true_values[i])
            pred_values_abs[i] = true_values_abs[i-target_length] * (1 + pred_values[i])

    
    # Optionally print dimensions and plot true vs. predicted absolute values
    if verbose:
        
        # Fidelity check: Plot true vs. predicted absolute values
        plt.figure(figsize=(16,8))
        plt.plot(range(len(true_values_abs)),true_values_abs,'b',label="true data", alpha=0.3)
        plt.plot(range(len(pred_values_abs)),pred_values_abs,'k',  alpha=0.8, label='pred data')
        plt.legend()
        plt.show()
        
        print("input true_values shape: ", true_values.shape)
        print("input pred_values shape: ", pred_values.shape)
        print("output true_values_abs shape: ", true_values_abs.shape)
        print("output pred_values_abs shape: ", pred_values_abs.shape)
    
    # return values
    return true_values_abs, pred_values_abs


### Define function to get all n_res final reservoir states for all n_samples input series for input_length steps:

## Function input: 
# Feed weight matrices W_in, W_res from trained ESN.
# Feed validation data with relative change values: val_input with dimensions (num samples, num timesteps).

## Parameters:
# activation ('tanh', 'sigmoid'): Specify function for transition of reservoir states. (Shoud be the same as in training!)
# verbose (True/False)): if True, print shapes of function output

## Function output:
# Returns array containing all n_res final reservoir states for all n_samples input series for all T input steps

def get_all_states(W_in, W_res, val_input, activation='tanh', verbose=True):
    
    ### Get reservoir states timeseries from trained base ESN.
    # Get number of reservoir units (n_res), number of samples (n_samples) and input length (T) from inputs
    n_res = len(W_res)
    n_samples = val_input.shape[0]
    T = val_input.shape[1]

    ## Use X here to store all n_res final reservoir states for all n_samples input series for all T input steps,
    ## use specified activation function.

    # initialize X (n_res x n_samples)
    X = np.zeros((n_res,n_samples, T))

    # Loop over n_samples    
    for i in range(n_samples):

        # Loop over timesteps in current sample
        for j in range(T):

            # If desired activation function is 'tanh':
            if activation=='tanh':
                # first input timestep needs special treatment, since reservoir state is not yet initialized
                if j == 0:
                    X[:,i:i+1,j] = np.tanh(W_in * val_input[i,j])
                elif j > 0:
                    X[:,i:i+1,j] = np.tanh(W_in * val_input[i,j] + np.reshape(np.matmul(W_res, X[:,i:i+1,j-1]),(n_res,1)))

            # If desired activation function is 'sigmoid':
            elif activation=='sigmoid':
                # first input timestep needs special treatment, since reservoir state is not yet initialized
                if j == 0:
                    X[:,i:i+1,j] = expit(W_in * val_input[i,j])
                elif j > 0:
                    X[:,i:i+1,j] = expit(W_in * val_input[i,j] + np.reshape(np.matmul(W_res, X[:,i:i+1,j-1]),(n_res,1)))

    # Optionally print output dimension
    if verbose:
        print("X shape (n_res, n_samples, input_length): ", X.shape)
    
    # Return values:
    return X