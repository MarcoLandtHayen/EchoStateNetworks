#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version: 04 (2022-03-04)
@author: mlandt-hayen

Content: Useful functions / class for EchoStateNetworks

- split_data
- rel_to_abs
- decompose_split
- decompose_gridsearch
- class ESN
- setESN
- trainESN
- predESN
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
import tensorflow.keras.initializers as tfi


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
# Returns series (samples, 1) train_target and val_target as abs. or rel. changes.
# Returns train_min and train_max, used for scaling, set to ZERO, if scaled_YN=False.

def split_data(data_abs, input_length, target_length, time_lag=0, 
               train_val_split=0.8, val_samples_from='end', abs_to_rel_YN=True, binary_YN=False, scaled_YN=False,
               scale_to='zero_one', verbose=True):
    
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
  
    # Optionally scale train and validation inputs and targets to [0,1] or [-1,1]
    # according to min/max obtained from train inputs.
    # Initialize train_min and _max to zero, to have return values even if no scaling is desired.
    train_min = 0
    train_max = 0
    if scaled_YN:
        train_min = np.min(train_input)
        train_max = np.max(train_input)

        if scale_to == 'zero_one':
            
            # Scale input values and targets to [0,1], according to min/min of ONLY train inputs:
            # substract min and divide by (max - min)
            train_input = (train_input - train_min) / (train_max - train_min)
            val_input = (val_input - train_min) / (train_max - train_min)
            train_target = (train_target - train_min) / (train_max - train_min)
            val_target = (val_target - train_min) / (train_max - train_min)
            
            # Alternatively scale input values and targets to [-1,1]
        elif scale_to == 'one_one':
            train_input = 2 * (train_input - train_min) / (train_max - train_min) - 1
            val_input = 2 * (val_input - train_min) / (train_max - train_min) - 1
            train_target = 2 * (train_target - train_min) / (train_max - train_min) - 1
            val_target = 2 * (val_target - train_min) / (train_max - train_min) - 1
            

    # Re-sample targets to have dimension (samples, 1), as model prediction also have this dimension.
    train_target = np.reshape(train_target, (train_target.shape[0], 1))
    val_target = np.reshape(val_target, (val_target.shape[0], 1))

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



## Define function "decompose_split":
# First step is additive decomposing given time series of absolute values into trend, seasonality and residual noise.
# Second step is splitting the original series plus obtained series from decomposition into samples of desired
# input length. Then create targets from specified target length and split data into training and validation sets.
#
## Additional input parameters for decomposing: 
# Smoothing parameters alpha and gamma from [0,1]
# Length of season s (int).
# scaled_YN (True/False):if True, scale input_series to [0,1] applying min/max scaling before decomposing.
# 
## Additional input parameters for splitting: 
# input length, target length (int)
# train_val_split from [0,1] to specify the relative amount of samples used for training.
#
# verbose: if True, output some information on dimensionalities, plot results from decomposing and give 
#          results from augmented Dickey Fuller test.
#
## Function output:
# Returns arrays (samples, timesteps, features) train_input and val_input as abs. values.
# Returns train_min and train_max, used for scaling, set to ZERO, if scaled_YN=False.

def decompose_split(input_series, input_length, target_length, train_val_split,
                    alpha, gamma, s, scaled_YN=False, verbose=False):
    
    ## First step: Decomposing given input series.
    
    # Get length T of input series:
    T = len(input_series)
     
    # Initialize arrays for storing trend L, seasonality S and residual noise R:
    L = np.zeros(T)
    S = np.zeros(T)
    R = np.zeros(T)
    
    # Optionally scale input series to [0,1] with min/max scaling.
    # Note: Get min/max only from train inputs, pretending to not know validation data, yet.
    # Initialize input_min and _max to zero, to have return values even if no scaling is desired.
    train_min = 0
    train_max = 0
    if scaled_YN:
        train_min = np.min(input_series[:int(T * train_val_split)])
        train_max = np.max(input_series[:int(T * train_val_split)])
        
        # substract min and divide by (max - min)
        input_series = (input_series - train_min) / (train_max - train_min)
    
    # First step (t=0) needs special treatment, since we don't have L(t-1):
    L[0] = input_series[0]
    S[0] = 0
    R[0] = 0
    
    # Loop over input series and calculate trend, seasonality and residual noise.
    # Note: Start at 1, since step (t=0) is already taken care of.
    for t in range(1, T):
        
        # First s timesteps need special treatment, since we don't have S(t-s) for these steps:
        if t < s:
            L[t] = alpha * input_series[t] + (1 - alpha) * L[t-1]
            S[t] = gamma * (input_series[t] - L[t])
            R[t] = input_series[t] - L[t] - S[t]
        else:
            L[t] = alpha * (input_series[t] - S[t-s]) + (1 - alpha) * L[t-1]
            S[t] = gamma * (input_series[t] - L[t]) + (1 - gamma) * S[t-s]
            R[t] = input_series[t] - L[t] - S[t]
    
    # Reshape input_series, L, S and R to have feature as second dimensios: New shape (T, 1).
    input_series = input_series.reshape((T,1))
    L = L.reshape((T,1))
    S = S.reshape((T,1))
    R = R.reshape((T,1))
    
    # Get min/max for scaled input series, L, S and R for plotting vertical barrier separating train and val data:
    input_series_min = np.min(input_series)
    input_series_max = np.max(input_series)
    L_min = np.min(L)
    L_max = np.max(L)
    S_min = np.min(S)
    S_max = np.max(S)
    R_min = np.min(R)
    R_max = np.max(R)
        
    # Concatenate input_series, L, S and R to have ONE series with 4 features, shape (T, 4).
    data = np.concatenate((input_series, L, S, R), axis=-1)
    
    ## Second step: Splitting data (consisting of input_series and decomposed parts L, S, R).
    
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
    
    # Create targets from original input series, store in Y.
    Y = input_series[target_length:]
    
    # Cut the first (input_length - 1) targets, since we don't have input samples for these targets.
    Y = Y[input_length-1:]
    
    # Split inputs and targets into train and validation data according to train_val_split
    n_train = int(len(X) * train_val_split)

    train_input = X[:n_train]
    val_input = X[n_train:]
    train_target = Y[:n_train]
    val_target = Y[n_train:]

    # Optionally print dimensions and plot (optionally) scaled original series plus decomposed parts.
    if verbose:
        print("original input series shape: ", input_series.shape)
        print("train_input shape: ", train_input.shape)
        print("val_input shape: ", val_input.shape)
        print("train_target shape: ", train_target.shape)
        print("val_target shape: ", val_target.shape)
        print("train_min: ", train_min)
        print("train_max: ", train_max)
        
        # Plot original timeseries, together with decomposed parts: Trend L, seasonality S and residual noise R.
        # Note: Concatenate train and validation input, shape (samples, timesteps, features).
        #       Take last timestep for each sample and all features seperately.
        fig, axes = plt.subplots(4, 1, figsize=(15,15))
        axes[0].plot(range(1, T-input_length-target_length+2),
                     np.concatenate((train_input[:,-1,0],val_input[:,-1,0])), color="k", label="original data")  
        axes[1].plot(range(1, T-input_length-target_length+2),
                     np.concatenate((train_input[:,-1,1],val_input[:,-1,1])), color="b", label="Trend")
        axes[2].plot(range(1, T-input_length-target_length+2),
                     np.concatenate((train_input[:,-1,2],val_input[:,-1,2])), color="b", label="Seasonality")
        axes[3].plot(range(1, T-input_length-target_length+2),
                     np.concatenate((train_input[:,-1,3],val_input[:,-1,3])), color="b", label="Residuum")
        axes[3].set_xlabel("timestep", fontsize=12)
        axes[0].set_ylabel("original data", fontsize=12)
        axes[1].set_ylabel("trend", fontsize=12)
        axes[2].set_ylabel("season", fontsize=12)
        axes[3].set_ylabel("residuum", fontsize=12)
        axes[0].set_title("Original input data and decomposed parts (additive decomposing)")
        
        # Include vertical barriers separating train and validation data.
        axes[0].plot([n_train,n_train],[input_series_min+np.spacing(1),input_series_max-np.spacing(1)],'k:', linewidth=3)
        axes[1].plot([n_train,n_train],[L_min+np.spacing(1),L_max-np.spacing(1)],'k:', linewidth=3)
        axes[2].plot([n_train,n_train],[S_min+np.spacing(1),S_max-np.spacing(1)],'k:', linewidth=3)
        axes[3].plot([n_train,n_train],[R_min+np.spacing(1),R_max-np.spacing(1)],'k:', linewidth=3)

        plt.show()
        
        # Now apply augmented Dickey Fuller test (ADFT) to optionally scaled original input series and 
        # results from decomposition: Trend, seasonality and residuum.
        # Note: Concatenate train and validation input, shape (samples, timesteps, features).
        #       Take last timestep for each sample and all features seperately.
        ADFT_orig = adfuller(np.concatenate((train_input[:,-1,0],val_input[:,-1,0])))
        ADFT_L = adfuller(np.concatenate((train_input[:,-1,1],val_input[:,-1,1])))
        ADFT_S = adfuller(np.concatenate((train_input[:,-1,2],val_input[:,-1,2])))
        ADFT_R = adfuller(np.concatenate((train_input[:,-1,3],val_input[:,-1,3])))

        # Print results from ADFT:
        print('\noriginal:')
        print('=========')
        print('ADF Statistic: %f' % ADFT_orig[0])
        print('p-value: %f' % ADFT_orig[1])
        
        print('\nTrend L:')
        print('========')
        print('ADF Statistic: %f' % ADFT_L[0])
        print('p-value: %f' % ADFT_L[1])
        
        print('\nSeason S:')
        print('=========')
        print('ADF Statistic: %f' % ADFT_S[0])
        print('p-value: %f' % ADFT_S[1])
        
        print('\nResiduum R:')
        print('===========')
        print('ADF Statistic: %f' % ADFT_R[0])
        print('p-value: %f' % ADFT_R[1])
        print('\n\nCritical Values:')
        for key, value in ADFT_R[4].items():
            print('\t%s: %.3f' % (key, value))

    # Return values:
    # train_input and val_input contain original input series, trend L, seasonality S and residual noise R.
    # train_target and val_target contain targets from original input series.
    return train_input, val_input, train_target, val_target, train_min, train_max        


## Define function to perform gridsearch on smoothing parameters alpha and beta, plus run series of cycle length s.
#  Optionally scale input series of absolute values to [0,1] applying min/max scaling before decomposing.
# 
## Additional input parameters for decomposing: 
# Smoothing parameters alpha and gamma running from given _min to _max value [0,1] in given number of _steps (int).
# Store p-value from augmented Dickey Fuller test (ADFT) for as heatmap (alpha-gamma-grid) for 
# original input_series, L, S and R seperately.
# Create heatmaps for all cycle lengths from _min to _max (int) in given number of _steps (int).
# scaled_YN (True/False):if True, scale input_series to [0,1] applying min/max scaling before decomposing.
# verbose: if True, output some information on dimensionalities and results from augmented Dickey Fuller test on
#          (optionally scaled) original input_series.
#
## Function output:
# Returns array of p-values from gridsearch, shape (alpha_steps, gamma_steps, num_features, s_steps).
# Here "num_features = 3", since we store p-values for L, S and R, seperately, but don't store p-value for original
# input series, to save computing time.
# Returns series used for gridsearch (alpha, gamma, s).
# Returns input_series_min and input_series_max, used for scaling, set to ZERO, if scaled_YN=False.

def decompose_gridsearch(input_series,
                         alpha_min, alpha_max, alpha_steps,
                         gamma_min, gamma_max, gamma_steps,
                         s_min, s_max, s_steps,
                         scaled_YN=False, verbose=False):
    
    ## First step: Decomposing given input series.
    
    # Get length T of input series:
    T = len(input_series)
     
    # Initialize arrays for storing trend L, seasonality S and residual noise R:
    L = np.zeros(T)
    S = np.zeros(T)
    R = np.zeros(T)
    
    # Optionally scale input series to [0,1] with min/max scaling.
    # Note: Get min/max only from train inputs, pretending to not know validation data, yet.
    # Initialize input_min and _max to zero, to have return values even if no scaling is desired.
    input_series_min = 0
    input_series_max = 0
    if scaled_YN:
        input_series_min = np.min(input_series)
        input_series_max = np.max(input_series)
        
        # substract min and divide by (max - min)
        input_series = (input_series - input_series_min) / (input_series_max - input_series_min)
    
    # First step (t=0) needs special treatment, since we don't have L(t-1):
    L[0] = input_series[0]
    S[0] = 0
    R[0] = 0
    
    # Initialize storage for p-value heatmaps, shape: (alpha_steps, gamma_steps, num_features, s_steps).
    # Here "num_features = 3", since we only store L, S and R, seperately.
    p_values = np.zeros((alpha_steps, gamma_steps, 3, s_steps))
                                
    # Create linspaces for alpha, gamma and s:
    alpha_series = np.linspace(alpha_min, alpha_max, alpha_steps)
    gamma_series = np.linspace(gamma_min, gamma_max, gamma_steps)
    s_series = np.linspace(s_min, s_max, s_steps)
    
    # Loop over alpha, gamma and s:
    for current_alpha in range(alpha_steps):
        print("\n Status: alpha ", current_alpha+1, " of ", alpha_steps)
        for current_gamma in range(gamma_steps):
            print("Status: gamma ", current_gamma+1, " of ", gamma_steps)
            for current_s in range(s_steps):
                print("Status: s ", current_s+1, " of ", s_steps)
                
                # Get current alpha, gamma and s:
                alpha = alpha_series[current_alpha]
                gamma = gamma_series[current_gamma]
                s = int(s_series[current_s])  # Need to force s to be integer, since used as timestep.                
                                
                # Loop over input series and calculate trend, seasonality and residual noise.
                # Note: Start at 1, since step (t=0) is already taken care of.
                for t in range(1, T):

                    # First s timesteps need special treatment, since we don't have S(t-s) for these steps:
                    if t < s:
                        L[t] = alpha * input_series[t] + (1 - alpha) * L[t-1]
                        S[t] = gamma * (input_series[t] - L[t])
                        R[t] = input_series[t] - L[t] - S[t]
                    else:
                        L[t] = alpha * (input_series[t] - S[t-s]) + (1 - alpha) * L[t-1]
                        S[t] = gamma * (input_series[t] - L[t]) + (1 - gamma) * S[t-s]
                        R[t] = input_series[t] - L[t] - S[t]
    
                # Now apply augmented Dickey Fuller test (ADFT) to results from decomposition: L, S and R.
                ADFT_L = adfuller(L)
                ADFT_S = adfuller(S)
                ADFT_R = adfuller(R)

                # Store p-values for original input series, L, S and R.                
                p_values[current_alpha, current_gamma, 0, current_s] = ADFT_L[1]  
                p_values[current_alpha, current_gamma, 1, current_s] = ADFT_S[1]  
                p_values[current_alpha, current_gamma, 2, current_s] = ADFT_R[1]  
                   
    # Optionally print dimensions and plot (optionally) scaled original series plus decomposed parts.
    if verbose:
        print("\n\noriginal input series shape: ", input_series.shape)
        print("p_values shape: ", p_values.shape)
        
        print("input_series_min: ", input_series_min)
        print("input_series_max: ", input_series_max) 
        
        # Apply ADFT to original (optionally scaled) input series:
        ADFT_orig = adfuller(input_series)
        
        # Print results from ADFT:
        print('\noriginal series:')
        print('================')
        print('ADF Statistic: %f' % ADFT_orig[0])
        print('p-value: %f' % ADFT_orig[1])
        
    # Return values:
    # train_input and val_input contain original input series, trend L, seasonality S and residual noise R.
    # train_target and val_target contain targets from original input series.
    return p_values, alpha_series, gamma_series, s_series, input_series_min, input_series_max



### Define custom ESN layer, extending existing tensorflow class Layer
#
## Called with parameters for initialization:
# n_res: Number of reservoir units in this layer
# W_in_lim: Determines the range for initialization of input weights W_in with RandomUniform in [-W_in_lim,+W_in_lim].
# leak_rate: Used in reservoir state transition
# leak_rate_first_step_YN: If True, use multiplication with alpha already for calculating first timestep's res. states.
# activation (from ['tanh', 'sigmoid']): Choose activation function to be used in reservoir state transition.
#
## Function output:
# Returns Tensor X with all reservoir states for all samples for all timesteps, shape: (samples, timesteps, n_res)
# Returns Tensor X_T with all FINAL reservoir states for all samples, shape: (samples, n_res)

class ESN(tf.keras.layers.Layer):
    def __init__(self, n_res, W_in_lim=1, leak_rate=0.5, leak_rate_first_step_YN=True, activation='tanh', verbose=0):
        super(ESN, self).__init__()
        
        # Setup reservoir units
        self.n_res = n_res
        self.W_in_lim = W_in_lim
        self.leak_rate = leak_rate
        self.leak_rate_first_step_YN = leak_rate_first_step_YN
        self.activation = activation
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
        self.verbose=verbose
    
    def call(self, inputs):
    
        # Connect inputs to reservoir units to get initial reservoir state (t=1), called x_prev, since
        # it will be used as "previous" state when calculating further reservoir states.
        
        # Apply desired activation function for reservoir state transition: 'tanh' or 'sigmoid'
        if self.activation=='tanh':
            
            # Optionally omit multiplication with alpha for calculating first timestep's reservoir states:
            if self.leak_rate_first_step_YN:
                x_prev = self.leak_rate * tf.tanh(self.res_units_init(inputs[:,0:1,:])) # x(1) = leak_rate * tanh(W_in * u(1))
            else:
                x_prev = tf.tanh(self.res_units_init(inputs[:,0:1,:])) # x(1) = tanh(W_in * u(1))
        
        elif self.activation=='sigmoid':
            
            # Optionally omit multiplication with alpha for calculating first timestep's reservoir states:
            if self.leak_rate_first_step_YN:
                x_prev = self.leak_rate * tf.keras.activations.sigmoid(self.res_units_init(inputs[:,0:1,:])) # x(1) = leak_rate * sigm(W_in * u(1))
            else:
                x_prev = tf.sigmoid(self.res_units_init(inputs[:,0:1,:])) # x(1) = sigm(W_in * u(1))
        
        # Initialize storage X for all reservoir states (samples, timesteps, n_res):
        # Store x_prev as x_1 in X
        X = x_prev
        
        # Now loop over remaining time steps t = 2..T
        T = inputs.shape[1]
        
        for t in range(1,T):
            
            # Considered desired activation function for state transition:
            if self.activation=='tanh':
                x_t = (1 - self.leak_rate) * x_prev + self.leak_rate * tf.tanh(self.res_units_init(inputs[:,t:t+1,:]) + self.res_units(x_prev))
            elif self.activation=='sigmoid':
                x_t = (1 - self.leak_rate) * x_prev + self.leak_rate * tf.keras.activations.sigmoid(self.res_units_init(inputs[:,t:t+1,:]) + self.res_units(x_prev))
    
            # Store x_t in X
            X = tf.concat([X, x_t], axis=1)
            
            # x_t becomes x_prev for next timestep
            x_prev = x_t
        
        # Return both: ALL reservoir states X and final reservoir states X[T].
        return X, X[:,-1,:]


## Define function setESN to set up ESN model.
#
# Sets up an ESN model with desired number of ESN layers. Then modifies reservoir weights for all ESN layers,
# to fulfill desired properties according to specified spectral radius.
#
## Input parameters:
# input_length (int): Specified number of timesteps per input sample.
# n_features: Number of input features, e.g. original series plus decomposed parts L, S and R --> 4
# n_layers (int): Number of ESN layers in the model.
# n_res (int): Number of reservoir units.
# W_in_lim: Initialize input weights from random uniform distribution in [- W_in_lim, + W_in_lim]
# leak_rate: Leak rate used in transition function of reservoir states.
# spec_radius: Spectral radius, becomes largest Eigenvalue of reservoir weight matrix
# sparsity: Sparsity of reservoir weight matrix.
#
## Function output:
# Returns complete model "model".
# Returns short model "model_short" without output layer, for getting final reservoir states for given inputs.
# Returns model "all_states" without output layer, for getting reservoir states for ALL timesteps for given inputs.

def setESN(input_length, n_features, n_layers, n_res, W_in_lim, leak_rate, leak_rate_first_step_YN, activation, spec_radius, sparsity, verbose=False):
    
    ## Set up model
    
    # Input layer
    model_inputs = Input(shape=(input_length, n_features)) # (timesteps, features)
    
    # Set up storage for layers' final reservoir state tensors:    
    X_T_all = []
    
    ## Loop for setting up desired number of ESN layers:
    for l in range(n_layers):
        
        # First ESN needs to be connected to model_inputs:
        if l == 0:
            
            # Use custom layer for setting up reservoir, returns ALL reservoir states X and FINAL reservoir states X_T.
            X, X_T = ESN(n_res=n_res, W_in_lim=W_in_lim, leak_rate=leak_rate,
                         leak_rate_first_step_YN=leak_rate_first_step_YN,
                         activation=activation)(model_inputs)
            
            # Store resulting final reservoir states:            
            X_T_all.append(X_T)
            
        # Further ESN layers need to be connected to previous ESN layer:
        else:
            
            # Use new custom layer for setting up reservoir, again returns ALL reservoir states X and 
            # FINAL reservoir states X_T.
            X, X_T = ESN(n_res=n_res, W_in_lim=W_in_lim, leak_rate=leak_rate,
                         leak_rate_first_step_YN=leak_rate_first_step_YN,
                         activation=activation)(X)
            
            # Store resulting final reservoir states:            
            X_T_all.append(X_T)
            
    ## Concatenate final reservoir states from ALL layers before passing result to output layer:

    # In case we only have ONE layer, no concatenation is required:
    if n_layers == 1:
        X_T_concat = X_T_all[0]

    # Else concatenate stored final reservoir states using lambda-function:
    else:
        X_T_concat = Lambda(lambda x: concatenate(x, axis=-1))(X_T_all)

    # Output unit
    output = Dense(units=1, activation=None, use_bias=True, 
                   kernel_initializer=tfi.RandomUniform(minval=-W_in_lim, maxval=W_in_lim, seed=None),
                   bias_initializer=tfi.RandomUniform(minval=-W_in_lim, maxval=W_in_lim, seed=None),
                   name='output')(X_T_concat)

    # Define complete model "model" plus short model "model_short" omitting the output layer, 
    # for getting reservoir states for given inputs.
    model = Model(model_inputs, output, name='model')
    model_short = Model(model_inputs, X_T_concat, name='model_short')
    
    # Define another shortened model "all_states" to get all reservoir states X from last ESN layer:
    all_states = Model(model_inputs, X, name='all_states')    
    
    ## Modify reservoir weights W_res using spectral radius:

    # Get model weights for ALL layers
    model_weights = np.array(model.get_weights())

    # Loop over desired number of ESN layers for reservoir weights:
    for l in range(n_layers):

        # Extract reservoir weights
        W_res = model_weights[2 + (l * 4)]

        # Need temporary matrix W_temp to implement sparsity manually
        W_temp = np.random.uniform(low=0, high=1, size=(n_res,n_res))
        W_sparse = W_temp <= sparsity

        # Now apply sparsity to initial W_res
        W = W_sparse * W_res

        # Get largest Eigenvalue of W
        ev_max = np.max(np.real(np.linalg.eigvals(W)))

        # Finally set up W_res
        W_res = spec_radius * W / ev_max

        # Integrate modified reservoir weights back into model weights
        
        # Extract reservoir weights
        model_weights[2 + (l * 4)] = W_res

    # Get modified reservoir weights for all ESN layers back into the model
    model.set_weights(model_weights)
    
    
    # Optionally reveal model summaries proof of sparsity and max. Eigenvalues for reservoir weights
    if verbose:
        
        # Print model summaries
        model.summary()
        model_short.summary()  
        
       # Check sparsity and max Eigenvalues for ALL ESN layers' reservoir weights:
        # Get model weights for ALL layers
        model_weights = np.array(model.get_weights())

        # Loop over layers:
        for l in range(n_layers):
            W_res = model_weights[2 + (l * 4)]            

            print("\nLayer ", l+1)
            print("========")
            print("W_res sparsity: ", sum(sum(W_res != 0)) / (W_res.shape[0]**2))
            print("W_res max EV: ", np.max(np.real((np.linalg.eigvals(W_res)))))

    # Return models
    return model, model_short, all_states


## Define function trainESN to train output weights and bias for an already set up ESN model.
#
## Input parameters:
#
# model: complete ESN model, as returned from e.g. setESN.
# model_short: Short model as provided by e.g. setESN, without output layer, 
#              for getting reservoir states for given inputs.
# train_input: Input samples (samples, timesteps, input features) to be used for training.
# train_target: True targets for train inputs (samples, output features).
# verbose (True/False)): if True, plot histogram of trained output weights and give additionala information
#                        on bias after training
#
## Function output:
# Returns complete model "model" with trained output weights and bias.

def trainESN(model, model_short, train_input, train_target, verbose=False):
    
    # Get final reservoir states for all train samples from short model    
    X_T_train = model_short.predict(train_input)

    # Extract output weights and bias.
    # Note: output layer is the LAST layer of the model, find weights and bias at position "-2" and "-1", respectively.
    model_weights = np.array(model.get_weights())
    W_out = model_weights[-2]
    b_out = model_weights[-1]
    
    # Create vector of shape (samples, 1) containing ONEs to be added as additional column to final reservoir states.
    X_add = np.ones((X_T_train.shape[0], 1))
    
    # Now add vector of ONEs as additional column to final reservoir states X_T_train.
    X_T_train_prime = np.concatenate((X_T_train, X_add), axis=-1)
    
    # Then need pseudo-inverse of final reservoir states in augmented notation
    X_inv_prime = np.linalg.pinv(X_T_train_prime)

    # Then get output weights, in augmented notation
    W_out_prime = np.matmul(X_inv_prime, train_target)
   
    # Now split output weights in augmented notation into trained output weights W_out and output bias b_out.
    W_out = W_out_prime[:-1,:]
    b_out = W_out_prime[-1:,0]
    #print("\nW_out: \n", W_out)
   
    # Integrate trained output weights and bias into model weights
    model_weights[-2] = W_out
    model_weights[-1] = b_out
    model.set_weights(model_weights)

    
    # Optionally reveal model summaries proof of sparsity and max. Eigenvalues for reservoir weights
    if verbose:
        
        print("\nshape of train input (samples, timesteps, input features): ", train_input.shape)
        print("shape of model output X_T (samples, n_res): ", X_T_train.shape)

        print("\nW_out shape: ", W_out.shape)
        print("b_out shape: ", b_out.shape)

        print("\nFinal reservoir states in augmented notation, shape: ", X_T_train_prime.shape)

        print("\ntrain_target shape (samples, output features): ", train_target.shape)        
        print("W_out_prime shape: ", W_out_prime.shape)

        print("\ntrained b_out: \n", b_out)
        
        # Plot histogram of trained output weights
        nBins = 100
        fig, axes = plt.subplots(1, 1, figsize=(10,5))
        axes.hist(W_out[:,0], nBins, color="blue")
        axes.set_ylabel("counts")
        axes.set_title("Histogram of trained output weights")
        plt.show()

    return model


## Define function predESN to evaluate performance of trained ESN model.
#
## Input parameters:
#
# model: complete ESN model, as returned from e.g. setESN.
# train_input: Input samples (samples, timesteps, input features) to be used for training.
# val_input: Input samples (samples, timesteps, input features) to be used for validation.
# train_target: True targets for train inputs (samples, output features).
# val_target: True targets for validation inputs (samples, output features).
# verbose (True/False)): if True, reveal details on model performance and show fidelity plots.
#
## Function output:
# Returns model predictions on train and validation inputs.
# Returns evaluation metrics 'mean-absolute-error' (mae) and 'mean-squared-error' (mse) on train and val. data.

def predESN(model, train_input, val_input, train_target, val_target, verbose=False):
    
    ## Get predictions from "long" model on train and validation input
    val_pred = model.predict(val_input)
    train_pred = model.predict(train_input)
    
    # Calculate mean-absolute and mean-squared error of model predictions compared to targets:
    train_mae = np.round(sum(np.abs(train_target[:,0] - train_pred[:,0])) / len(train_target), 4)
    val_mae = np.round(sum(np.abs(val_target[:,0] - val_pred[:,0])) / len(val_target), 4)
    train_mse = np.round(sum((train_target[:,0] - train_pred[:,0])**2) / len(train_target), 4)
    val_mse = np.round(sum((val_target[:,0] - val_pred[:,0])**2) / len(val_target), 4)
    
    # Optionally reveal model summaries proof of sparsity and max. Eigenvalues for reservoir weights
    if verbose:
        
        print("\nshape of val input (samples, timesteps, features): ", val_input.shape)
        print("shape of train input (samples, timesteps, features): ", train_input.shape)

        print("\nshape of model predictions on validation input (samples, 1): ", val_pred.shape)
        print("shape of val targets (samples, 1): ", val_target.shape)

        print("\ntrain_mae: ", train_mae)
        print("val_mae: ", val_mae)
        
        print("\ntrain_mse: ", train_mse)
        print("val_mse: ", val_mse)
        
        # Fidelity check: Plot train_pred vs. train_targets
        plt.figure(figsize=(16,8))
        plt.plot(range(len(train_target)),train_target,'b',label="true data", alpha=0.3)
        plt.plot(range(len(train_pred)),train_pred,'k',  alpha=0.8, label='pred ESN')
        plt.title('Fidelity check on TRAIN data', fontsize=16)
        plt.xlabel('timestep', fontsize=14)
        plt.ylabel('target', fontsize=14)
        plt.legend(fontsize=14)
        plt.show()

        # Fidelity check: Plot val_pred vs. val_targets
        plt.figure(figsize=(16,8))
        plt.plot(range(len(val_target)),val_target,'b',label="true data", alpha=0.3)
        plt.plot(range(len(val_pred)),val_pred,'k',  alpha=0.8, label='pred ESN')
        plt.title('Fidelity check on VALIDATION data', fontsize=16)
        plt.xlabel('timestep', fontsize=14)
        plt.ylabel('target', fontsize=14)
        plt.legend(fontsize=14)
        plt.show()
        
    return train_pred, val_pred, train_mae, val_mae, train_mse, val_mse
