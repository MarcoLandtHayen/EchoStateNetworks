# EchoStateNetworks
Project on Echo State Networks in all kinds of flavors.

* **ESN\_vs\_CNNLSTM\_amazon.ipynb** was the first experiment on manually setting up a basic ESN.
 * Prediction is done on Amazon stock price and results are compared with prediction of hybrid CNN/LSTM network.
 * First 1.500 steps are taken as train data. However, training is done "wrong": Input first step u(t=1) and noted resulting reservoir state X(t=1). Then input second step u(t=2) and noted X(t=2), and so on. Hence the input length is different for all the resulting reservoir states.
 * For prediction / validation then used constant input length of 1.500 steps.
* **ESN\_exp01.ipynb** covers the first systematic view an basic ESNs.
 * Set up everything manually. But use **functional form** for preparing inputs, training ESN and prediction.
 * Here we choose an input length and then use this input length for all input samples in *one* training run.
 * Later tune and play with hyperparameters, input length, number of reservoir units, target length (1d, 2d,.., 10d) and prepare using multi-reservoirs with lagged input series.
 * Analyse convergence / divergence behavior of reservoir states, depending on several parameters: Activation, input length, scaling of relative change input values, spectral radius, sparsity.
 * Distribution of trained output weights W_out and final reservoir states x(T) for all reservoir units for single sample.
* **ESN\_exp02.ipynb** plays with toy series and compares prediction performance of base ESN with CNN/LSTM hybrid network.
 * Extended own functions **ESN_functions_v01.py** for preparing inputs, training ESN and prediction.
 * Then set up 4 timeseries with increasing complexity based on sin-function with Gaussian noise.
 * Try do model "regime-shift" or in other words: non-stationarity.
 * Found similar performance of base ESN and CNN/LSTM with increasing noise in term of mean-absolute-error.
 * But ESN shows advantages in stability (spread of mae), number of parameters and training-time.
* **ESN\_exp03.ipynb**: Additive decommposition of timeseries.
 * Play with additive decomposition of timeseries into linear trend, seasonality and residual noise, as in [Kim and King, 2020] paper "Time series prediction using Deep Echo State Networks".
* **ESN\_exp04.ipynb**: Try custom implementation of ESN layer on base ESN.
 * After playing with additive decomposition of timeseries into linear trend, seasonality and residual noise, as in [Kim and King, 2020] paper "Time series prediction using Deep Echo State Networks" (--> exp03):
 * We are ready to set up implementation of customized ESN layer, based on tensorflow implementation, extendable to multiple ESN layer networks (= DeepESN).
* **ESN\_exp05.ipynb**: Set up first DeepESN model.
 * Use customized implementation of ESN layer with customized functions for setting up ESN model, training model and getting predictions with evaluation metrics. 
* **ESN\_wrapUP_decomposition_and_DeepESN.ipynb**: Summarize current knowledge on decomposing timeseries and setting up (Deep)ESN models with customized class layer and functions.
* **ESN\_exp06_ENSO.ipynb**: Apply ESN methodology to well known real-world proclem: ENSO
* **ESN\_exp07\_AggRel.ipynb**: Predicting ENSO index from its own history as only input feature is found to be limited to short prediction horizons (one to three months). Tried to use successively increasing number of gridpoints (from lat/lon grid in Nino3.4 region) and their sst timeseries as input features. Start e.g. with the gridpoint, that allone yields highest prediction accuracy on prediction ENSO index (or sst anomaly). Call the approach “Agglomerative Relevance”, inspired by paper „Physically Interpretable Neural Networks for the Geosciences: Applications to Earth System Variability“. However, the information contained in sst field for Nino3.4 box is limited. Need to take additional input features into account in subsequent experiments.
* **ESN\_exp08\_LRP.ipynb**: This experiments consists of various parts, stored in individual notebooks. The general outline is to try classification of sst anomaly fields according to El Nino / La Nina events with different model architectures. We then try layer-wise relevance propagation (LRP) for different approaches:
 * **ESN\_exp08\_LRP.ipynb**: Classify sst anomaly fields with linear regression and base ESN model on continuous sst anomaly index as target. Optimize baseESN (hyper-)parameters. Feed sst anomaly fields row-wise into baseESN.
 *	**ESN\_exp08\_LRP\_part2.ipynb**: Try LRP with multilayer perceptron (MLP) and baseESN. Still working with continuous sst anomaly index as target and feeding sst anomaly fields row-wise into baseESN.
 * **ESN\_exp08\_LRP\_part3**: Continued to find more perfection on LRP with baseESN.
 * **ESN\_exp08\_LRP\_part4**: Classification and LRP with baseESN on one-hot targets, feeding inputs column-wise into model.
 * **ESN\_exp08\_LRP\_part5**: Couple inputs directly to reservoir units by feeding whole. Sst anomaly field in first timestep. Then let reservoir swing for several timesteps, without additional inputs.


