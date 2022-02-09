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