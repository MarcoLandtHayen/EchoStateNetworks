# EchoStateNetworks
Project on Echo State Networks in all kinds of flavors.

* **ESN\_vs\_CNNLSTM\_amazon.ipynb** was the first experiment on manually setting up a basic ESN.
 * Prediction is done on Amazon stock price and results are compared with prediction of hybrid CNN/LSTM network.
 * First 1.500 steps are taken as train data. However, training is done "wrong": Input first step u(t=1) and noted resulting reservoir state X(t=1). Then input second step u(t=2) and noted X(t=2), and so on. Hence the input length is different for all the resulting reservoir states.
 * For prediction / validation then used constant input length of 1.500 steps.
* **ESN\_exp01.ipynb** covers the first systematic view an basic ESNs.
 * Still set up everything manually. But used functional form for preparing inputs, training ESN and prediction.
 * Here we use similar input length for all input samples.
 * Play with hyperparameters, targets (1d, 2d,.., 10d) and use multi-reservoirs with lagged input series.
