# Paper 1

### Title: "It's a long way! Layerwise Relevance Propagation for Echo State Networks applied to Earth System Variability."

**Abstract**

Artificial neural networks (ANNs) are known to be powerful methods for many hard problems (e.g. image classification or timeseries prediction). However, these models tend to produce black-box results and are often difficult to interpret. Here we present Echo State Networks (ESNs) as a certain type of recurrent ANNs, also known as reservoir computing. ESNs are easy to train and only require a small number of trainable parameters. They can be used not only for timeseries prediction but also for image classification, as shown here: Our ESN model serves as a detector for El Nino Southern Oscillation (ENSO) from sea-surface temperature anomalies. ENSO is actually a well-known problem and has been widely discussed before. But here we use this simple problem to open the black-box and apply layerwise relevance propagation to Echo State Networks.

**Plain Language Summary**

**1. Introduction**

* Related Work
* Our contribution

**2. Echo State Network**

* Sketch and parameters of base ESN, reservoir state transition for leaky reservoir.
* How to use base ESN for image classification: 
 * Coupled ESN, not shown here, since reservoirs become to large.
 * Feed image column-wise (or row-wise, not shown here).

**3. Layerwise Relevance Propagation**

* General theory behind LRP.
* Need to unfold timesteps as layers: It's a long way unfolding 180 timesteps, but it works!
 * First timestep is treated in a special way: No multiplication with leakrate alpha.
 * Total relevance is not preserved, since a portion alpha leaks out in every step, show exponential decay and highlight approriate choice of alpha, to enable model to remember the whole input sample for prediction.
 * And keep in mind that we work with single continuous target, not one-hot.

**4. Application to ENSO**

* The ENSO Pattern (composite El Nino / La Nina samples), sst anomaly index used for labelling sst anomaly fields and also used as target.
* mean LRP maps for MLP compared to our base ESN.
* Robustness against random permutation: MLP doesn't care, since we feed all gridpoints at once. But base ESN struggles, at least accuracy drops. Mean relevance maps still appear to be reasonable.

**5. Discussion and Conclusion**

* LRP with RNNs on 180 timesteps is a loooong way, but it works.
* Accuracy is competitive compared to MLP and linReg as baselines. Big advantage of baseESN is the low number of trainable parameters plus fast and stable convergence.
* A problem is, that classification of sst anomaly fields by feeding images col-wise into the model implicitely makes use of the zonal structure observed for ENSO patterns.
* Applying random permutation we still find reasonable relevance maps but accuracy drops.

Two fields for future work:
* We find a strong necessity to chose leakrate appropriately. This could be further investigated: Increasing leakrate leeds to fading memory.
* But application to geospatial data appears to be limited, since some zonal or meridional structure is required for robustness.
* However, we could use the same technique on timeseries prediction with base ESN models: Instead of feeding #latitude features for #longitude timesteps into the model, we could feed a certain number of climate indices over with specific input length. And still LRP should yield an alternative for otherwise often used temporal attention.

**Appendix**

* Technical details on used MLP and baseESN.
 

# Paper 2

### Title: "Sorry, I don't remember! Fading memory of Echo State Networks used for detecting ENSO."

- synthetic samples experiment: sweeping from left to right and vice versa
- LRP on ENSO with col-wise base ESN
- fading memory and de-creasing accuracy
- further work: Go back to timeseries prediction with same base ESN model and apply LRP (instead of Attention in LSTMs).


Paper Draft / points to keep in mind:
* Peer's suggestion: Split results in two papers.
 * Paper 1: Classification and LRP with ESN models on ENSO problem. Memory parameter alpha needs to be chosen appropriately. Teasing 2nd paper in future work.
 * Paper 2: Focus on memory parameter and fading memory effect. Then extend method to timeseries prediction, including LRP: Feed number of indices each with certain number of timesteps T into the model to predict T+1 for some target value.

---

* ESN models with competitive accuracy but with the advantage of far less trainable parameters, compared to baseline models.

Theory part of 1st ESN paper:
* Usually total relevance is conserved in each layer.
* That holds true for coupled ESN but not for col-wise base ESN: A certain part of the final relevance "leaks" out in every. timestep for input column.
* Hence the remaining relevance that is backpropagated shows an exponential decay (1-alpha)^t.
* AND note: 1st timestep is special, since there is no multiplication with alpha!
* So in the extreme case (alpha=0), the total relevance would be assigned to the very first column.
* By the way: So far I have omitted the very first column in the relevance calculation: Relevance maps have only dimension (89 x 179) but need (89 x 180) --> **fix that for paper!**

Regarding 2nd ESN paper:
* I could do the experiment on synthetic input samples. Class 1 and 2 identical on the right half of the sample (e.g. two dots, one above, one below). But samples differ on the left half (e.g. single dot above vs. dot below). Add enough noise!
* Then sweep with appropriate alpha to reach 100% accuracy.
* Increase alpha so that memory fades and classification fails - at least when sweeping from left to right.
* However - sweeping from right to left with increased alphs we should still reach 100% accuracy.
