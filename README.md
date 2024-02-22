# Classification Surrogates

This Repo contains the model, training and evaluation files used for the Paper ...

## Structure

The <code>models/</code> directory contains all necessary definitions for definition of a conditional CFM model. 
  * <code>models/cond_CFM.py</code> defines our a simple CFM model modded to handle conditional generation (see also https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa).
  * <code>models/custem_linear_flipout.py</code> defines a wrapper around the LinearFlipout Layer from https://github.com/IntelLabs/bayesian-torch to surpress sampling of the weights at every call during ODE solving.
  * <code>models/MCMC_Adam.py</code> contains the code for MCMC sampling the network weights. For more Detail we refer to https://github.com/sbieringer/csMALA.
  * <code>jet_dataset.py</code> defines the wrapper used for data handling.

For Training of a (Variational Inference Bayes-) CFM run <code>cond_flow_matching.py</code>. To subsequnetly sample from the point estimate using AdamMCMC, we provide <code>cond_flow_matching_AdamMCMC.py</code>. 

<code>evalution.ipynb</code> and <code>plotting.ipynb</code> can be used to generate the plots of the paper.
