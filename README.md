# Bayesian Neural Networks

This repo is a work in progress and explores the applications of Bayesian Neural Networks (BNNs) for actuarial mortality modelling.

# Scope

Currently, 3 ways of 'training' a BNN are explored:
1. Neural network trained using regular methods, afterwhich the output is assumed to be Normally distributed with a variance term assigned a prior. Markov chain Monte Carlo (MCMC) is used to approximate the posterior for the variance term.
2. Neural network where all weights are iid Normal priors including a variance term, and MCMC is used to approximate the posteriors of the weights.
3. Neural network where all weights are iid Normal priors including a variance term, and variational inference (VI) is used to approximate the posteriors of the weights.

Symbolic regression is explored in Case #1 to illustrate some analytical approximations for the neural network.

# Initial Results

Here is an example of a discovered equation from the neural network for the log of the central mortality rate ($x$ is age, $t$ is time period (measured from $t=0=1950$), and $g$ is an indicator variable for gender ($g=1$ means Male)):

$$\log\left( m\left( x, t, g \right) \right) = \left( \frac{x - \mu_{x}}{\sigma_{x}} - \left( \frac{\frac{t - \mu_{t}}{\sigma_{t}}}{0.18350607} - \left( 0.075287916^{\frac{x - \mu_{x}}{\sigma_{x}}} + \frac{g}{0.075287916} \right) \right) \cdot 0.013670013 \right) \cdot 2.9507854 -5.799911$$

$x$ and $t$ are scaled, based on the training set. The above equation can be simplified as needed and constant terms expressed as parameters one can re-fit and investigate to see if any learnt 'laws of mortality' emerge.

Below are initial comparisons of a deep neural net (red) against repeated samples from a smaller BNN (green). The data reflects unseen test data (training/test split is done based on calendar years):

![image](https://github.com/patrickm663/bayesian-neural-networks/assets/77886027/6e2433d0-3868-4bb7-9ab7-1c4f0064f62f)

# Further Work

- The above is subject to change as the neural network re-evaulated and the method is extended to the BNN results.
- Further exploratory work will be added to compare traditional methods (e.g. Lee-Carter) and BNN extensions to tradtional methods.
- These approaches will also be stress-tested over different mortality datasets (Ireland is being used at the moment).

# Data

The data is Irish mortality data from 1950-2020, sourced from https://mortality.org.
