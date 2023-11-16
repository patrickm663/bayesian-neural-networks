# Bayesian Neural Networks

This repo is a work in progress and explores the applications of Bayesian Neural Networks (BNNs) for actuarial mortality modelling.

# Scope

Currently, 3 ways of 'training' a BNN are explored:
1. Neural network trained using regular methods, afterwhich the output is assumed to be Normally distributed with a variance term assigned a prior. Markov chain Monte Carlo (MCMC) is used to approximate the posterior for the variance term.
2. Neural network where all weights are iid Normal priors including a variance term, and MCMC is used to approximate the posteriors of the weights.
3. Neural network where all weights are iid Normal priors including a variance term, and variational inference (VI) is used to approximate the posteriors of the weights.

Symbolic regression is explored in Case #1 to illustrate some analytical approximations for the neural network.

# Initial Results

## Regular Neural Network with Variance Prior vs Bayesian Neural Network (using MCMC)
Below are initial comparisons of a deep neural net (red) against repeated samples from a smaller BNN (green). The data reflects unseen test data (training/test split is done based on calendar years):

![image](https://github.com/patrickm663/bayesian-neural-networks/assets/77886027/6e2433d0-3868-4bb7-9ab7-1c4f0064f62f)

Both performed well on unseen data in future years. It remains to be seen how they forecast and differences in performance when we have low data columes. Making the variance term non-constant but instead linked to the time period may assist in expressing forecast uncertainty.

## Bayesian Neural Network for Symbolic Equation Discovery

Here is an example of a discovered equation from the BNN for the log of the central mortality rate ($x$ is age, $t$ is time period (measured from $t=0=1950$), and $g$ is an indicator variable for gender ($g=1$ means Male)):

$$\log\big( m\left( x, t, g \right) \big) = \left( \frac{x - \mu_{x}}{\sigma_{x}} - \left( \frac{\frac{t - \mu_{t}}{\sigma_{t}}}{0.18350607} - \left( 0.075287916^{\frac{x - \mu_{x}}{\sigma_{x}}} + \frac{g}{0.075287916} \right) \right) \cdot 0.013670013 \right) \cdot 2.9507854 -5.799911$$

$x$ and $t$ are scaled, based on the training set. The above equation can be generalised as follows:

$$\log\big( m\left( x, t, g \right) \big) = A + B\cdot x - c\cdot t + D\cdot F^x + G\cdot g$$

Refitting the equation above using MCMC provides a reasonable, smooth equation for Irish mortality:

![sym_plot](https://github.com/patrickm663/bayesian-neural-networks/assets/77886027/a6d888c2-c07d-43e5-b35b-befaff61043e)

The equation can be roughly expressed as:

$$\log\big( m\left( x, t, g \right) \big) = -9.29 + 0.09\cdot x - 0.02\cdot t + 4.5\cdot 0.66^x + 0.54\cdot g$$

Taking it one further and performing some numerical integration, we can calculate life expectancies based off of our derived equation:

![life_exp](https://github.com/patrickm663/bayesian-neural-networks/assets/77886027/9441749a-e633-4d5f-8eef-b891f23932d7)

_Further work is planned to have the MCMC smaples flow through the integrations steps to produce an uncertainty band over the above life expectancies._

Convergance is fairly reasonable after applying NUTS for 5 000 samples, all taking Normal priors:

![sym_samples_2](https://github.com/patrickm663/bayesian-neural-networks/assets/77886027/a67f3847-d6fb-4ef2-8d78-e552efaec2ff)

## Bayesian Neural Networks to Enhance Existing Mortality Equations

Work is underway to test out whether Lee-Carter can be adapted to use BNNs to express the $\beta(x)$ and $\kappa(t)$ terms. These will be benchmarked against a Lee-Carter model fitted using SVD. The $\beta(x)$ and $\kappa(t)$ terms can then be analysed in terms of their gradients (what is the general movement as age and time change) and estimated symbolically (as the above equation).

$$\log\big( m\left( x, t \right) \big) = \alpha(x) + \beta(x)\kappa(t)$$

Gender can also be incorporated into the equation to extend the model.

## Bayesian Neural Networks to Capture Interesting Priors

In addition, we will look at how BNNs fitted on auxilarly-but-informative-data (life tables, mortality for neighbouring countries) can form a "foundation" of priors for Irish mortality example. For instance, have a BNN (or regular NN) first fit against a UK life table so we have a function $f$ that approximates mortality given age and gender, then use these fitted weights as priors for a BNN targetting Irish mortality -- rather than just $Normal(0, \sigma)$ priors. The hypothesis is that this could lead to better convergance of the BNN priors. This can be adapted to the Lee-Carter example above by e.g. having the $\beta(x)$ term "start" with some life table as its prior.

# Further Work

- The above is subject to change as the neural network re-evaulated and different sampling methods compared
- These approaches will also be evaluated over different mortality datasets (Ireland is being used at the moment).

# Data

The data is Irish mortality data from 1950-2020, sourced from https://mortality.org.
