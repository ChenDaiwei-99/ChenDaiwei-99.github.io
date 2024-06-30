---
layout: post
title: Gumbel Softmax
date: 2024-06-26
description: Gumbel softmax provides us a reliable method to learn categorical variables in PyTorch.
tags: Deep-Learning PyTorch Code
featured: true
tabs: true
---

## Categorical Reparameterization with Gumbel-Softmax (create categorical variables in neural networks)

- Background:
    - **Discrete variables** are important in neural networks. E.g. discrete variables have been used to learn probabilistic latent representations that correspond to distinct semantic classes.
- Contributions
    - **Gumbel-Softmax**, a continuous distribution on the simplex that can approximate categorical samples.
    - this paper provides a simple, differentiable approximate sampling mechanism for categorical variables that can be integrated into neural networks and trained using standard back-propagation.

### The Gumbel-Softmax Distribution

{% include figure.liquid path="assets/img/blog-gumbel-softmax/fig1.png" class="img-fluid" %}

{% include figure.liquid path="assets/img/blog-gumbel-softmax/equations.png" class="img-fluid" %}

For a more comprehensive explanation of why the Gumbel-Softmax distribution can approximate the categorical distribution, I would like to refer to <a href="https://homes.cs.washington.edu/~ewein//blog/2022/03/04/gumbel-max/">this website</a>.

### Gumble Softmax Estimator

> For learning ,there is a tradeoff between small temperatures, where samples are close to one-hot but the variance of the gradients is large, and large temperatures, where samples are smooth but the variance of the gradient is small. 

In practice, we start at a high temperature and anneal to a small but non-zero temperature.
>  **Incorporates Noise**: Gumbel-Softmax incorporates Gumbel noise into the input, which allows the model to explore a variety of outputs, making it stochastic as opposed to the deterministic nature of Softmax.