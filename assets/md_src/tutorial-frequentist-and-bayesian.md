---
type: permanent-note
created: 2026-06-05
updated: 2026-06-20
derived_from: []
tags:
  - permanent/principles
---

# Frequentist & Bayesian Uncertainty

## Problem Setup: quantify uncertainty about an unknown parameter

Assume data $X_1,\dots,X_n\overset{\mathrm{iid}}{\sim} p_\theta(x), \theta\in \Theta.$ The parameter $\theta$ is unknown, and we want to estimate this unknown parameter. We observe a dataset: $x_{1:n}=(x_1,\dots,x_n).$ Our goal of uncertainty quantification is not merely to output an estimate $\hat{\theta}$, but also quantify how uncertain we are about $\theta$.

There are two classical languages to quantify uncertainty:

| Viewpoint   | Unknown object  | Probability is over              | Output                                    |
| ----------- | --------------- | -------------------------------- | ----------------------------------------- |
| Frequentist | fixed $\theta$  | random data $X_{1:n}$            | acceptance region, confidence interval    |
| Bayesian    | random $\theta$ | posterior $\theta \vert X_{1:n}$ | posterior distribution, credible interval |

---
## Frequentist Uncertainty: Coverage before seeing data (Pre-data calibration guarantee)

In the Frequentist viewpoint, we have a fixed, unknown $\theta\in \Theta$ and we have $X_{1:n}\sim P_\theta$. After sampling the data $X_{1:n}=x_{1:n}$, we use a pre-data-constructed measurable map: $C: \mathcal{X}^n\to 2^{\Theta}$ to get the frequentist confidence interval, such that 
$$
\mathbb{P}_\theta(\theta\in C(X_{1:n}))\ge 1-\alpha, \quad \forall\theta\in \Theta.
$$
This probability is over repeated sampling of the data.

The measurable map $C(\cdot)$ is constructed from the pre-data viewpoint. Our ultimate goal on the $C(\cdot)$ can be described as: for any fixed true parameter $\theta^*$, we want $\theta^*\in C(X_{1:n})$ to happen with probability at least $1-\alpha$ when data $X_{1:n}$ are generated from $P_\theta^*$, i.e.
$$
\mathbb{P}_\theta(X_{1:n}:\theta\in C(X_{1:n}))\ge 1-\alpha.
$$
Equivalently, define the **event**, called "acceptance region":
$$
A_\theta=\{x_{1:n}\in\mathcal{X}^n:\theta\in C(x_{1:n})\}
$$
Then the above inequality becomes:
$$
\mathbb{P}_\theta(A_\theta)\ge 1-\alpha.
$$

The above formalization enlightens the following confidence interval construction procedure:
1. For each $\theta\in\Theta$, construct a set $A_\theta\subseteq \mathcal{X}^n$ such that $\mathbb{P}_\theta(A_\theta)\ge 1-\alpha$.
2. We observe the dataset $x_{1:n}$.
3. Define the set $C(x_{1:n}):=\{\theta\in\Theta: x_{1:n}\in A_\theta\}$.
> [!NOTE]
> Simple proof of $C(x_{1:n})$ obey the definition of confidence interval:
> $$
> \mathbb{P}_\theta(\theta\in C(X_{1:n}))=\mathbb{P}_\theta(X_{1:n}\in A_\theta)\ge 1-\alpha.
> $$
> Therefore, $C(X_{1:n})$ is a valid $1-\alpha$ confidence set.

> [!IMPORTANT] 
> The frequentist does not say for a fixed sampled dataset $x_{1:n}$
> $$
> \mathbb{P}(\theta\in C(x_{1:n}))=1-\alpha.
> $$
> Because in the frequentist view, $\theta$ is fixed, unknown parameter. Once the dataset $x_{1:n}$ being sampled, the interval is also fixed. Thus, $\theta\in C(x_{1:n})$ is either True or False.
> 
> Instead, the fundamental meaning of the statement is:
> > The procedure of 1) sample the dataset $x_{1:n}$ from $p_\theta(x)$ 2) construct confidence interval $C(\cdot)$ has long-run coverage $1-\alpha$.

So frequentist UQ is a pre-data calibration guarantee.

---
## Bayesian Uncertainty: Posterior probability after seeing data
In the Bayesian viewpoint, we specify a joint probabilistic model over both the unknown parameter and the data. Let $\theta\sim\pi(\theta)$ and conditioned on $\theta$, we have $X_{1:n}\vert \theta \sim P_\theta$. Equivalently, we specify the joint distribution as $p(\theta,x_{1:n})=\pi(\theta)p_\theta(x_{1:n}).$ After sampling the data $X_{1:n}=x_{1:n}$, we update our uncertainty about $\theta$ using Bayes' rule, i.e. $\pi(\theta|x_{1:n})=\frac{p_\theta(x_{1:n})\pi(\theta)}{\int_\Theta p_\vartheta(x_{1:n})\pi(\vartheta)d\vartheta}.$ Define the corresponding probability measure $\Pi$ as: $\Pi(A\vert x_{1:n})=\int_A \pi(\theta\vert x_{1:n})d\theta$, for any measurable set $A\subseteq \Theta$.

Then, a Bayesian $1-\alpha$ credible set is a data-dependent set
$$
B(X_{1:n})\subseteq \Theta
$$
such that, after observing $x_{1:n}$,
$$
\Pi(B(x_{1:n})\vert x_{1:n})=1-\alpha.
$$
Equivalently, 
$$
\Pi(\theta\in B(x_{1:n})\vert X_{1:n}=x_{1:n})=1-\alpha.
$$
The probability is over the posterior distribution of $\theta$, conditional on the observed data.

> [!IMPORTANT]
> The fundamental meaning of the statement is:
> > Given a fixed $x_{1:n}$, the posterior mass assigned to $B(X_{1:n})$ is $1-\alpha$.



