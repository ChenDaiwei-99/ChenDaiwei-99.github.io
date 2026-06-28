---
type: permanent-note
created: 2026-02-02
updated: 2026-06-20
derived_from: []
tags:
  - permanent/principles
source_refs:
  - "[[knowledge-base/raw/pdfs/An Introduction to Variational Autoencoders]]"
---

## Intro to Automatic Differentiation (AD)
Define a forward parametric computation, in the form of a directed acyclic graph, that computes the desired objective. If the components of the graph are differentiable, then a backwards computation for the gradient of the objective can be derived automatically with the chain rule.

> [!important] Stochastic Node of a computation graph: 
> It is a random variable whose distribution depends in some deterministic way on the values of the parent nodes.

### Two types of discrete node
1. Deterministic discreteness can be relaxed and approximated reasonably well with sigmoidal functions or the softmax.
2. Stochastic discreteness poses a challenge for AD libraries, i.e. a scenario where a distribution over discrete states is needed.

---
## Intro to Optimizing Stochastic Computation Graphs (SCGs)
The state of each non-input node in such a graph is obtained from the states of its parent nodes by either evaluating a deterministic function or sampling from a conditional distribution.

Let's take a simple SCG as an example to explain the challenge behind the backpropagation with stochastic nodes: one single stochastic node in the graph

1. sample $X$ from the conditional distribution $p_\phi(x)$ of the stochastic node given its parents.
2. evaluate a deterministic function $f_\theta(x)$ at $X$

We can think of $f_\theta(X)$ as a noisy objective, and we are interested in optimizing its expected value $L(\theta,\phi)=\mathbb{E}_{X\sim p_\phi(x)}[f_\theta(x)]$ w.r.t parameters $\theta,\phi$.

In general, both the objective and its gradients are intractable.
> [!tip] Reason: the expectation objective requires integral over $X$, which usually has no closed form for complex neural nets.

Let's write down the gradients for $\theta$ and $\phi$ separately.

The gradient w.r.t. the parameter $\theta$ has the form
$$\nabla_\theta L(\theta,\phi)=\nabla \mathbb{E}_{X\sim p_\phi(x)}[f_\theta(X)]=\mathbb{E}_{X\sim p_\phi(x)}[\nabla_\theta f_\theta(X)]$$
Hence, we can easily estimate the gradient via Naive Monte Carlo sampling.

The gradient w.r.t the parameters $\phi$ has the form
$$\nabla_\phi L(\theta,\phi)=\nabla_\phi \int p_\phi(x)f_\theta(x)dx=\int f_\theta(x)\nabla_\phi p_\phi(x)dx$$
doesn't have the form of an expectation w.r.t $x$, and thus does not directly lead to a Monte Carlo gradient estimator.
> [!warning] That's why calculating the gradients w.r.t the stochastic node is challenging

---
## Two existing methods to bypass the issue

### 01. Score Function Estimator
By the identity $\nabla_\phi p_\phi(x)=p_\phi(x)\nabla\phi \log p_\phi(x)$, the gradient w.r.t. $\phi$ can be written as:
$$\nabla_\phi L(\theta, \phi)=\mathbb{E}_{X\sim p_\phi(x)}[f_\theta(X)\nabla_\phi \log p_\phi(X)]$$
which again allows easy estimation via Naive Monte Carlo sampling.
> Though this basic version can suffer from high variance, various variance reduction techniques can be used to make the estimator more effective.
### 02. Reparameterization Trick
In many cases, we can reshape the sampling from $p_\phi(x)$ by first sampling $Z$ from some fixed distribution $\epsilon(z)$ and then transforming the sample using some function $g_\phi(z)$. Such refactorization allows us to transfer the dependence on $\phi$ from the distribution $p$ into the deterministic function $f$ by writing $f_\theta(x)=f_\theta(g_\phi(z))$ for $x=g_\phi(z)$, 

>[!tip] Estimate the gradient w.r.t. params of a distribution (Intractable) => Estimate the gradient w.r.t. params of a deterministic function (Simple).

Formally, with reparameterization $g_\phi(z), z\sim\epsilon(z)$, we can now express the objective as an expectation w.r.t. $\epsilon(z)$:$$L(\theta,\phi)=\mathbb{E}_{X\sim p_\phi(x)}[f_\theta(X)]=\mathbb{E}_{Z\sim\epsilon(z)}[f_\theta(g_\phi(Z))]$$ As $\epsilon(\cdot)$ doesn't depend on $\phi$, we can estimate the gradient w.r.t $\phi$ in the same way we estimated the gradient w.r.t $\theta$.

---
