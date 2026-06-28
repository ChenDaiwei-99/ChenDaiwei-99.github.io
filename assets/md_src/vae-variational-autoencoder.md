# VAE: Variational Autoencoder
## 01. Mixture Modeling Design
### 1. The Expressiveness–Tractability Tradeoff in Direct Density Modeling
The goal of generative modeling is to learn a parametric distribution $p_\theta(x)$ over observed data $x \in \mathbb{R}^D$ by maximizing the log-likelihood:
$$\max_\theta \sum_i \log p_\theta(x_i)$$
A fundamental tension arises:
- If $p_\theta(x)$ belongs to a **simple parametric family** (e.g., a multivariate Gaussian), density evaluation and MLE are tractable, but the family is insufficiently expressive for complex, multimodal data distributions.
- If $p_\theta(x)$ is specified via a **flexible neural parameterization** — e.g., an energy-based model $p_\theta(x) = Z(\theta)^{-1} e^{-E_\theta(x)}$ where $E_\theta$ is an arbitrary neural network — the model is expressive, but the normalizing constant $Z(\theta) = \int e^{-E_\theta(x)} dx$ is intractable, making likelihood evaluation and gradient-based MLE infeasible in general.
### 2. Latent Variable Models as Infinite Mixtures of Simple Densities
The latent variable approach resolves this tradeoff by introducing a latent variable $z \in \mathbb{R}^d$ and defining:
$$p_\theta(x) = \int p(z) \, p_\theta(x \mid z) \, dz$$
where $p(z)$ is a fixed prior (typically $\mathcal{N}(0, I)$) and $p_\theta(x \mid z)$ is a conditionally simple density — e.g., $\mathcal{N}(x; \mu_\theta(z), \sigma^2 I)$ — whose parameters are a nonlinear function of $z$ via a neural network (the *decoder*).

For any fixed $z$, $p_\theta(x \mid z)$ is a unimodal Gaussian — individually inexpressive. But the marginal $p_\theta(x)$ is a **continuous, infinite mixture** of such components, where different values of $z$ place Gaussian bumps at different locations in $x$-space determined by $\mu_\theta(z)$. The expressiveness of the marginal arises from the richness of this mixture, not from any single component.
### 3. A More Useful Form of Intractability
This construction trades an intractable normalizing constant for an intractable marginalization integral. However, the latter is a strictly more useful form of intractability:
- **Ancestral sampling is tractable:** draw $z \sim p(z)$, then $x \sim p_\theta(x \mid z)$. Generation never requires evaluating $p_\theta(x)$.
- **A differentiable lower bound exists:** the Evidence Lower Bound (ELBO) provides a tractable optimization objective without computing the marginal exactly.
## 02. MLE Objective & Posterior Inference
Given the above mixture modeling design, our MLE objective can be written as: $$\max_\theta\sum_{i=1}^n\log p_\theta(x_i)\Rightarrow \max_\theta \sum_{i=1}^n\int p(z)p_\theta(x|z)dz$$Let's take a closer look at the gradient:$$\nabla_\theta \log p_\theta(x)=\nabla_\theta \log \int p(z)p_\theta(x|z)dz$$
The challenge is how to compute the gradient through the integral term:
By chain rule: $$\nabla_\theta \log p_\theta(x)=\frac{1}{p_\theta(x)}\nabla_\theta p_\theta(x)$$
Hence, $$\nabla_\theta \log p_\theta(x)=\frac{1}{p_\theta(x)}\nabla_\theta\int p(z) \, p_\theta(x \mid z) \, dz$$
Then, $$\nabla_\theta \log p_\theta(x)=\frac{1}{p_\theta(x)}\int p(z)\nabla_\theta p_\theta(x|z)dz$$
Substituting the identity: $\nabla_\theta p_\theta(x|z)=p_\theta(x|z)\nabla_\theta\log_\theta p(x|z)$,$$\nabla_\theta \log p_\theta(x)=\frac{1}{p_\theta(x)}\int p(z)p_\theta(x|z)\nabla_\theta \log_\theta p_\theta(x|z)dz=\int \frac{p(z)p_\theta(x|z)}{p_\theta(x)}\nabla_\theta p_\theta (x|z)dz$$Which leads to the expectation expression:$$\nabla_\theta \log p_\theta(x)=\mathbb{E}_{z\sim p_\theta(z|x)}[\nabla_\theta \log p_\theta(x|z)]$$
> [!important] Although the mixture modeling design only requires $p(z)$ and $p_\theta(x|z)$, in learning procedure, we naturally require the posterior inference $p(z|x)$, i.e. the "encoder".

---
## 03. Challenge in Learning $\theta$

### 1. Background about Variational Inference
> [!tip] 
> 1. "Variational" comes from the "Calculus of Variations" --- 在微积分里，我们做优化求一个变量使得函数值最小；在Variational Inference里，我们做优化求一个函数使得泛函值最小 
> 2. "Inference" 特指 "Bayesian Inference" --- 根据我们观察到的$X$，去反推产生这些数据的隐藏变量，这个分布被称为后验分布，记作$p(Z|X)$

> Variational Inference is a technique for approximating an intractable probability distribution by searching over a tractable family of distributions to find the closest match. Suppose we have some target distribution $p(z|x)$ that we cannot compute in closed form --- typically because it involves the intractable marginal $p(x)=\int p(z)p(x|z)dz$ in the denominator via Bayes' rule: $$p(z|x)=\frac{p(z)p(x|z)}{p(x)}$$The key idea behind Variation Inference is: instead of computing $p(z|x)$ exactly, let's define a parametric family $\mathcal{Q}=\{q_\phi(z|x)\}$ of tractable distributions, and find the member that is closest to the true posterior by minimizing the KL divergence:$$\min_{\phi} D_{KL}(q_\phi(z|x)\|p_\theta(z|x))$$This turns an inference problem into an optimization problem.
### 2. Approximating $p_\theta(z|x)$ with a chosen parametric family $\mathcal{Q}=\{q_\phi(z|x)\}$ 
From the above derivation about the gradient: $$\nabla_\theta \log p_\theta(x)=\mathbb{E}_{z\sim p_\theta(z|x)}[\nabla_\theta \log p_\theta(x|z)]$$We need samples from $p_\theta(z|x)$, but can't compute the closed form, the obvious next step is learn a parameterized model $q_\phi(z|x)$ that approximates it. (That's where the "Variational Inference" comes up)
### 3. Now we need to learn both $\theta$ and $\phi$, what objective should we use?
The two sets of parameters has a circular dependency:
1. To learn $\theta$, we need good samples of $z$ for each observation $x$, i.e. the above expectation expression.
2. To learn $\phi$, we want $q_\phi(z|x)\approx p_\theta(z|x)$, but we can't get the closed form expression, need learning.

We need a single, joint objective that simultaneously:
1. pushes $p_\theta$ to original MLE objective
2. pushes $q_\phi$ close to the true posterior

Let's start from the MLE objective.

We have the log-likelihood $\log p_\theta(x)$ and an arbitrary distribution $q_\phi(z \mid x)$ over the latent variable $z$.
**Step 1.** Since $\log p_\theta(x)$ does not depend on $z$, it passes through any expectation over $z$ as a constant:
$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)}\big[\log p_\theta(x)\big]$$
**Step 2.** Apply Bayes' rule. From $p_\theta(x, z) = p_\theta(x) \cdot p_\theta(z \mid x)$, we have $p_\theta(x) = \frac{p_\theta(x, z)}{p_\theta(z \mid x)}$. Substitute:
$$= \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p_\theta(x, z)}{p_\theta(z \mid x)}\right]$$
**Step 3.** Insert $1 = \frac{q_\phi(z \mid x)}{q_\phi(z \mid x)}$ inside the logarithm:
$$= \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \cdot \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)}\right]$$
**Step 4.** Split the log of the product into a sum of logs:
$$= \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right] + \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)}\right]$$
**Step 5.** Recognize the two terms:
$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right]}_{\mathcal{L}(\theta,, \phi;, x) ;\text{(ELBO)}} + \underbrace{D_{KL}\big(q_\phi(z \mid x) ;|; p_\theta(z \mid x)\big)}_{\geq, 0}$$**Result.** Since $D_{KL} \geq 0$, we have the bound:
$$\mathcal{L}(\theta, \phi;, x) \leq \log p_\theta(x)$$
with equality if and only if $q_\phi(z \mid x) = p_\theta(z \mid x)$ almost everywhere.


## Reference
[Variational Autoencoders](https://arxiv.org/pdf/1906.02691)

