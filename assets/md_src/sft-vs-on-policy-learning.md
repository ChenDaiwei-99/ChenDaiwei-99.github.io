# SFT vs. On-Policy Learning: Coverage, Capacity, and Reward

## The difference starts with where training samples come from

Suppose a task has several valid but well-separated solutions, represented by a target distribution $p(x)$, while a student represents a distribution $q_\theta(x)$.

Supervised fine-tuning (SFT) trains on examples sampled from the target:
$$
\max_\theta \mathbb{E}_{x\sim p}\left[\log q_\theta(x)\right].
$$
Equivalently, this minimizes the forward KL divergence $D_{\mathrm{KL}}(p\Vert q_\theta)$. Every target mode continues to supply examples, so a valid region that the student misses still applies optimization pressure. This creates a **mode-covering** tendency.

On-policy learning instead trains on samples from the current student:
$$
\nabla_\theta \mathbb{E}_{x\sim q_\theta}\left[r(x)\right]
=
\mathbb{E}_{x\sim q_\theta}\left[r(x)\nabla_\theta\log q_\theta(x)\right].
$$
The student receives signal where it already samples. A reachable success is reinforced, which makes similar future successes more likely. Valid modes that the student never visits provide no direct reward gradient. This feedback can create a **mode-seeking** tendency.

> [!IMPORTANT]
> On-policy policy-gradient learning is not literally reverse-KL minimization. The shared intuition comes from student-weighted sampling: optimization pressure is concentrated where the current student already places probability.

## Why limited student capacity changes the outcome

Mode coverage is not always useful. If a low-capacity student cannot represent all separated target modes, SFT must compromise. It may broaden or average its limited components to assign some probability to every demonstrated solution, while placing much of its probability between the valid regions.

On-policy learning can use the same limited capacity differently. If the task rewards **any** valid solution, the student can specialize in one reachable mode and make that answer highly reliable. It gives up distributional coverage but can achieve higher task reward.

| Setting | SFT tendency | On-policy tendency |
| --- | --- | --- |
| Training samples | Drawn from the target or dataset | Drawn from the current student |
| Optimization pressure | Remains on every demonstrated mode | Concentrates on visited successes |
| Low-capacity behavior | Spreads capacity across incompatible modes | Specializes in reachable modes |
| Best fit | Reproduce the full target distribution | Produce at least one high-reward answer |

This is not a universal advantage for on-policy learning. Specialization only helps when one good solution is sufficient, reward accurately identifies success, and the initial policy can reach a successful region. With poor initialization, sparse reward may provide no useful signal. When the student has enough capacity to represent every mode, SFT can cover them without the same approximation compromise.

## Interactive toy experiment

[One Good Mode](https://one-good-mode.daiwei-chen.chatgpt.site/) makes the mechanism concrete with eight valid Gaussian modes and a student whose capacity can be varied. The interactive comparison shows SFT following examples from all eight targets while on-policy learning reinforces successful samples from modes the student can already reach.

The central lesson is:

> When the goal is one reliable solution rather than full distribution matching, mode-seeking can be an efficient use of limited model capacity. When coverage itself matters, that same specialization is a failure.
