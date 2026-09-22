Suppose a pretrained model has already turned each input into a collection of numbers. We call these numbers features and use them to predict a target. A natural question follows: which features are actually useful?

A simple starting point is to freeze the pretrained model and train a **one-layer linear model** on its features. This is called **linear probing**.

If the predictions are good, the features collectively support the task. But we usually want to go further: since every feature has a weight, can we interpret that weight as its importance?

## 1. The prediction can be clear while the contributions remain ambiguous

Different features may carry overlapping information. One feature might even be a linear combination of others. In that case, increasing its weight and adjusting the other weights can leave the predictions unchanged.

Let $A\in\mathbb{R}^{n\times d}$ contain $n$ examples (rows) and $d$ features (columns). A linear probe predicts $\hat y=Aw$. Its possible predictions form the **column space** $\operatorname{col}(A)$, whose dimension is $\operatorname{rank}(A)$.

When $\operatorname{rank}(A)<d$, the columns are linearly dependent: there is a nonzero $v$ with $Av=0$, so

$$
A(w+v)=Aw.
$$

The weights change, but the predictions stay the same. Fitting a prediction within $\operatorname{col}(A)$ therefore need not uniquely determine the feature weights.

A large weight in one fitted model does not establish that a feature matters more than others carrying the same information. Even without linear dependence, changing a feature's numerical scale changes the size of its corresponding weight.

This suggests another approach: instead of inspecting the weight, why not remove the feature and see whether predictions get worse?

## 2. Does no effect from removal mean no value?

The procedure is straightforward: remove a feature, retrain on the remaining features, and compare predictive performance. Retraining matters because we want to know whether the remaining features can take over its role.

In the linear fitting setting above, removing a column that can be expressed using the others leaves the column space unchanged. The remaining features can still produce the original predictions, so the best fitting error does not increase.

Useful information has not disappeared. Other features are still supplying it.

More generally, a group of features may share useful predictive information. Removing any one of them leaves enough behind, while removing the whole group may hurt performance. Every individual removal can have no effect even though the group is collectively useful.

**A removal experiment measures the extra value of a feature when the other features remain available.** That is a different question from whether the feature contains useful information at all.

If we care about the value of shared information, we should examine the group. But even after establishing that the group matters, another question remains: how should its shared value be assigned to individual features?

## 3. Can mutual information measure useful information directly?

If overlapping information is the difficulty, we might set aside model weights and ask a more direct question: does knowing this feature tell us more about the target?

**Mutual information** measures this reduction in uncertainty. Written as $I(X_j;Y)$, it describes how much observing feature $X_j$ alone tells us about the target $Y$.

But features can share information or provide it only together:

- **Redundancy:** If two features carry the same information about the target, either one can help us predict it. But after observing one, observing the other tells us nothing new. Adding their individual mutual information scores would count the same information twice. The second feature adds no value because its information is already available, even though that information is useful.
- **Synergy:** A feature can become informative only when another feature is known. In XOR, two independent inputs are each equally likely to be 0 or 1, and the target is whether they differ. Seeing either input alone leaves us with a fifty-fifty guess; seeing both gives the answer. Each feature therefore has zero mutual information on its own, even though the pair fully determines the target. Scoring them separately would miss the information carried by their combination.

**Conditional mutual information**, $I(X_j;Y\mid X_{-j})$, measures what feature $X_j$ adds once all the other features, $X_{-j}$, are known. In the redundancy case, it is zero: the other feature has already supplied the information. In the XOR case, it is positive: knowing the other input makes this input useful. It can therefore include information gained by combining features, rather than only information a feature provides on its own.

The same feature can therefore receive different scores depending on which other features we treat as already known.

## 4. What if we account for different combinations?

A feature's contribution now depends on which other features accompany it. Could we consider different combinations and average those contributions?

The **Shapley value** offers one such allocation rule. Imagine randomly ordering the features and adding them to a predictor one at a time. Record the extra predictive value each feature provides when it joins, then average over orderings. [SAGE](https://arxiv.org/abs/2004.00668) applies this idea to global predictive contributions.

A duplicate is no longer assigned zero just because its copy happens to be present. In some orderings it arrives first and provides information; in others it arrives later and adds nothing. Symmetric copies receive equal allocations.

This gives a definite answer to how much credit each feature receives under the chosen rule. It does not turn shared information into something each feature independently owns, and the resulting scores do not fully describe redundancy and synergy.

Computation also becomes difficult. With $d$ features, there are $2^d$ subsets. Without additional structure, examining every combination is generally impractical. The difficulty comes from tracking each feature's role across combinations; merely comparing the complete feature set with a baseline does not require this enumeration.

In practice, people reduce this computation through sampling or structural assumptions:

- **Monte Carlo Shapley:** Sample random feature orderings and average each feature's contribution across them, as in [permutation sampling](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html). This estimates Shapley values without evaluating all $2^d$ subsets; more samples reduce sampling noise.
- **Limit interaction order:** Approximate the value of a feature set using individual effects and pairwise interactions, assuming higher-order interactions are negligible. This reduces the combinations we need to model, but can miss information that appears only in larger groups.
- **Group features:** Treat strongly redundant features as a single unit and measure their joint importance. [Grouped importance](https://arxiv.org/abs/2104.11688) reduces the number of units to consider and captures their shared value without having to divide it among individual features.
- **Model-based importance:** Exploit a model's structure, using methods such as [Tree SHAP](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) or specialized calculations for [linear models](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html). The available shortcuts depend on the model, assumptions about feature dependence, and whether we are explaining a prediction or measuring predictive performance.
- **Conditional independence assumptions:** Assume the relevant dependencies can be captured within small neighborhoods of features, allowing the analysis to focus on those neighborhoods. Methods such as [L-Shapley](https://arxiv.org/abs/1808.02610) use this kind of structure; a sparse dependency graph alone does not guarantee that distant features can be ignored.

Sampling estimates the chosen score with less computation. Grouping changes what we assign importance to, while structural assumptions simplify the relationships we consider. The practical choice therefore depends on which question we want to answer and which assumptions we can justify.

## 5. The question to settle first: important in what sense?

We began by looking for a more reliable score than a model weight. Along the way, several apparently similar questions turned out to be different:

- **Does the feature contain useful information?** This concerns its relationship with the target.
- **What does it add when the other features are available?** This concerns its contribution given existing information.
- **What does a group provide collectively?** This includes shared or synergistic value that individual removals can obscure.
- **How should total value be assigned to individual features?** This requires an explicit allocation rule.

The right question depends on the purpose of the analysis. To simplify a predictor's inputs, performance after removal and retraining may be what matters. To understand what a representation contains, shared information cannot be dismissed simply because removing one feature has no effect.

Rather than reporting that a feature is unimportant, we can make a more precise statement: with the other features available and retraining allowed, removing it did not reduce performance on this task. That tells us what was measured and gives the next question somewhere concrete to start: can other features replace this information? What happens if the whole group is removed? Would the conclusion hold for another task?
