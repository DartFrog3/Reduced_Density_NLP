# Reduced_Density_NLP
*Toy project implementing Tai-Danae Bradley's "Language Modeling with Reduced Densities" in pure Python*

---

## 1  What is Reduced_Density_NLP?

The RD model is akin to a classical bag-of-words model in that in replaces every **term‑frequency scalar** with a **positive‑semi‑definite density matrix** that preserves  word‑order information (in close proximity). Applying Tensor-Train compression, the model's storage requirements remain feasible and allows it to stream text over large corpora.

---

## 2  Implementation Details

As in classical Bag of Words, counts of prefixes and suffixes correspond to probabilities. These can be mapped to probability amplitudes to assemble quantum state vectors. Taking the outer product will then yield a density matrix. 

```math
\begin{aligned}
&\text{Joint counts:} &C(u,v) &= \sum_{i\in\text{corpus}} 1[\text{prefix}_i = u,\;\text{token}_i = v]\\[4pt]
&\text{State vector (with smoothing):} &\psi_v[u] &= \sqrt{C(u,v)+\alpha}\;\; (u:\text{prefix})\\[4pt]
&\text{Reduced density:} &\rho_v &= \psi_v\psi_v^{\top}\in\mathbb R^{N_{\text{pre}}\times N_{\text{pre}}}\\[4pt]
\end{aligned}
```
Applying TT compression allows the storage to scale as 
```math
\mathcal O((k-1)\,r^2\sigma) \text{ rather than } \mathcal O(\sigma^{(k-1)})
```
which is essential for actual, practical use.

---

## 4  Persisting Issues/TODO:

1. Expand on theory in readme
2. Evaluation/use still requires encoding to sparse vectors, which has been causing memory overload errors. Still need to find a fix for this.
3. Train+Evaluate on larger corpora
4. Include function to check Loewner order/textual well-ordering. 
5. Make training/eval executables with args

---

## 5  Repository Contains:

This repo contains the class definition of Reduced_Density_NLP, a training example on BookCorpus, and evaluation against classical Bag of Words through training of identical logistic regression classifiers.

---

## 6  Testing Results:

As an initial test, a 50,000 document subset of BookCorpus was used. The RD Model used a minimum_prefix_count=3 (with default inputs) and was incorporated into a logistic regression classifiers. TF-IDF was then used to train an identical logistic regression classifier and the performance between the two is as follows.

| Model          | Accuracy | F1-score |
|----------------|:--------:|:--------:|
| TF-IDF         |  0.8771  |  0.8783  |
| RD Model       |  0.8288  |  0.8300  |

The RD Model performs worse, though this is not too surprising, given the small corpus. The context-aware nature of the RD Model should thrive more within larger corpora. 

---

## 7  License

This project is licensed under the MIT License – see `LICENSE` for details.

