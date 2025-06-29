# Reduced_Density_NLP
*Toy project implementing Tai-Danae Bradley's "Language Modeling with Reduced Densities" in pure Python*

---

## 1  What is Reduced_Density_NLP?

The RD model is akin to a classical bag-of-words model in that in replaces every **term‑frequency scalar** with a **positive‑semi‑definite density matrix** that preserves  word‑order information (in close proximity). Applying Tensor-Train compression, the model's storage requirements remains feasible and can stream text over large corpora.

---

## 2  Implementation Details

As in classical Bag of Words, counts of prefixes and suffixes correspond to probabilities. These can be mapped to probability amplitudes to assemble quantum state vectors. Taking the outer product will then yield a density matrix. 

```math
\begin{aligned}
&\text{Joint counts:} &C(u,v) &= \sum_{i\in\text{corpus}} 1[\text{prefix}_i = u,\;\text{token}_i = v]\\[4pt]
&\text{State vector (with smoothing):} &\varphi_v[u] &= \sqrt{C(u,v)+\alpha}\;\; (u:\text{prefix})\\[4pt]
&\text{Reduced density:} &D_v &= \varphi_v\varphi_v^{\top}\in\mathbb R^{N_{\text{pre}}\times N_{\text{pre}}}\\[4pt]
\end{aligned}
```
Applying TT compression allows the storage to scale as ```math\mathcal O((k-1)\,r^2\sigma)``` rather than ```math\mathcal O(\sigma^(k-1))```

---

## 3  Repository Contains:

This repo contains the class definition of Reduced_Density_NLP, a training example on BookCorpus, and evaluation against classical Bag of Words through training of identical logistic regression classifiers.

&\text{Loewner order:} &A\preceq B &\iff B-A \text{ is p.s.d.}

---

## 4  License

This project is licensed under the MIT License – see `LICENSE` for details.

