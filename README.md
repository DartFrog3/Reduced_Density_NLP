# Reduced_Density_NLP
*Project implementing Tai-Danae Bradley's "Language Modeling with Reduced Densities" in pure Python*

---

## 1  What is Reduced_Density_NLP?

In a classical bag-of-words model every token contributes a single scalar count, losing almost all information about local word order or contextual ambiguity. However, this information can be retained with an analagoy to quantum physics, in density matrices. In the reduced density model (RD Model), for each vocabulary item $w$, form state vectors (to later produce a positive-semidefinite operator $\rho_w$, or a reduced density matrix) that live in the Hilbert space spanned by all $(k-1)$-gram prefixes within the corpus.  The spectrum of $\rho_w$ captures how many distinct, possible contexts exist for $w$.  Working with density operators therefore preserves the conditional probabilities of an $n$-gram model with the added benefit of more contexual understanding. Applying Tensor-Train compression, the model's storage requirements remain feasible and allows it to stream text over large corpora.

---

## 2  Implementation Details

Begin by fixing a prefix length $k-1$.  Let $C(p,w)$ be the number of times prefix $p$ is followed by token $w$ in the corpus, and let $\alpha>0$ be an $\text{add} - \alpha$ smoothing constant. As in classical bag-of-words models, counts of prefixes and suffixes correspond to probabilities which can then map to probability amplitudes by defining a ket

$$
\psi_w[p] \\ = \\ \sqrt{C(p,w)+\alpha},
$$

in keeping with the quantum analogy. So have $\psi_w \in \mathbb{R}^{N_{\text{pre}}}$ where $N_{\text{pre}}$ is the number of retained prefixes. The outer product

$$
\rho_w \\ = \\ \psi_w \psi_w^\top
$$

is then a rank-1 reduced density matrix for suffix $w$. Document statistics are obtained by summing these operators over the tokenized text: $\rho_{\text{doc}} = \sum_{w\in\text{doc}} \rho_w$. Because $N_{\text{pre}}$ grows roughly like $\sigma^{k-1}$ for vocabulary size $\sigma$, each $\psi_w$ is reshaped into a $(k-1)$-way tensor and compressed with a Tensor-Train decomposition of maximum internal rank $r$. In doing so, this reduces storage scaling from $O(\sigma^{k-1})$ to $O\bigl((k-1)r^{2}\sigma\bigr)$ and preserves state vectors to a reasonable degree of approximation.

---

## 3  Persisting Issues/TODO:

1. Evaluation/use still requires encoding to sparse vectors, which has been causing memory overload errors. Still need to find a fix for this.
2. Train+Evaluate on larger corpora
3. Include function to check Loewner order/textual well-ordering. 
4. Make training/eval executables with args

---

## 4  Repository Contains:

This repo contains the class definition of Reduced_Density_NLP, a training example on BookCorpus, and evaluation against classical Bag of Words through training of identical logistic regression classifiers.

---

## 5  Testing Results:

As an initial test, a 50,000 document subset of BookCorpus was used. The RD Model used a minimum_prefix_count=3 (with default inputs) and was incorporated into a logistic regression classifiers. TF-IDF was then used to train an identical logistic regression classifier and the performance between the two is as follows.

| Model          | Accuracy | F1-score |
|----------------|:--------:|:--------:|
| TF-IDF         |  0.8771  |  0.8783  |
| RD Model       |  0.8288  |  0.8300  |

The RD Model performs worse, though this is not too surprising, given the small corpus. The context-aware nature of the RD Model should thrive more within larger corpora. 

---

## 6  License

This project is licensed under the MIT License – see `LICENSE` for details.

