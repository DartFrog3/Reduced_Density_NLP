import math
import numpy as np
from collections import defaultdict, Counter
from typing import Callable, List, Dict, Sequence, Tuple, Optional
from __future__ import annotations
import matplotlib.pyplot as plt
import spacy
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor
from density_matrix_nlp import reduced_density_bag
import sklearn
from pathlib import Path
import itertools, joblib, gc, os, re
import scipy.sparse as sp
from tqdm.auto import tqdm
from datasets import load_dataset
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

""" Evaluation Versus BOW Model """

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
tokenizer = lambda s: [t.text.lower() for t in nlp(s) if not t.is_space]
MAX_LINES = 100000

def token_stream(text_iter, batch=1_000, workers=2):
    for doc in nlp.pipe(text_iter, batch_size=batch, n_process=workers, disable=["parser"]):
        yield [t.text.lower() for t in doc if not t.is_space]

books = load_dataset("bookcorpus", split="train", streaming=True, download_mode="force_redownload")

raw_corpus_iter = (rec["text"] for rec in books)
raw_lines = list(itertools.islice(raw_corpus_iter, MAX_LINES))

# nlp tokenizer is slow. For initial eval use a simpler fast regex tokenizer
fast_tokenizer = lambda s: re.findall(r'\b\w+\b', s.lower())

def fast_token_stream(text_iter, max_lines=MAX_LINES):
  iterable = text_iter
  if max_lines is not None:
    iterable = tqdm(text_iter, total=max_lines)
  for text in iterable:
    yield fast_tokenizer(text)

"""Train BOW model"""

# BOW model
tfidf = TfidfVectorizer(
    tokenizer=fast_tokenizer,       # identical pre-processing so use fast_tokenizer for now
    lowercase=False,
    min_df=3,
    max_features=MAX_LINES # upper bound here?
)

print("Fitting TF-IDF …")
tfidf.fit(tqdm(raw_lines, desc="TFIDF-fit"))

print("TF-IDF vocab size:", len(tfidf.vocabulary_))

# save model for later
#joblib.dump(tfidf, "tfidf_model.joblib", compress=3)

"""Train Reduced Density Model"""

model = reduced_density_bag(k=5, tokenizer=fast_tokenizer, min_prefix_count=2, max_prefix_count=50000, backend_storage="tt")

print("Counting prefixes/targets …")
for tokens in fast_token_stream(raw_lines):
  model.update_from_tokens(tokens)

print("Beginning training on BookCorpus...")
model.fit()
print("Finished training on BookCorpus.")
print(f"dimension: {model.dim}")

# save model for later
#joblib.dump(model, "/content/rdbow_tt_bookcorpus.joblib", compress=3)

# encode as sparse vectors
def encode_nohash(text: str) -> sp.csr_matrix:
    """Return a 1×dim CSR row with exactly `model.dim` columns."""
    vecs = model.document_state_vectors(text)
    if not vecs:
        return sp.csr_matrix((1, model.dim))
    agg = np.sum(vecs, axis=0)
    nz  = agg.nonzero()[0]
    return sp.csr_matrix((agg[nz], ([0]*len(nz), nz)),
                         shape=(1, model.dim))

"""Evaluation Dataset"""

imdb = load_dataset("imdb")
texts = imdb["train"]["text"] + imdb["test"]["text"]
labels = imdb["train"]["label"] + imdb["test"]["label"]

X_raw_train, X_raw_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

print("Vectorising TF-IDF …")
X_bow_train = tfidf.transform(tqdm(X_raw_train))
X_bow_test  = tfidf.transform(tqdm(X_raw_test))

print("Vectorising RD Model …")
X_rd_train = sp.vstack([encode_nohash(t) for t in tqdm(X_raw_train)])
X_rd_test  = sp.vstack([encode_nohash(t) for t in tqdm(X_raw_test)])

clf_bow  = LogisticRegression(max_iter=500, n_jobs=-1, random_state=42)
print("Training / scoring TF-IDF …")
clf_bow.fit(X_bow_train, y_train)
pred_bow  = clf_bow.predict(X_bow_test)
acc_bow   = accuracy_score(y_test, pred_bow)
f1_bow    = f1_score(y_test, pred_bow)

clf_rd = LogisticRegression(solver='saga', max_iter=500, n_jobs=-1, random_state=42)
print("Training / scoring RD Model …")
clf_rd.fit(X_rd_train, y_train)
pred_rd = clf_rd.predict(X_rd_test)
acc_rd = accuracy_score(y_test, pred_rd)
f1_rd = f1_score(y_test, pred_rd)

print(f"\nTF-IDF  acc={acc_bow:.4f} | F1={f1_bow:.4f}")
print(f"RD Model acc={acc_rd:.4f} | F1={f1_rd:.4f}")

