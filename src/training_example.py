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
from density_matrix_nlp import reduced_density_bag # include model

""" Training Demo on BookCorpus """

# there may be a conflict when streaming, resolve with one of these two methods
# !pip install -U datasets
# !pip install fsspec==2023.9.2

from datasets import load_dataset

books = load_dataset("bookcorpus", split="train", streaming=True, download_mode="force_redownload")

import itertools
from itertools import islice
from pathlib import Path
from tqdm.auto import tqdm

def token_stream(text_iter, batch=1000, workers=2, max_lines=150000):
  for doc in nlp.pipe(tqdm(text_iter, total=max_lines), batch_size=batch, n_process=workers, disable=["parser"]):
    yield [t.text.lower() for t in doc if not t.is_space]

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
tokenizer = lambda s: [t.text.lower() for t in nlp(s) if not t.is_space]
MAX_LINES = 150000 # change to amount to be read

"""Stream Tokens and Make Counts. Execution time can be larger, usually dependent on the tokenizer used."""

model = reduced_density_bag(k=5, tokenizer=tokenizer, min_prefix_count=3, backend_storage="tt")

print("Counting prefixes/targets …")
books_iter = itertools.islice(books, MAX_LINES)
text_iter  = (rec["text"] for rec in books_iter)
tokens_list = token_stream(text_iter)
for tokens in tokens_list:
    model.update_from_tokens(tokens)

"""Training Execution"""

print("Beginning training on BookCorpus...")
model.fit()
print("Finished training on BookCorpus.")

print(f"dimension: {model.dim}")

# save model here for later use if desired
