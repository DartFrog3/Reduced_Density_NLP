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

class reduced_density_bag:
  """
  Implementation of BoW-esque model using reduced density matrices.

  Params:
    k - length of n-gram with k-1 length prefix [ int ]
    tokenizer - any tokenizer model to map strings to tokens [ func ]
    alpha - smoothing param [ float ]
    min_prefix_count - discard prefixes occuring fewer than this amount of times [ int ]
    max_prefix_count - maximum number of distinct prefixes (if storage contraints) [ int ]
    backend_storage - option of "ket' or "tt" where each reduced density is stored as a ket pre outer product and
                      tt uses tensor-train decomposition for larger n-gram sizes [ str ]
    tt_rank - rank of tt decomposition for larger n-gram sizes which trades accuracy for storage [ int ]
  """

  def __init__(
      self,
      k: int = 3, # default to 3
      tokenizer: Optional[Callable[[str], List[str]]] = None,
      alpha: float = 0.05,
      min_prefix_count: int = 2,
      max_prefix_count: int = 100000, # depends on available RAM
      backend_storage: str = "ket",
      tt_rank: int = 8,
  ):
    # check validity
    if k < 2:
      raise ValueError("k must be at least 2")
    if backend_storage not in ["ket", "tt"]:
      raise ValueError("backend_storage must be either 'ket' or 'tt'")

    self.k = k
    self.tokenizer = tokenizer or (lambda s: s.split()) # just split word by word if no tokenizer
    self.alpha = alpha
    self.min_prefix_count = min_prefix_count
    self.max_prefix_count = max_prefix_count
    self.backend_storage = backend_storage
    self.tt_rank = tt_rank

    # initialize extracted objects
    self.prefix2idx: Dict[Tuple[str, ...], int] = {}
    self.idx2prefix: List[Tuple[str, ...]] = []
    self.token_idx: Dict[str, int] = {}
    self._state: Dict[str, np.ndarray] = {}
    self._cores: Dict[str, List[np.ndarray]] = {}
    self._joint_counts: Dict[Tuple[Tuple[str, ...], str], int] = defaultdict(int)
    self._prefix_counts: Counter[Tuple[str, ...]] = Counter()

  def update_from_tokens(self, tokens: List[str]):
    """
      Increments counts for prefixes and suffixes from tokens of one document.
    """
    k = self.k

    # get counts
    padded = ["<s>"] * (k - 1) + tokens
    for i in range(k-1, len(padded)):
      prefix = tuple(padded[i-k+1:i]) # take prefix
      suffix = padded[i]
      self._joint_counts[(prefix, suffix)] += 1
      self._prefix_counts[prefix] += 1

  def fit(self):
    """
      Assemble reduced density matrices from derived counts lists
    """
    k = self.k

    # filtering by min first
    filtered_prefixes = {prefix for prefix, count in self._prefix_counts.items() if self.min_prefix_count <= count}
    # check if exceed max size and adjust if so
    if len(filtered_prefixes) > self.max_prefix_count:
      print(f"Maximum number of distinct prefixes exceeded: {len(filtered_prefixes)}. Considering only the {self.max_prefix_count} most frequent.")
      filtered_prefixes = set(p for p, _ in self._prefix_counts.most_common(self.max_prefix_count))

    print(f"Number of distinct prefixes after filtering: {len(filtered_prefixes)}")

    self.prefix2idx = {prefix: idx for idx, prefix in enumerate(filtered_prefixes)}
    dim = len(self.prefix2idx) # get dimension of system
    self.dim = dim
    self.idx2prefix = [prefix for prefix, _ in sorted(self.prefix2idx.items(), key=lambda x: x[1])]

    print(f"Dimension of reduced density bag: {dim}")

    # build state vectors
    print("starting to build state vectors")
    for (prefix, suffix), count in self._joint_counts.items():
      # check if pruned
      if prefix not in self.prefix2idx:
        continue
      row = self.prefix2idx[prefix]
      self.token_idx.setdefault(suffix, len(self.token_idx))

      state = self._state.setdefault(suffix, np.zeros(dim, dtype=float))
      state[row] = math.sqrt(count + self.alpha) # do we even need smoothing??

    # tt compression
    print("finished states")
    if self.backend_storage == "tt":
      print("starting tt compression")
      for suffix, state in self._state.items():
        # scaling law to get size estimate
        sigma = math.ceil(dim ** (1 / (k - 1)))  # compress exponential space to linear
        new_shape = [sigma] * (k - 1)
        last_dim = np.prod(new_shape)
        if last_dim > dim:  # pad with zeros if too small currently
          state = np.pad(state, (0, last_dim - dim))
        state_tensor = state.reshape(new_shape)

        # now get tt representation cores
        self._cores[suffix] = tensor_train(state_tensor, rank=self.tt_rank)

      # clear states for storage
      print("finished tt compression, dumping state vectors") # ADD FALLBACK HERE ON FAILURE OF COMPRESSION
      self._state.clear()

    print("finished building reduced density bag")
    return self

  # utility functions
  def _get_state(self, token: str) -> Optional[np.ndarray]:
    """
      Get ket vector for token if it exists.
    """
    if self.backend_storage == "ket":
      return self._state.get(token)
    if token not in self._cores:
      return None

    cores = self._cores[token]
    return tt_to_tensor(cores).ravel()[:self.dim]

  def document_reduced_density_matrix(self, raw_text: str) -> List[np.ndarray]:
    """
      Get reduced density matrix for a document.
    """
    # assemble full matrix
    M = np.zeros((self.dim, self.dim), dtype=float)
    for token in self.tokenizer(raw_text):
      state = self._get_state(token)
      if state is None:
        continue
      M += np.outer(state, state)

    return M

  def document_state_vectors(self, raw_text: str) -> List[np.ndarray]:
    """
      Get list of ket vectors for tokens in a document.
    """
    return [state for token in self.tokenizer(raw_text) if (state := self._get_state(token)) is not None]
