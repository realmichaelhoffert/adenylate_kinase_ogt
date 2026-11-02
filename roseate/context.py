# Written by Michael Hoffert
# October 2026


# typing libs
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string

# python libs
import os
import pickle
import random

# ML utils
import numpy as np
# from sklearn.preprocessing import StandardScaler

# Sequence IO
# from Bio.Seq import Seq as Seq
# from Bio import SeqIO as SeqIO

# # PLM stuff
# import esm as esm
# import torch as torch

# other
from tqdm import tqdm

# roseate
from roseate import io

MSADataset = list[tuple]
def get_random_context(msa_dataset : MSADataset, 
                       n_contexts : int,
                       context_size : int,
                       self_exclude=True,
                       seed=1234) -> list[list[int]]:
    '''
    function to get context (additional proteins) for embedding
    msa_dataset: list[tuple], the dataset to set a context for
    context_size: int, number of inds to select
    self_exclude: Bool, if True, excludes ind i from selection for context i
    returns:
    for M proteins and context size n:
    M x n: indices of dataset
    works with get_embeddings, required to embed proteins in a
    consistent context
    '''
    random.seed(a=seed)
    
    indices = list(range(len(msa_dataset)))
    
    if self_exclude:
        assert_msg = f'self_exclude = {self_exclude}, context: {context_size} must be <= dataset size -1: {len(indices) -1}'
        assert context_size <= len(indices)-1, assert_msg
    else:
        assert_msg = f'self_exclude = {self_exclude}, context: {context_size} must be <= dataset size: {len(indices) -1}'
        assert context_size <= len(indices), assert_msg
    
    contexts = []
    if self_exclude:
        for i in indices:
            contexts.append(random.sample([j for j in indices if j != i], context_size))
    else:
        for i in range(n_contexts):
            contexts.append(random.sample(indices, context_size))

    return contexts

def load_legacy_context(context_path : str,
                       context_dataset):
    '''
    reload context
    context_path: path to pickle of context list
    context_dataset: MSADataset
    returns: list of indices
    
    '''
    
    assert os.path.exists(context_path), f'{context_path} not found'
    context = io.reload_pickle(context_path)

    msa_dataset = context_dataset

    indices = context_to_indices(context, msa_dataset)
    return indices

def context_to_indices(ctx, dataset, allow_oversampling=False):
    """
    Saved get_embeddings contexts are lists of (id, seq); convert to integer indices.
    ctx: list of (id, seq)
    dataset: msa_dataset
    allow_oversampling: whether duplicate IDs are allowed: fix later
    """
    id2i = {}
    for i, item in enumerate(dataset):
        if allow_oversampling:
            assert item[0] not in id2i, f"duplicate id in context dataset: {item[0]}"
        id2i[item[0]] = i
    rows = []
    for row in ctx:
        idx = []
        for cid, cseq in row:
            i = id2i[cid]
            assert dataset[i][1] == cseq, f"sequence mismatch for context id {cid}"
            idx.append(i)
        rows.append(idx)
    return np.array(rows, dtype=np.int64)
    
