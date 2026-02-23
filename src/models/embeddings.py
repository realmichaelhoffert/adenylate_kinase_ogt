# typing libs
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string

# python libs
import pickle

# ML utils
from sklearn.preprocessing import StandardScaler

# Sequence IO
from Bio.Seq import Seq as Seq
from Bio import SeqIO as SeqIO

# PLM stuff
import esm as esm
import torch as torch
import numpy as np

MSADataset = list[tuple]
def get_random_context(msa_dataset : MSADataset, 
                       n_contexts : int,
                       context_size : int,
                       self=True) -> list[list[int]]:
    '''
    function to get context (additional proteins) for embedding
    msa_dataset: list[tuple], the dataset to set a context for
    context_size: int, number of inds to select
    self: Bool, if True, excludes ind i from selection for context i
    returns:
    for M proteins and context size n:
    M x n: indices of dataset
    works with get_embeddings, required to embed proteins in a
    consistent context
    '''
    indices = list(range(len(msa_dataset)))
    
    if self:
        assert_msg = f'self = {self}, context: {context_size} must be <= dataset size -1: {len(indices) -1}'
        assert context_size <= len(indices)-1, assert_msg
    else:
        assert_msg = f'self = {self}, context: {context_size} must be <= dataset size: {len(indices) -1}'
        assert context_size <= len(indices), assert_msg
    
    contexts = []
    if self:
        for i in indices:
            contexts.append(random.sample([j for j in indices if j != i], context_size))
    else:
        for i in range(n_contexts):
            contexts.append(random.sample(indices, context_size))

    return contexts
    
def get_embeddings(esm_model : esm.model.msa_transformer.MSATransformer, 
                   converter : esm.data.MSABatchConverter, 
                   data_to_embed : MSADataset, 
                   training_context : MSADataset,
                   context_inds : list[list[int]],
                   layer=12, return_mean=True) -> (np.array, np.array):
    '''
    esm_model: esm msa transformer
    converter: esm batch converter
    data_to_embed: list(tuple): (id, msa seq, ogt)
    training_context: list(tuple): (id, msa seq, ogt)
    context_inds: specification of which seqs in training context are used for each item to embed
    n_context: int, used to select number of proteins to embed query alongside
    layer: embedding layer to extract
    returns embeddings and temps
    '''
    esm_model.eval()
    embeddings = []
    temperatures = []

    contexts = []
    with torch.no_grad():
        ind = 0
        for item in tqdm(data_to_embed):
            # Prepare a simple MSA for extraction
            context = [(training_context[c][0], training_context[c][1]) for c in context_inds[ind]]
            msa = [(item[0], item[1])] + context
            _, _, tokens = converter(msa)
            contexts.append(context)
            
            # Extract Layer 12 Mean-Pooled embedding
            output = esm_model(tokens, repr_layers=[layer])["representations"][layer]
            if return_mean:
                emb = output[:, 0, :, :].mean(dim=1).squeeze(0).cpu().numpy()
            else:
                emb = output[:, 0, :, :].squeeze(0).cpu().numpy()
            embeddings.append(emb)
            temperatures.append(item[2])
            ind += 1

    return np.array(embeddings), np.array(temperatures), contexts