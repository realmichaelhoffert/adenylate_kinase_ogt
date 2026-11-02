# typing libs
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string

# python libs
import pickle
import random

# ML utils
from sklearn.preprocessing import StandardScaler

# Sequence IO
from Bio.Seq import Seq as Seq
from Bio import SeqIO as SeqIO

# PLM stuff
import esm as esm
import torch as torch
import numpy as np

# other
from tqdm import tqdm

MSADataset = list[tuple]

def load_plm():
    '''
    Load ESM-based PLM 
    '''
    # load ESM model
    esm_msa_model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    conv = alphabet.get_batch_converter()
    # turn off gradients
    for name, param in esm_msa_model.named_parameters():
        param.requires_grad = False

    return esm_msa_model, alphabet, conv

def get_embeddings(model : esm.model.msa_transformer.MSATransformer, 
                   batch_converter : esm.data.MSABatchConverter, 
                   data_to_embed : MSADataset, 
                   context_dataset : MSADataset,
                   context_inds : list[list[int]],
                   layer=12, return_mean=True) -> (np.array, np.array):
    '''
    model: esm msa transformer
    converter: esm batch converter
    data_to_embed: list(tuple): (id, msa seq, ogt)
    training_context: list(tuple): (id, msa seq, ogt)
    context_inds: specification of which seqs in training context are used for each item to embed
    n_context: int, used to select number of proteins to embed query alongside
    layer: embedding layer to extract
    returns embeddings and temps
    '''
    model.eval()
    embeddings = []
    traits = []

    contexts = []
    with torch.no_grad():
        ind = 0
        for item in tqdm(data_to_embed):
            # Prepare a simple MSA for extraction
            context = [(context_dataset[c][0], context_dataset[c][1]) for c in context_inds[ind]]
            msa = [(item[0], item[1])] + context
            _, _, tokens = batch_converter(msa)
            contexts.append(context)
            
            # Extract Layer 12 Mean-Pooled embedding
            output = model(tokens, repr_layers=[layer])["representations"][layer]
            if return_mean:
                emb = output[:, 0, :, :].mean(dim=1).squeeze(0).cpu().numpy()
            else:
                emb = output[:, 0, :, :].squeeze(0).cpu().numpy()
            embeddings.append(emb)
            traits.append(item[2])
            ind += 1

    return np.array(embeddings), np.array(traits), contexts