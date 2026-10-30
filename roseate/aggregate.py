# Code for aggregating predictions across sites
# Written by Michael Hoffert
# October, 2026

# python imports
import os

# data imports
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler

# ROSEATE functions
from roseate.io import unscale_traits

def mean_sites(full_preds : np.array, cols=None) -> np.array:
    '''
    full_preds: matrix M x N
    M: sites
    N: proteins
    '''
    
    if cols is None:
        return full_preds[cols, :].mean(axis=0)
    else:
        return full_preds.mean(axis=0)

def load_site_spearmans(file):
    '''
    path to spearmans, measured per site
    assumes column 0 = index, column 1 = spearmans
    '''
    assert os.path.exists(file), f'{file} not found'
    spearmans = pd.read_csv(file, index_col=0,
                            sep=None, compression='infer', engine='python')
    
    return spearmans.iloc[:, 0]
    
# def site_spearmans(models : dict, 
#                    scalers : dict, 
#                    emb, emb_labels, y):
#     '''
#     Calculate per-site spearman R values
#     models: dict, {key -> model object}
#     scalers: dict, {key -> StandarScaler}
#     emb: N x M x L matrix: 
#         N = number of proteins in test
#         M = embedding dim (768)
#         L = length of alignment
#     y: true values 
#     Note: model_label needs to be int to correspond to index in embedding matrix
#     '''
#     model_rs = pd.Series()
#     # good_cols = []
#     model_keys = sorted(list(models.keys()))
#     for model_label in model_keys:
#         preds = pd.Series(index=model_keys, 
#                       data = models[model_label].predict(scalers[model_label].transform(test_emb[:, model_label, :])))
    
#     # for model_label in models.keys(): 
#         spr = spearmanr(preds.reindex(, y.reindex())[0]
#         col_rs.loc[col] = spr