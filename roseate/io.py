# Written by Michael Hoffert
# October 2026

# typing libs
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string

# # python libs
import pickle
import copy
import os

# ML utils
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
# from sklearn.neighbors import KernelDensity

# # Sequence IO
# from Bio.Seq import Seq as Seq
from Bio import SeqIO as SeqIO

# plotting, data, stats
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from scipy.stats import spearmanr



MSADataset = list[tuple]
# get_ogt = lambda _str: float(_str.split('temp=')[-1].split('|')[0])

def construct_msa_dataset(file : str,
                          trait_data : str) -> MSADataset:
    '''
    Function to read in an aligned fasta and get a dataset:
    list(tuple): (id, sequence, trait_value)
    file: string, path to fasta file
    trait_data: string, path to comma or tab-delimited file with:
        column 0: sequence id (exactly as in fasta)
        column 1: trait values
    '''
    # checks
    assert os.path.exists(file), f'{file} not found'
    assert os.path.exists(trait_data), f'{trait_data} not found'
    
    # set up dataset
    with open(file, 'r') as handle:
        records = [r for r in SeqIO.parse(handle=handle, format='fasta')]

    # read trait data, assumes column 1 is ids, column 2 is trait data
    trait_data = pd.read_csv(trait_data, sep=None, engine='python', index_col=0).iloc[:, 0]
    # check ids
    assert all([r.id in trait_data.index for r in records]), f'check seq ids match trait data'
    
    seqs_with_trait = {}
    for r in records:
        seqs_with_trait[r.id] = (str(r.seq), trait_data.loc[r.id])
    return [(k, item[0], item[1]) for k, item in seqs_with_trait.items()]

# trait scaling 
def scale_traits(msa_dataset : MSADataset, scaler : StandardScaler = None) -> MSADataset:
    '''
    dataset: list of tuples: (id, msa seq, trait)
    if given a scaler, scales
    otherwise, generates a scaler and scales the trait
    '''

    traits = [[i[2]] for i in msa_dataset]
    
    if scaler is None:
        _scaler = StandardScaler()
        _scaler.fit(traits)
    else:
        _scaler = copy.deepcopy(scaler)
    scaled_traits = [i[0] for i in _scaler.transform(traits)]
    scaled_msa_dataset = [(msa_dataset[i][0], msa_dataset[i][1], scaled_traits[i]) for i in range(len(msa_dataset))]
    return scaled_msa_dataset, _scaler

def unscale_traits(trait_list, scaler : StandardScaler) -> list:
    """
    function to unscale data given a scaler
    """
    return scaler.inverse_transform(np.array(trait_list).reshape(1, -1))[0]

def reload_pickle(path : str) -> dict:
    """
    Code to reload models saved as pickles
    (usually a dictionary)
    """
    with open(path, 'rb') as handle:
        model_data = pickle.load(handle)
    return model_data

def assess_model(y_true, y_pred):
    '''
    Run model assessment
    '''
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    _spearmanr = spearmanr(y_true, y_pred)[0]
    return '$R^{2}$=' + f'{r2:.2f}|RMSE={rmse:.2f}|SpR={_spearmanr:.2f}', (r2, rmse, _spearmanr)


def subset_corkrey(dataset : MSADataset,
                   corkrey_gtdb_match : str,
                   corkrey_data : str) -> MSADataset:

    '''
    Temporary function to reconstruct corkrey dataset
    '''

    

    for file in [corkrey_gtdb_match, corkrey_data]:
        assert os.path.exists(file), f'{file} not found'

    corkrey_matches = pd.read_csv(, sep='\t', index_col=0)
    corkrey_data = pd.read_csv('/data/mhoffert/fiererlab/ogt/source_data/Corkey_temperatures.csv')
    corkrey_full = pd.merge(corkrey_data.groupby('binomial.name').agg({'T.C':'mean'}), corkrey_matches, left_index=True, right_on='original_id')
    
    # get ids
    corkrey_ids = set(corkrey_full['match_accession'].unique())
    #
    corkrey_msa_dataset = [i for i in full_msa_dataset if i[0].split('|')[1] in corkrey_ids]
    print("Number of observations in Corkrey dataset:", len(corkrey_msa_dataset))

# # function to get weights based on y variable density
# def compute_gaussian_weights(_y):
#     # Normalize the target variable to compute KDE
#     scaler = StandardScaler()
#     y_scaled = scaler.fit_transform(_y.reshape(-1, 1))
    
#     # Estimate density of target variable using KDE
#     kde = KernelDensity(kernel='gaussian', bandwidth=0.4)
#     kde.fit(y_scaled)
#     log_density = kde.score_samples(y_scaled)
#     density = np.exp(log_density)
    
#     # Compute weights as inverse of density
#     weights = 1 / density

#     return weights

# def plot_performance(y_true, y_pred, title, show=False, save_loc=None, colors=None, label=None):
#     """
#     Plot performance of a given model
#     """
    
#     fig, ax  = plt.subplots(figsize=(4, 4))
#     sc = plt.scatter(y_true, y_pred, 
#                      alpha=0.7, edgecolors='w')
#     # plt.colorbar(sc, label='Normalized OGT')

#     r2 = r2_score(y_true, y_pred)
#     mse = np.sqrt(mean_squared_error(y_true, y_pred))
        
#     ax.set_title(f"{title}\n$R^{2}$: {r2:.2f} | RMSE: {mse:.2f}")
#     # plt.title(title)
#     plt.tick_params(labelsize=14, axis='both')
#     plt.xlabel(f"True", fontsize=16)
#     plt.ylabel(f"Predicted", fontsize=16)
#     sns.despine()
#     ax.set_aspect('equal')
    
#     _min = np.floor(np.min([np.min(y_true), np.min(y_pred)]))
#     print(_min, _min-(_min*.5))
#     _max = np.floor(np.max([np.max(y_true), np.max(y_pred)]))
#     _range = np.abs(_max-_min)
#     ax.set_ylim(_min-(_range*.1),
#                 _max+(_range*.1))
#     ax.set_xlim(_min-(_range*.1),
#             _max+(_range*.1))
    
#     plt.plot((_min+0.5, _max-0.5), (_min+0.5, _max-0.5), 'k--', label='x=y')
#     if save_loc is not None:
#         plt.savefig(save_loc, bbox_inches='tight')
#     if show:
#         plt.show()

#     plt.cla()
#     plt.close()