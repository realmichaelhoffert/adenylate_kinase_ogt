# Written by Michael Hoffert
# October 2026

# typing libs
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string

# # python libs
import pickle
import copy
import os
import warnings

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
import numpy as np
# from scipy.stats import spearmanr



MSADataset = list[tuple]
# get_ogt = lambda _str: float(_str.split('temp=')[-1].split('|')[0])

def construct_msa_dataset(file : str,
                          trait_data : str = None) -> MSADataset:
    '''
    Function to read in an aligned fasta and get a dataset:
    list(tuple): (id, sequence, trait_value)
    file: string, path to fasta file
    trait_data: string, path to comma or tab-delimited file with:
        column 0: sequence id (exactly as in fasta)
        column 1: trait values

    Note: only returns MSA for values found in trait_data
    '''
    # checks
    assert os.path.exists(file), f'{file} not found'
    

    # set up dataset
    with open(file, 'r') as handle:
        records = [r for r in SeqIO.parse(handle=handle, format='fasta')]

    if trait_data is not None:
        assert os.path.exists(trait_data), f'{trait_data} not found'
    
        # read trait data, assumes column 1 is ids, column 2 is trait data
        trait_data = pd.read_csv(trait_data, sep=None, engine='python', index_col=0).iloc[:, 0].dropna()
        # check ids
        assert len(records) == len(set([r.id for r in records]))
        record_ids = set([r.id for r in records])
        assert all([t_id in record_ids for t_id in trait_data.index]), f'all trait data in records'
        intersection_size =  len(record_ids.intersection(set(trait_data.index)))
        if intersection_size != len(record_ids):
            warnings.warn(f'Only {intersection_size} records are in the trait data')
    
    seqs_with_trait = {}
    for r in records:
        if trait_data is not None:
            if r.id in trait_data.index:
                seqs_with_trait[r.id] = (str(r.seq), trait_data.loc[r.id])
        else:
            seqs_with_trait[r.id] = (str(r.seq), np.nan)
    return [(k, item[0], item[1]) for k, item in seqs_with_trait.items()]

def save_split(dataset, path, dataset_name):
    '''
    Save a subset of the data
    '''
    assert os.path.exists(path)
    
    with open(os.path.join(path, dataset_name, '_msa_dataset.pickle'), 'wb') as handle:
        pickle.dump(dataset, handle)

    with open(os.path.join(path, dataset_name, '_ids.pickle'), 'wb') as handle:
        pickle.dump([i[0] for i in dataset], handle)

def load_new_dataset(afa_file : str, 
                     scaler : StandardScaler, 
                     exclude_ids : list = [], 
                    trait_data : str = None, ) -> MSADataset:
    '''
    Load a new dataset
    afa_file: strfile sequences of dataset inserted into backbone
    trait_data: str, file of trait data, column 0 = seq id column 1 = trait
    scaler: existing trait data scaler
    exclude_ids: ids to exclude (e.g. those from backbone)
    '''
    assert os.path.exists(afa_file), f'{afa_file} not found'
        
    # read msa
    msa_dataset = construct_msa_dataset(afa_file, trait_data)
    msa_dataset = [i for i in msa_dataset if not i[0] in exclude_ids]
    # scale traits using provided scaler
    scaled_msa_dataset, _ = scale_traits(msa_dataset, scaler)
    return scaled_msa_dataset

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
                   corkrey_gtdb_match_file : str,
                   corkrey_data_file : str) -> MSADataset:

    '''
    Temporary function to reconstruct corkrey dataset
    legacy paths: '/data/mhoffert/fiererlab/ogt/results/db/corkey_gtdb_matching.tsv'
    '/data/mhoffert/fiererlab/ogt/source_data/Corkey_temperatures.csv'
    
    '''
    for file in [corkrey_gtdb_match_file, corkrey_data_file]:
        assert os.path.exists(file), f'{file} not found'

    corkrey_matches = pd.read_csv(corkrey_gtdb_match_file, sep='\t', index_col=0)
    corkrey_data = pd.read_csv(corkrey_data_file)
    corkrey_full = pd.merge(corkrey_data.groupby('binomial.name').agg({'T.C':'mean'}), corkrey_matches, left_index=True, right_on='original_id')
    
    # get ids
    corkrey_ids = set(corkrey_full['match_accession'].unique())
    #
    corkrey_msa_dataset = [i for i in dataset if i[0].split('|')[1] in corkrey_ids]
    # print("Number of observations in Corkrey dataset:", len(corkrey_msa_dataset))
    return corkrey_msa_dataset, corkrey_ids

def sample_even(dataset, exclude, n_bins=40, n_per_bin=100 ):
    '''
    Function to recreate sampling of even dataset
    dataset: MSADataset to subsample
    exclude: IDs from test data to exclude
    n_bins: number of bins to segment trait data into
    n_per_bin: max number to select per bin
    '''
    
    dataset_df = pd.DataFrame(dataset, columns=['id', 'seq', 'ogt'])
    lower = np.floor(dataset_df['ogt'].min()) - 1
    upper = np.ceil(dataset_df['ogt'].max())
    dataset_df['group'] = pd.cut(dataset_df['ogt'], np.linspace(lower, upper, n_bins))
    even_ids = dataset_df.groupby('group', observed=True).apply(lambda x: x.sample(np.min([len(x), n_per_bin]), random_state=111, replace=False))['id'].unique()

    even_dataset = [i for i in dataset if i[0] in even_ids and not i[0] in exclude]
    return even_dataset, set([i[0] for i in even_dataset])


    
    
    

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