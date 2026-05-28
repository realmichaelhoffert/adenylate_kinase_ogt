# Written by Michael Hoffert

# typing libs
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string

# python libs
import pickle

# ML utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.neighbors import KernelDensity

# Sequence IO
from Bio.Seq import Seq as Seq
from Bio import SeqIO as SeqIO

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.stats import spearmanr

import copy

def reload_model(path : str) -> dict:
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
    return '$R^{2}$=' + f'{r2:.2f}|RMSE={rmse:.2f}|SpR={_spearmanr:.2f}', (r2, rmse, spearmanr)

def unscale_ogts(ogt_list, scaler : StandardScaler) -> list:
    """
    function to unscale data given a scaler
    """
    return scaler.inverse_transform(np.array(ogt_list).reshape(1, -1))[0]


# function to get weights based on y variable density
def compute_gaussian_weights(_y):
    # Normalize the target variable to compute KDE
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(_y.reshape(-1, 1))
    
    # Estimate density of target variable using KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.4)
    kde.fit(y_scaled)
    log_density = kde.score_samples(y_scaled)
    density = np.exp(log_density)
    
    # Compute weights as inverse of density
    weights = 1 / density

    return weights


MSADataset = list[tuple]
get_ogt = lambda _str: float(_str.split('temp=')[-1].split('|')[0])
def construct_msa_dataset(file : string,
                         ogt_parse_fn=get_ogt) -> MSADataset:
    '''
    Function to read in an aligned fasta and get a dataset:
    list(tuple): (id, sequence, ogt)
    # maybe should modify to take OGT list
    '''
    # set up dataset
    with open(file, 'r') as handle:
        records = [r for r in SeqIO.parse(handle=handle,format='fasta')]
    seqs_with_ogt = {}
    for r in records:
        seqs_with_ogt[r.id] = (str(r.seq), ogt_parse_fn(r.id))
    return [(k, item[0], item[1]) for k, item in seqs_with_ogt.items()]

def scale_ogts(msa_dataset : MSADataset, scaler : StandardScaler = None) -> MSADataset:
    '''
    dataset: list of tuples: (id, msa seq, ogt)
    if given a scaler, scales
    otherwise, generates a scaler and scales the ogts
    '''

    ogts = [[i[2]] for i in msa_dataset]
    
    if scaler is None:
        _scaler = StandardScaler()
        _scaler.fit(ogts)
    else:
        _scaler = copy.deepcopy(scaler)
    scaled_ogts = [i[0] for i in _scaler.transform(ogts)]
    scaled_msa_dataset = [(msa_dataset[i][0], msa_dataset[i][1], scaled_ogts[i]) for i in range(len(msa_dataset))]
    return scaled_msa_dataset, _scaler


def plot_performance(y_true, y_pred, title, show=False, save_loc=None, colors=None, label=None):
    """
    Plot performance of a given model
    """
    
    fig, ax  = plt.subplots(figsize=(4, 4))
    sc = plt.scatter(y_true, y_pred, 
                     alpha=0.7, edgecolors='w')
    # plt.colorbar(sc, label='Normalized OGT')

    r2 = r2_score(y_true, y_pred)
    mse = np.sqrt(mean_squared_error(y_true, y_pred))
        
    ax.set_title(f"{title}\n$R^{2}$: {r2:.2f} | RMSE: {mse:.2f}")
    # plt.title(title)
    plt.tick_params(labelsize=14, axis='both')
    plt.xlabel(f"True", fontsize=16)
    plt.ylabel(f"Predicted", fontsize=16)
    sns.despine()
    ax.set_aspect('equal')
    
    _min = np.floor(np.min([np.min(y_true), np.min(y_pred)]))
    print(_min, _min-(_min*.5))
    _max = np.floor(np.max([np.max(y_true), np.max(y_pred)]))
    _range = np.abs(_max-_min)
    ax.set_ylim(_min-(_range*.1),
                _max+(_range*.1))
    ax.set_xlim(_min-(_range*.1),
            _max+(_range*.1))
    
    plt.plot((_min+0.5, _max-0.5), (_min+0.5, _max-0.5), 'k--', label='x=y')
    if save_loc is not None:
        plt.savefig(save_loc, bbox_inches='tight')
    if show:
        plt.show()

    plt.cla()
    plt.close()