# Written by Michael Hoffert

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

def reload_model(path : str) -> dict:
    """
    Code to reload models saved as pickles
    (usually a dictionary)
    """
    with open(path, 'rb') as handle:
        model_data = pickle.load(handle)
    return model_data

def unscale_ogts(ogt_list, scaler : StandardScaler) -> list:
    """
    function to unscale data given a scaler
    """
    return scaler.inverse_transform(np.array(ogt_list).reshape(1, -1))[0]


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