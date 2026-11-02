# Written by Michael Hoffert
# October 2026

# python
import pickle
from pathlib import Path

# numbers, ML
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV

# other libs
from tqdm import tqdm

# roseate code
from roseate.predict import per_site_predict
from roseate.io import MSADataset




ENETCV_PARAMS = {'l1_ratio':[0.01, 0.1, 0.25, .5, .75, .9, 0.99, 1.0], 
                'cv':5, 
                'n_alphas':100, 
                'max_iter':10000, 
                'selection':'random', 
                'tol':1e-3, 
                'precompute':False, 
                'random_state':123, 
                'n_jobs':24}

def train_per_site(train_emb : np.array, train_y : np.array, sites = None, verbose = False):

    '''
    Fit per-site models
    train_emb: matrix, (n_proteins x protein_len x n_emb_dim)
    train_y: vector, values of trait
    sites: which sites to train
    '''

    col_enets = {}
    col_scalers = {}
    if sites is not None:
        try:
            emb = train_emb[:, sites, :]
        except KeyError:
            assert False, 'sites must be subset of columns'
    else:
        emb = train_emb
        sites = range(emb.shape[1])
        
    if verbose:
        print(f'Training on {emb.shape[1]} sites...')
        
    for col in tqdm(sites):
        # get column of embedding
        col_embedding = train_emb[:, col, :]
        col_scalers[col] = StandardScaler()
        X = col_scalers[col].fit_transform(col_embedding)
        # fit
        enet = ElasticNetCV(**ENETCV_PARAMS)
        enet.fit(X, train_y)
        
        col_enets[col] = enet

    return col_enets, col_scalers

def test_per_site(models, scalers, test_emb):
    '''
    wrapper for test predictions
    '''
    return per_site_predict(models, test_emb, scalers)

def package_model(model_label : str,
                  enets : dict, 
                  scalers : dict, 
                  test_preds : np.array, 
                  test_dataset : MSADataset, 
                  input_shape : tuple,
                  model_type : str, 
                  extra_metadata=None, extra_fields=None):
    '''
    save the model

    enets, scalers : dicts {site: ElasticNetCV}, {site: StandardScaler} from fit_per_site
    test_preds : (n_sites, n_test) per-site predictions from predict_test, in range order
    test_dataset : MSADataset the predictions were made on, in the same order
    input_shape : embedding shape the model expects, e.g. train_emb.shape
    model_type : free-text metadata['type']
    '''
    
    full_predictions = np.asarray(test_preds)
    metadata = {"input_shape": input_shape, 
                "type": model_type, 
                'label':model_label}
    metadata.update(extra_metadata or {})
    model = {
        "best_model": enets,
        "scalers": scalers,
        "full_predictions": full_predictions,
        "predictions": full_predictions.mean(axis=0),
        "true_values": [i[2] for i in test_dataset],
        "true_labels": [i[0] for i in test_dataset],
        "metadata": metadata,
    }
    model.update(extra_fields or {})      # e.g. {"statsmodels_version": ...}, to match v1 exactly
    return model


def save_model(model, path, overwrite=False):
    """Pickle a model dict. Refuses to replace an existing file unless overwrite=True."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path
    
# validation models
def fit_onehot_per_site():
    ''
def hot_encode_site():
    ''
def add_dummy():
    ''







    
    

    