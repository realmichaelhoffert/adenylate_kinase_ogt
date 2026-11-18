import numpy as np
from tqdm import tqdm
from roseate.io import reload_pickle, load_embeddings

def run_model(
              embedding_loc : str, 
              model_loc : str,
              msa_loc : str,
              predict_fn):

    '''
    embedding loc: location of pickled embeddings (.npy or pickle)
    model_loc: location of model to use
    msa_loc: location of msa used to generate embeddings
    predict_fn: function used to generate predictions
    '''
    loaded_model = reload_pickle(model_loc)
    loaded_embeddings = load_embeddings(embedding_loc)
    loaded_msa_dataset = reload_pickle(msa_loc)

    # make sure loaded embeddings match embeddings data was run on
    assert loaded_model['metadata']['input_shape'][1:] == loaded_embeddings.shape[1:]
    
    predictions = np.array(predict_fn(models=loaded_model['best_model'], 
                             emb=loaded_embeddings,
                             scalers=loaded_model['scalers'] if 'scalers' in loaded_model.keys() else None))
    return predictions

def per_site_predict(models, emb, scalers):
    '''
    Function to compute per-site predictions
    models: dict, site/alignment col -> model obj
    emb: matrix, proteins x aln_len x n_emb_dim
    scalers: dict, site/alignment col -> scaler obj
    '''

    preds = []
    for col in tqdm(sorted(models.keys())):
        if scalers is not None:
            site_emb = scalers[col].transform(emb[:, col, :])
        else:
            site_emb = emb[:, col, :]
            
        preds.append(models[col].predict(site_emb))
    
    return preds
