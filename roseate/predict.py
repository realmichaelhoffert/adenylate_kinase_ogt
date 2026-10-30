import numpy as np
from tqdm import tqdm
from roseate.io import reload_pickle

def run_model(
              embedding_loc : str, 
              model_loc : str,
              msa_loc : str,
              predict_fn):

    '''
    embedding loc: location of pickled embeddings
    model_loc: location of model to use
    msa_loc: location of msa used to generate embeddings
    predict_fn: function used to generate predictions
    '''
    loaded_model = reload_pickle(model_loc)
    loaded_embeddings = reload_pickle(embedding_loc)
    loaded_msa_dataset = reload_pickle(msa_loc)

    # make sure loaded embeddings match embeddings data was run on
    assert loaded_model['metadata']['input_shape'][1:] == loaded_embeddings.shape[1:]
    
    predictions = np.array(predict_fn(model=loaded_model['best_model'], 
                             emb=loaded_embeddings,
                             scalers=loaded_model['scalers'] if 'scalers' in loaded_model.keys() else None))
    return predictions

def per_site_predict(model, emb, scalers):
    '''
    Function to compute per-site predictions
    '''

    preds = []
    for col in tqdm(sorted(model.keys())):
        if scalers is not None:
            site_emb = scalers[col].transform(emb[:, col, :])
        else:
            site_emb = emb[:, col, :]
            
        preds.append(model[col].predict(site_emb))
    
    return preds
