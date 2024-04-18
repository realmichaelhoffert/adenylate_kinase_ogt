import glob
import sys

from pdb_metrics import process_structure

import pandas as pd

sys.path.append('/data/mhoffert/fiererlab/adenylate_kinase_ogt/protein_utils/')

structures = glob.glob('/data/mhoffert/fiererlab/adenylate_kinase_ogt/data/test_structures/*.pdb')
inputs = [(f, f.split('/')[-1].split('_closed')[0], True, sys.argv[1]) for f in structures]

from multiprocessing import Pool, freeze_support

def main():
    with Pool(25) as pool:
        L = pool.starmap(process_structure, inputs)
    
    return L

if __name__=="__main__":
    freeze_support()
    results = main()
    all_result = pd.concat(results, axis=1).T
    all_result.to_csv(sys.argv[2], sep='\t', index=False)