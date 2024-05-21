
import sys
sys.path.append('/data/mhoffert/fiererlab/adenylate_kinase_ogt/protein_utils/')


import glob
from multiprocessing import Pool, freeze_support
import argparse
import os

import pandas as pd

from pdb_metrics import process_structure

def main(input_list):
    with Pool(25) as pool:
        L = pool.starmap(process_structure, input_list)
    return L

if __name__=="__main__":
    # freeze_support()
    
    parser = argparse.ArgumentParser(description='Analyze structures with multiprocessing and BioPBD')

    parser.add_argument('-s', '--structures', type=str, required=True,
                        help='filepath of folder containing structures')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='outfile')
    parser.add_argument('-t', '--threads', type=int, default=6, required=False,
                        help='number of threads (default 6)')
    parser.add_argument('-n', '--number', type=int, default=-1, required=False,
                        help='number of structures to process (default all)')
    parser.add_argument('-p', '--write-processed', type=bool, default=False, required=False,
                        help='Write processed structures (default False)')
    parser.add_argument('--save-processed', type=str, default = './', required=False, 
                       help='filepath to save processed structures')
    
    
    
    args = parser.parse_args()
    path = os.path.abspath(args.structures)
    print(path)
    structures = glob.glob(f'{args.structures}*.pdb')
    print(len(structures))
    print(structures[:5])
    # inputs: file, unique id, whether to write processed structures,
    # save path for processed stuctures
    inputs = [(f, 
               f.split('/')[-1].split('_closed')[0], 
               args.write_processed, 
               os.path.abspath(args.save_processed)) 
              for f in structures]
    
    if args.number > 0:
        inputs = inputs[:args.number]
        
    print(inputs)
    if len(inputs) > 0:
        results = main(inputs)
        
        all_result = pd.concat(results, axis=1).T
        all_result.to_csv(args.output, sep='\t', index=False)
    else:
        raise ValueError('input filepath invalid')
    
