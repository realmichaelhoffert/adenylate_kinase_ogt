
import sys
sys.path.append('/data/mhoffert/fiererlab/adenylate_kinase_ogt/protein_utils/')


import glob
from tqdm import tqdm
import multiprocessing
from functools import partial
import argparse
import os

import pandas as pd

from pdb_metrics import process_structure

import pickle
    
# def main(input_list, n_pool):
#     with Pool(n_pool) as pool:
#         L = pool.starmap(process_structure, input_list)
#     return L

def update_progress(lock : multiprocessing.Manager.Lock, 
                    completed_tasks : multiprocessing.Manager.Value, 
                    total_tasks : int):
    '''
    Use multiprocessing.Lock to print a progress bar
    '''
    
    with lock:
        completed_tasks.value += 1
        progress = completed_tasks.value / total_tasks
        bar_length = 40
        block = int(round(bar_length * progress))
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {round(progress * 100, 2)}%"
        sys.stdout.write(text)
        sys.stdout.flush()

def worker_wrapper(args : tuple):
    '''
    args: 
        data : tuple, inputs for process_structures
        data[0] : str, pdb file to analyze
        data[1] : str, unique id of protein in file
        data[2] : bool, whether to save processed structure
        data[3] : str, filepath to save structure
    args 1-3:
    lock, completed tasks, total tasks
    for updating progress bar
    '''
    data, total_tasks, completed_tasks, lock = args
    result = process_structure(*data)
    update_progress(lock, completed_tasks, total_tasks)
    return result

def main(input_list : list, n_pool : int) -> list:
    '''
    input_list: list of tuples to input into function
    n_pool : number of workers in the pool
    '''
    # get total tasks
    total_tasks = len(input_list)

    # construct lock and manager
    with multiprocessing.Manager() as manager:
        completed_tasks = manager.Value('i', 0)
        lock = manager.Lock()

        # argument list for wrapper function
        partial_worker_args = [(data, total_tasks, completed_tasks, lock) for data in input_list]

        # run wrapper with Pool and progress bar
        with multiprocessing.Pool(n_pool) as pool:
            results = pool.map(worker_wrapper, partial_worker_args)
        
        print()
        
    return results


if __name__=="__main__":
    # freeze_support()
    
    # create args
    parser = argparse.ArgumentParser(description='Analyze structures with multiprocessing and BioPBD')

    # structure file path
    parser.add_argument('-s', '--structures', type=str, required=True,
                        help='filepath of folder containing structures')
    # output file path
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
    # get absolute path
    path = os.path.abspath(args.structures)
    # get structures
    structures = glob.glob(f'{args.structures}*.pdb')

    # inputs: file, unique id, whether to write processed structures,
    # save path for processed stuctures
    inputs = [(f, 
               f.split('/')[-1].split('_closed')[0], 
               args.write_processed, 
               os.path.abspath(args.save_processed)) 
              for f in structures]
    
    # process only n files 
    if args.number > 0:
        inputs = inputs[:args.number]
        
    # if no inputs
    if len(inputs) > 0:
        results = main(inputs, args.threads)
        # with open(args.output, 'wb') as phandle:
        #     pickle.dump(results, phandle)
        # concatenate and save to output fp
        all_result = pd.concat(results, axis=0)
        all_result.to_csv(args.output, sep='\t', index=False)
    else:
        raise ValueError('input filepath invalid')
    
