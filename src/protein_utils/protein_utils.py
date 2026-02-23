import pandas as pd
import sys
# import subprocess



pfam_table_columns = ['target_name','target_accession','query_name','query_accession', 
                      'full_e_value', 'full_score', 'full_bias', 
                      'best_domain_e_value', 'best_domain_score', 'best_domain_bias', 
                      'exp', 'reg', 'clu',  'ov', 'env', 'dom', 'rep', 'inc', 'description_of_target']


def read_pfam_table_pandas(file_path : str) -> pd.DataFrame:
    """
    file_path: path to pfam "tblout"
    """
    # read table with special parameters
    pfam_as_table = pd.read_table(file_path, skiprows=3,  skipfooter=10, sep='\s+', header=None, engine='python')
    # un-split last 6 columns
    description = pfam_as_table.iloc[:, 18:].apply(lambda row: ''.join([str(v).lstrip() for v in row.values]), axis=1)
    # remove last 6 columns, add new one of joined values
    pfam_as_table = pfam_as_table.iloc[:, :18].assign(description_of_target=description.values)
    # reset column names
    pfam_as_table.columns = pfam_table_columns
    
    return pfam_as_table

def get_lid(hmm_model, sequence_file):
    '''
    hmm_model: absolute file path to HMM of lid domain (PF05191)
    sequence_file: protein fasta of adenylate kinase sequence
    '''

    # Run hmmscan command
    p = subprocess.run(['hmmsearch', '-o', '/dev/null', '--domtblout', '/dev/stdout', hmm_model, sequence_file], 
                       capture_output=True, text=True)
    # parse hit lines
    evals = []
    for i, line in enumerate(p.stdout.split('\n')[:-1]):
        if line.startswith('#'):
            continue
        else:
            fields = line.split()
            # append evalue, start, stopß
            evals.append((fields[13], fields[17], fields[18]))

    # if a hit was found
    if len(evals) > 0:
        # sort by evalue
        sorted_evals = sorted(evals, key=lambda x: x[0])
        # return lowest
        return [int(i) for i in sorted_evals[-1][1:]]
    else:
        # return negative locs if not founds
        return [-1, -1]
