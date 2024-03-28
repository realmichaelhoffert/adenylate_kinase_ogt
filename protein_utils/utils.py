import pandas as pd
import sys

def blah():
    print('test')

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
