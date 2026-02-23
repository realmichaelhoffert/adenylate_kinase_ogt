import os
import glob
import argparse

import pandas as pd
from Bio import SeqIO

from protein_utils import read_pfam_table_pandas, pfam_table_columns

parser = argparse.ArgumentParser()
parser.add_argument('--loc', '-l', required=True,
                   help='Location of files')
parser.add_argument('--n-proteins', '-n', required=False, default=1, 
                   help='Number of proteins per genome to select', type=int)
parser.add_argument('--min-score', '-s', required=False, default=100,
                   help='Minimum required bitscore', type=float)

def process_hmms(loc, 
                 n_proteins,
                 min_score,
                 write_file=True):

    # for each pfam table    
    count = 0

    hmm_files = sorted(glob.glob(os.path.join(loc, '**/*.tbl'), recursive=True))
    print(f'Found {len(hmm_files)} tables to parse')
    print(hmm_files[0], '...', hmm_files[-1])

    pfam_tables = []
    
    for count, file in enumerate(hmm_files):
    
        # counter

        print(f'{count+1} tables processed of {len(hmm_files)}', end='\r')
        
        # read table
        try:
            genome = file.split('/')[-1].split('_PF')[0]
            df = read_pfam_table_pandas(file)
            pfam_tables.append(df.assign(genome=genome))

            # get genome, top gene, top score
            genome = file.split('/')[-1].split('_PF')[0]
            
            filtered = df[df['full_score'] >= min_score]
            top_genes = filtered.sort_values('full_score', ascending=False).iloc[:n_proteins, 0].values
            scores = filtered.sort_values('full_score', ascending=False).iloc[:n_proteins, 5].values
            gene2score = dict((g,s) for g,s in zip(top_genes, scores))
    
            # read ORFs from original genome file
            if write_file:
                with open(file.replace('_hmmsearch.tbl', '_proteins.faa'), 'r') as handle:
                    records = [r for r in SeqIO.parse(handle, 'fasta') if r.id in top_genes]
                    
                print(len(records), len(top_genes))
                assert len(records) == len(top_genes)
                
                # rename records corresponding to adks
                for r in records:
                    r.seq = r.seq.replace('*', '')
                    new_id = '|'.join([r.id.split(' # ')[0], 
                                       genome, 
                                       'XXX'
                                       f'temp=XXX',
                                       f'bitscore={gene2score[r.id]}'])
                    r.id = new_id
                    r.description = new_id
                
                # write adk to a new file
                outfile = file.replace('_hmmsearch.tbl', '_adk.faa')
                if not os.path.exists(outfile):
                    with open(outfile, 'w') as out_handle:
                        SeqIO.write(records, out_handle, 'fasta')
        
        except ValueError:
            print(f'Table contained no records:\n{file}')
        
if __name__ == '__main__':
    args = parser.parse_args()
    process_hmms(args.loc, 
                 args.n_proteins,
                 args.min_score,
                 write_file=True)
    