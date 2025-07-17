import os
import glob
import argparse

import pandas as pd
from Bio import SeqIO

from protein_utils import read_pfam_table_pandas, pfam_table_columns

parser = argparse.ArgumentParser()
parser.add_argument('--loc', '-l', required=True,
                   help='Location of files')

def process_hmms(loc):

    # for each pfam table
    WRITE_FILE = False
    count = 0

    pfam_tables = glob.glob(os.path.join(loc, '*.tblout'))
    
    for file in pfam_tables:
    
        # counter
        if count % 100 == 0:
            display(count)
            clear_output(wait=True)
        # read table
        try:
            genome = file.split('/')[-1].split('_PF')[0]
            df = read_pfam_table_pandas(file)
            all_adk_pfam_tables.append(df.assign(genome=genome))
            
        except ValueError:
            print(f'Table contained no records:\n{file}')
        
        # get genome, top gene, top score
        genome = file.split('/')[-1].split('_PF')[0]
        top_gene = df.sort_values('full_score', ascending=False).iloc[0, 0]
        bitscore = df.sort_values('full_score', ascending=False).iloc[0, 5]
    
        # read ORFs from original genome file
        if WRITE_FILE:
            with open(file.replace('_hmmsearch.tbl', '_protein.faa'), 'r') as handle:
                records = [r for r in SeqIO.parse(handle, 'fasta') if top_gene == r.id]
                
            # rename records corresponding to adks
            for r in records:
                r.seq = r.seq.replace('*', '')
                new_id = '|'.join([r.id.split(' # ')[0], 
                                   genome, 
                                   'XXX'
                                   f'temp=XXX',
                                   f'bitscore={bitscore}'])
                r.id = new_id
                r.description = new_id
            
            # write adk to a new file
            outfile = file.replace('_hmmsearch.tbl', '_adk.faa')
            if not os.path.exists(outfile):
                with open(outfile, 'w') as out_handle:
                    SeqIO.write(records, out_handle, 'fasta')
        
        count += 1

if __name__ == '__main__':
    