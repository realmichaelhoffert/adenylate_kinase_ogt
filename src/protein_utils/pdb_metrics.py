import os
import pandas as pd
import numpy as np
import Bio.PDB
from Bio.PDB import SASA

import pandas as pd

def process_structure(structure_file : str, uid : str, write_processed = False, savepath='', calculate_local=False):
    io=Bio.PDB.PDBIO()
    parser = Bio.PDB.PDBParser()
    # get structure
    struct = parser.get_structure(uid, structure_file)

    temp_df = pd.DataFrame()
    
    # remove hydrogens
    #for atom in Bio.PDB.Selection.unfold_entities(struct, 'A'):
        #if atom.get_name()[0] == 'H':
            #Bio.PDB.Selection.unfold_entities(atom, 'R')[0].__delitem__(atom.id)

    # read in queryable protein
    queryable_protein = []
    for res in Bio.PDB.Selection.unfold_entities(struct, 'R'):
        if Bio.PDB.Polypeptide.is_aa(res, standard=True):
            queryable_protein.append(res)
    
    # iteratively remove residues at the termini with low pLDDT
    res_to_remove = []
    counter = 0
    res = queryable_protein[counter]
    while res['CA'].get_bfactor() < 70: # and counter < 80:
        res_to_remove.append(res)
        counter += 1
        res = queryable_protein[counter]
        
    # iterate in other direction
    counter = 1
    res = queryable_protein[-counter]
    while res['CA'].get_bfactor() < 70: # and counter < 80:
        res_to_remove.append(res)
        counter += 1
        res = queryable_protein[-counter]
        
    # remove
    for res in res_to_remove:
        #struct[0]['A'].__delitem__(res.id)
        Bio.PDB.Selection.unfold_entities(res, 'C')[0].__delitem__(res.id)
    
    # build queryable protein again
    queryable_protein = []
    for res in Bio.PDB.Selection.unfold_entities(struct, 'R'):
        if Bio.PDB.Polypeptide.is_aa(res, standard=True):
            queryable_protein.append(res)

    # identify residues that are present, should be present, are not present, and are adjacent to residues that are not present
    # useful to avoid errors when running code that doesn't like missing residues or neighbors
    # not necessary for AF models but if we want to compare AF structures to crystal structures at any point it'll be handy
    residues_present = sorted([res.id[1] for res in queryable_protein])
    residues_range = list(range(residues_present[0], residues_present[-1] + 1))
    residues_absent = set(residues_range).difference(residues_present)
    residues_missing_nbrs = residues_absent.union({residues_range[0],residues_range[-1]},{resi+1 for resi in residues_absent},{resi-1 for resi in residues_absent})
    
    # set residue positions as indices for df
    temp_df['Position'] = residues_range
    temp_df = temp_df.set_index('Position')        
        
    for chain in Bio.PDB.Selection.unfold_entities(list(queryable_protein), 'C'):
            if chain.id == 'A':
                temp_df = temp_df.copy()
                temp_df['Position'] = temp_df.index
                #temp_df['chain'] = chain.id
    
                # calculate contact density
                contact_cutoffs = [4.5]
                nbr_search = Bio.PDB.NeighborSearch(Bio.PDB.Selection.unfold_entities(chain, 'A'), bucket_size=1000)
                for cutoff in contact_cutoffs:
                    contacts = nbr_search.search_all(cutoff, 'R')
                    contact_count_dict = {resi: 0 if resi not in residues_absent else np.NaN for resi in temp_df['Position']}
                    for (res1, res2) in contacts:
                        contact_count_dict[res1.id[1]] += 1
                        contact_count_dict[res2.id[1]] += 1
                    temp_df[f'Contacts at {cutoff} Å'] = temp_df['Position'].map(contact_count_dict)
    
                # solvent exposure - Shrake-Rupley algorithm ("rolling ball" method)
                sr = SASA.ShrakeRupley(probe_radius=1.40, n_points=1000)
                sr.compute(struct[0], level="R") # note that this omits burial due to bound ligands or contacts with other chains    
                temp_df['SASA'] = temp_df['Position'].map(lambda pos: chain[pos].sasa if not pos in residues_absent else np.NaN)
                
                #temp_df_m = temp_df.mean(numeric_only=True)
                            
                #temp_df_m['uid'] = uid
                # df_collector.append(temp_df_m)
                temp_df['uid'] = uid
                
                # save files with trimmed termini for analysis with rosetta
                if write_processed:

                    # remove hydrogens?
                    if False:
                        for atom in Bio.PDB.Selection.unfold_entities(struct, 'A'):
                            if atom.get_name()[0] == 'H':
                                Bio.PDB.Selection.unfold_entities(atom, 'R')[0].__delitem__(atom.id)
                    filename = struct.id + '_' + chain.id + '_processed' + '.pdb'
                    io.set_structure(struct)
                    io.save(os.path.join(savepath, filename))

    return temp_df
    #return temp_df_m