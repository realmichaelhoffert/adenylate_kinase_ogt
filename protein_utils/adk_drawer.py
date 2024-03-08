import IPython as IPython

import nglview as nv
import pytraj as pt

from scipy.stats import spearmanr

import sys
sys.path.append('/data/mhoffert/fiererlab/adenylate_kinase_ogt/protein_utils/')

from pdb_getter import collapse_and_label

from collections import Counter

# expected structure
# N - A3 - H1 - A5 - H2 - H3 - H4 - A4 - H5 - A2 - H6 - B1 - B2 - C1 - C2 - C3 - H7 - A1 - H8
# C - C  - C  - C  - N  - N  - N  - C  - C  - C  - L  - L  - L  - L  - L  - L  - C  - C  - C

def get_guiding_structures():
    return {
    'E1':'core', 'E2':'core', 'E3':'core', 'E4':'core', 
    'H1':'core', 'H5':'core', 'H7':'core', 'H8':'core', 'H9':'core',
    'H6':'lid', 'B1':'lid', 'B2':'lid','B3':'lid','B4':'lid','B5':'lid',
    'H2':'nmp', 'H3':'nmp', 'H4':'nmp',
    'T1':'ploop','L2':'ploop'
    }


def nglview_list(cmap, residue_ss):
    return [[cmap[r], str(i)] for i, r in enumerate(residue_ss)]

    
def ss_to_domains(ss):
    # transform to list
    structures = [s for s in ss]
    
    # get a dictionary: {'res_start - res_end':'ss_label'}
    # where ss_label = E1 if ss is the first sheet, H4 if ss is the fourth helix, etc.
    struct2res = dict(tuple(k.split(':')[::-1]) for k in collapse_and_label(structures))

    # iterate and make a new list:
    # ['E1', 'E1', 'E1', 'H2', 'H2', 'H2'], etc.
    structure_locs = []
    for key, item in struct2res.items():
        indeces = [int(k) for k in key.split('-')]
        structure_locs += [item] * len(list(range(indeces[0], indeces[1])))

    guiding_structures = get_guiding_structures()
    
    # map E1, etc. to domains in adk
    structure_annot = []
    current='none'
    for item in structure_locs:
        if item in guiding_structures.keys():
            
            current = guiding_structures[item]
    
        structure_annot.append(current)
        
    return struct2res, structure_annot

def draw_adk(pdb, color_ss=[], color_domains=[], ss='', seqfile=None):
    '''
    pdb: filepath to pdb file
    color_ss: cmap of secondary structure colors, if desired
    color_domains: cmap of domain colors, if desired
    ss = string of secondary structures from rosetta
    
    '''
    # load pdb
    traj = pt.load(pdb)
    w = nv.show_pytraj(traj, default_representation=False)
    w.add_cartoon(color='gray')
    
    if len(color_ss) > 0:
        if len(ss) == 0:
            raise ValueError('color_ss=True means ss must be supplied')
        
        colors = nglview_list(color_ss, ss)
        
        scheme = nv.color._ColorScheme(colors, label='secondary structure')
    
        w.add_cartoon(color=scheme)
    
    if len(color_domains) > 0:
        if len(ss) == 0:
            raise ValueError('domain_ss=True means ss must be supplied')

        s2r, sa = ss_to_domains(ss)
        colors = nglview_list(color_domains, sa)
        
        scheme = nv.color._ColorScheme(colors, label='domains')
    
        w.add_cartoon(color=scheme)
    
    w.center()
    return w