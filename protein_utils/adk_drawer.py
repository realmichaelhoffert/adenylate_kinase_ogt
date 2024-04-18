import IPython as IPython

import nglview as nv
import pytraj as pt

from scipy.stats import spearmanr

import sys
sys.path.append('/data/mhoffert/fiererlab/adenylate_kinase_ogt/protein_utils/')

from pdb_getter import collapse_and_label

from collections import Counter

import numpy as np

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


def nw(x, y, match = 1, mismatch = 1, gap = 1):
    nx = len(x)
    ny = len(y)
    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:,0] = np.linspace(0, -nx * gap, nx + 1)
    F[0,:] = np.linspace(0, -ny * gap, ny + 1)
    # Pointers to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:,0] = 3
    P[0,:] = 4
    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j]:
                t[0] = F[i,j] + match
            else:
                t[0] = F[i,j] - mismatch
            t[1] = F[i,j+1] - gap
            t[2] = F[i+1,j] - gap
            tmax = np.max(t)
            F[i+1,j+1] = tmax
            if t[0] == tmax:
                P[i+1,j+1] += 2
            if t[1] == tmax:
                P[i+1,j+1] += 3
            if t[2] == tmax:
                P[i+1,j+1] += 4
    # Trace through an optimal alignment.
    i = nx
    j = ny
    rx = []
    ry = []
    while i > 0 or j > 0:
        if P[i,j] in [2, 5, 6, 9]:
            rx.append(x[i-1])
            ry.append(y[j-1])
            i -= 1
            j -= 1
        elif P[i,j] in [3, 5, 7, 9]:
            rx.append(x[i-1])
            ry.append('-')
            i -= 1
        elif P[i,j] in [4, 6, 7, 9]:
            rx.append('-')
            ry.append(y[j-1])
            j -= 1
    # Reverse the strings.
    rx = ''.join(rx)[::-1]
    ry = ''.join(ry)[::-1]
    return '\n'.join([rx, ry])


def compute_domains(ss, m=1, mm=1, g=1, verbose=False):

    ## Test filtering on domain structure
    # old
    # 'CCCCNNNCCCLLLLLLCCC'
    # 'EHEHHHEHEHEEEEEEHEH'
    if len(ss) > 200:
        canonical_dom =        'CCPCCCNNNNNCCCCCCCCLLLLLLLLLLLLLLCCCCCC'
        canonical_structures = 'LELHLEHLHLHLLELHLELHLELELELELELELHLELHL'
    else:
        canonical_dom =        'CCPCCCNNNNNCCCCCCCCLLLLLCCCCCC'
        canonical_structures = 'LELHLEHLHLHLLELHLELHLELLHLELHL'
        
    structure = ''.join([item[0] for key, item in ss_to_domains(ss)[0].items()])
    if verbose:
        print(structure)
        print(len(structure))
    alignment = nw(structure, canonical_structures, match=m, mismatch=mm, gap=g)
    text = alignment.split('\n')[1]
    new_can_dom = canonical_dom
    for ind in [n for n in range(len(text)) if text.find('-', n) == n]:
        new_can_dom = new_can_dom[:ind] + new_can_dom[ind-1] + new_can_dom[ind:]

    if verbose:
        print('\n'.join([alignment, canonical_dom, new_can_dom]))
    
    ss2dom = list(zip(alignment.split('\n')[0], new_can_dom))

    temp_dict = {}
    map_dict = {}
    for item in [i for i in ss2dom if not '-' in i[0]]:
        if item[0] in temp_dict.keys():
            temp_dict[item[0]] += 1
        else:
            temp_dict[item[0]] = 1
    
        map_dict[f'{item[0]}{temp_dict[item[0]]}'] = item[1]

    r2s = ss_to_domains(ss)[0]

    structure_loc = []
    for key, item in r2s.items():
        indeces = [int(k) for k in key.split('-')]
        structure_loc += [map_dict[item]] * len(list(range(indeces[0], indeces[1])))

    return structure_loc