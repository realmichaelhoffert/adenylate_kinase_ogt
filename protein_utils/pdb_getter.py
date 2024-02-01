import os
import subprocess

from IPython.display import display, clear_output, IFrame
import IPython

import nglview as nv
import pytraj as pt

from Bio.PDB import PDBParser, Select, PDBIO, Polypeptide
from Bio.SeqUtils import seq1

from Bio.PDB.DSSP import DSSP
# had to symlink a bunch of libboost shit to get dssp to work

from traitlets import traitlets

from ipywidgets import Box
from ipywidgets import Button, GridBox, Layout, ButtonStyle, Output
from jupyter_ui_poll import ui_events

import time

import numpy as np

# helper function to define lists of domains
def collapse_and_label(lst):
    
    result = []
    collapse_counts = dict((k, 1) for k in np.unique(lst))
    start = 0
    for i in range(1, len(lst)):
        if not lst[i] == lst[i - 1]:
            result.append(f'{lst[i-1]}{collapse_counts[lst[i-1]]}:{start}-{i}')
            collapse_counts[lst[i-1]] += 1
            start = i
    result.append(f'{lst[-1]}{collapse_counts[lst[i]]}:{start}-{i}')
    return result

class ResSelect(Select):
    def __init__(self, chain_id):
        self.chain = chain_id
        
    def accept_residue(self, res):
        # print(vars(res))
        if res.parent.id == self.chain and Polypeptide.is_aa(res.resname):
            return True
        else:
            return False

guiding_structures = {
    'E1':'core', 'E2':'core', 'E3':'core', 'E4':'core', 
    'H1':'core', 'H5':'core', 'H7':'core', 'H8':'core', 'H9':'core',
    'H6':'lid', 'B1':'lid', 'B2':'lid','B3':'lid','B4':'lid','B5':'lid',
    'H2':'nmp', 'H3':'nmp', 'H4':'nmp',
    'T1':'ploop'
}

# expected structure
# N - A3 - H1 - A5 - H2 - H3 - H4 - A4 - H5 - A2 - H6 - B1 - B2 - C1 - C2 - C3 - H7 - A1 - H8
# C - C  - C  - C  - N  - N  - N  - C  - C  - C  - L  - L  - L  - L  - L  - L  - C  - C  - C

class PDBEntry():
    '''
    id: PDB 4-char ID
    chain: chain, (A by default)
    fp: download fp for structure
    '''
    def __init__(self, id, chain, fp=''):
        self.uid = f'{id}_{chain}'
        self.protein = id
        self.chain = chain
        self.saved = False
        if fp != '':
            self.saved = True
            self.download_structure(fp, True)
            
        
        if self.saved:
            self.locate_domains()

    
        # get dssp
    

    def download_structure(self, download_path, verbose=True):
        verboseprint = print if verbose else lambda *a, **k: None
        try:
            if not os.path.exists(f'{download_path}/{self.protein}.pdb'):
                
                verboseprint('wgetting...')
                command = f'wget -P {download_path} https://files.rcsb.org/download/{self.protein}.pdb >/dev/null 2>&1'
                # verboseprint(command)
            
                download_info = subprocess.check_output(command, shell=True)
            else:
                verboseprint('Already downloaded')
            
            # first, get complete protein
            parser = PDBParser()
            structure = parser.get_structure(f"{self.protein}", f"{download_path}{self.protein}.pdb")
            self.full_structure_loc = f"{download_path}{self.protein}.pdb"
            self.full_structure = structure
            # then, subset to matching chain with Bio.PDB.Select
            verboseprint('parsed')

        except subprocess.CalledProcessError:
            print('Could not download!')

    def save_chain_structure(self, path):
        
        io = PDBIO()
        io.set_structure(self.full_structure)
        io.save(f"{path}{self.protein}_{self.chain}.pdb", ResSelect(self.chain))
        self.chain_structure_loc = f"{path}{self.protein}_{self.chain}.pdb"

    def save_chain_pdb_sequence(self, path, verbose=True):
    
        # write chains for annotation
        chains = {chain.id:seq1(''.join(residue.resname for residue in chain if Polypeptide.is_aa(residue.resname))) for chain in self.full_structure.get_chains()}
        with open(f'{path}{self.protein}_chains.faa', 'w') as faa_handle:
            for key, item in chains.items():
                faa_handle.write(f'>{self.protein}_chain_{key}|length={len(item)}\n')
                faa_handle.write(item + '\n')

    def locate_domains(self):
        self.dssp = DSSP(self.full_structure[0], self.full_structure_loc)

        # extract annotation of each residue from dssp        
        structures = []
        for key in self.dssp.keys():
            if key[0] == self.chain:
                ind = key[1][1]
                aa, sec = self.dssp[key][1:3]
                structures.append(sec)

        struct2res = dict(tuple(k.split(':')[::-1]) for k in collapse_and_label(structures))
        structure_locs = []
        for key, item in struct2res.items():
            indeces = [int(k) for k in key.split('-')]
            structure_locs += [item] * len(list(range(indeces[0], indeces[1])))
        
        structure_annot = []
        current='none'
        for item in structure_locs:
            if item in guiding_structures.keys():
                
                current = guiding_structures[item]
        
            structure_annot.append(current)
            
        self.residue_domains = structure_annot
        total = len(self.residue_domains)
        self.nmp_start = self.residue_domains.index('nmp')
        self.nmp_end = total - self.residue_domains[::-1].index('nmp')
        self.lid_start = self.residue_domains.index('lid')
        self.lid_end = total - self.residue_domains[::-1].index('lid')


    def draw(self):
        domain_cmap = {'core':'red', 'lid':'blue', 'nmp':'green', 'ploop':'yellow', 'none':'gray'}

        colors = [] 
        for i, s in enumerate(self.residue_domains):
            colors.append([domain_cmap[s], str(i)])
        
        traj = pt.load(self.chain_structure_loc)
        w = nv.show_pytraj(traj, default_representation=False)
        w.add_cartoon(color='gray')

        scheme = nv.color._ColorScheme(colors, label='domains')
        
        w.add_cartoon(color=scheme)
        
        w.center()
        return w

    def assess_conformation(self, manual=False, draw=True):
        traj = pt.load(self.chain_structure_loc)
        pytraj_nmp = f':{self.nmp_start}-{self.nmp_end}'
        pytraj_lid = f':{self.lid_start}-{self.lid_end}'
        distance = pt.distance(traj, f'{pytraj_nmp} {pytraj_lid}')
        self.nmp_lid_dist = distance[0]
        if distance > 25:
            self.conformation = 'open'
        else:
            self.conformation = 'closed'
        self.conformation_method = 'auto'
        
        if draw:
            w = self.draw()

            # adding arrow
            p1 = list(pt.center_of_mass(traj, pytraj_nmp)[0])
            p2 = list(pt.center_of_mass(traj, pytraj_lid)[0])
            w.shape.add_arrow(p1, p2, [1,0,0], 1.0)
            
            if manual:    
                # button for actions
                open_butt = LoadedButton(description='open',
                                 layout=Layout(width='auto', grid_area='open_butt'),
                                 style=ButtonStyle(button_color='moccasin'))
                closed_butt = LoadedButton(description='closed',
                                 layout=Layout(width='auto', grid_area='closed_butt'),
                                 style=ButtonStyle(button_color='salmon'))
                
                # output widget type
                output = Output()
                # display all at once
                display(w, open_butt, closed_butt)
    
                open_butt.value = 0
                closed_butt.value = 0
                open_butt.on_click(add_num)
                closed_butt.on_click(add_num)
                
                # poll ui events to wait for click
                with ui_events() as poll:
                    while open_butt.value == 0 and closed_butt.value == 0:
                        poll(10) # poll queued UI events including button
                        time.sleep(0.5) # wait for 1 second before checking again
                if open_butt.value > 0:
                    self.conformation = 'open'
                else:
                    self.conformation = 'closed'
                self.conformation_method = 'manual'
            else:
                display(w)

class LoadedButton(Button):
    """A button that can holds a value as a attribute."""

    def __init__(self, value=None, *args, **kwargs):
        super(LoadedButton, self).__init__(*args, **kwargs)
        # Create the value attribute.
        self.add_traits(value=traitlets.Any(value))

def add_num(ex):
    ex.value = ex.value+1