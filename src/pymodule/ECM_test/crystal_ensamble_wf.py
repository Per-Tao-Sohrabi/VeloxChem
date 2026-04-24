from ase.io import read, iread, write
from ase.visualize.plot import plot_atoms
from ase.build import find_optimal_cell_shape
from ase.build import make_supercell
import os


def generate_ice_cell(size_coord = (1,1,1), path: str = None, general_filename: str = None, to_pdb: bool = True, plot_atoms: bool = False):
    if general_filename is None:
        general_filename = f'ihx{size_coord[0]}_{size_coord[1]}_{size_coord[2]}'
    if path is None:
        path = '../../.crystal_structs'

    os.system(f'touch {path}/{general_filename}.cif')
    os.system(f'genice2 --rep {size_coord[0]} {size_coord[1]} {size_coord[2]} 1h --format cif > {path}/{general_filename}.cif')
    print (f'Generated ice block with size {size_coord} at {path}/{general_filename}.cif')
    
    if to_pdb:
        os.system(f'ase convert {path}/{general_filename}.cif {path}/{general_filename}.pdb')
        print (f'Converted ice block to pdb at {path}/{general_filename}.pdb')
    
    ice = read(f'{path}/{general_filename}.cif')
    plot_atoms(ice, rotation='60x,60y,50z')
    return ice

# Make supercell, find transition matrix

def generate_ice_supercell(ice_cell,  path: str = None, general_filename: str = None, target_size: int = 10, target_shape: str = 'sc', plot_atoms: bool = False, to_pdb: bool = True):
    if general_filename is None:
        general_filename = f'ihx{size_coord[0]}_{size_coord[1]}_{size_coord[2]}'
    if path is None:
        path = '../../.structures'
    P = find_optimal_cell_shape(
        cell=ice_cell.cell,
        target_size=target_size,
        target_shape=target_shape
        ) 
    ice_block = make_supercell(ice_cell, P)

    if plot_atoms:
        plot_atoms(ice_block, rotation='60x,60y,50z')
        plot_atoms(ice_block, rotation='60x,60y,70z')
        plot_atoms(ice_block, rotation='60x,60y,90z')
    if to_pdb:
        os.system(f'touch {path}/{general_filename}_supercell.cif')
        os.system(f'ase convert {path}/{general_filename}_supercell.cif {path}/{general_filename}_supercell.pdb')
        print (f'Converted ice block to pdb at {path}/{general_filename}_supercell.pdb')
    return ice_block

def get_pdb_index(pdb_file: str):
    pdb_content = read(pdb_file)
    

# Using ensemble driver
import veloxchem.ensembleparser
import veloxchem.ensembledriver

RESNAME_QM_REGION = 'LIG'
RESNAME_PE_REGION = 'HOH'

PATH = '../../.crystal_structs'
GENERAL_FILENAME = 'ihx222'

ice = generate_ice_cell(size_coord=(2,2,2), path=PATH, general_filename=GENERAL_FILENAME, to_pdb=True, plot_atoms=True)
ice_block = generate_ice_supercell(ice, path=PATH, general_filename=GENERAL_FILENAME, target_size=10, target_shape='sc', plot_atoms=True, to_pdb=True)

# Update pdb 




