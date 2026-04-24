from methods import *
import os
from ase.io import read, write
from ase.build import find_optimal_cell_shape
from ase.build import make_supercell
from ase.visualize.plot import plot_atoms
import numpy as np
import veloxchem.ensembledriver
import veloxchem.ensembleparser

UNIT_CELL_DIMENSIONS = (3,3,3)
SUPERCELL_SIZE = 12 # Nr. Atoms
SUPERCELL_SHAPE = 'sc'

PATH = 'tempfiles'
GENERAL_FILE_NAME = f'1hx{UNIT_CELL_DIMENSIONS[0]}{UNIT_CELL_DIMENSIONS[1]}{UNIT_CELL_DIMENSIONS[2]}'
# SUPERCELL_FILE_NAME = f'{GENERAL_FILE_NAME}_supercell'

# PROCESSING PDB FILE
ATOMS_PER_MOL = 3
GLOBAL_RESNAME = 'WAT'
QM_RESNAME = 'LIG'

# DEFECT SPECIFICATIONS
DEL_INDECIES = [148,149,150]
DEFECT_PDB_FILE = f'{PATH}/defect_{GENERAL_FILE_NAME}.pdb'



# =======================================================================================================================
# =================================================ICE 1H CRYSTAL========================================================
# =======================================================================================================================

# Create tempfiles directory if it doesn't exist
os.system(f'mkdir {PATH}')

os.system(f'genice2 --rep {UNIT_CELL_DIMENSIONS[0]} {UNIT_CELL_DIMENSIONS[1]} {UNIT_CELL_DIMENSIONS[2]} 1h --format cif > {PATH}/{GENERAL_FILE_NAME}.cif')
atoms = read(f'{PATH}/{GENERAL_FILE_NAME}.cif')

PDB_FILE = f'{PATH}/{GENERAL_FILE_NAME}.pdb'
write(f'{PDB_FILE}', atoms)

candidate_qm_water = get_centeroid_region(print_status=False, filename=PDB_FILE, cuboid_threshold = 0.2)

process_pdb(
    filename=PDB_FILE,
    atoms_per_mol=ATOMS_PER_MOL,
    target_filename=PDB_FILE,
    global_resname=GLOBAL_RESNAME,
    qm_resname=QM_RESNAME,
    qm_indecies=candidate_qm_water
)

# scf_results = calc_energy(filename=PDB_FILE, qm_resname=QM_RESNAME, pe_cutoff=6.0, npe_cutoff=None)
# print(scf_results)

# del_atoms_pdb(
#     filename=PDB_FILE,
#     target_filename=DEFECT_PDB_FILE,
#     del_indecies=DEL_INDECIES
# )

# defect_scf_results = calc_energy(filename=DEFECT_PDB_FILE, qm_resname=QM_RESNAME, pe_cutoff=6.0, npe_cutoff=None)
# print(defect_scf_results)


