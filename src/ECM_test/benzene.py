from methods import *
# pyrefly: ignore [missing-import]
from ase.build import bulk
from ase.build import find_optimal_cell_shape
from ase.build import make_supercell
from ase.spacegroup import crystal
from ase.io import read, write
from ase.visualize.plot import plot_atoms
import os


PATH = 'benzene' 
# if not os.path.exists(PATH):
#     os.makedirs(PATH)
GENERAL_FILE_NAME = 'benzene'
PDB_FILE = f'{PATH}/{GENERAL_FILE_NAME}.pdb'
CIF_FILE = f'{PATH}/{GENERAL_FILE_NAME}.cif'

DEFECT_PDB_FILE = f'{PATH}/defect_{GENERAL_FILE_NAME}.pdb'

PE_CUTOFF = 16.0

# Change to make user input a molecule object. 
GEOMETRIES = {
    'H2O': "3\n\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.469\nH 0.0 -0.757 -0.469\n",
    'Na':  "1\n\nNa 0.0 0.0 0.0\n",
    'Cl':  "1\n\nCl 0.0 0.0 0.0\n",
    'NaCl': "2\n\nNa 0.0 0.0 0.0\nCl 2.36 0.0 0.0\n",
    'C6H6': "12\n\nC 0.0000 1.3975 0.0000\nH 0.0000 2.4839 0.0000\nC -1.2094 0.6987 0.0000\nH -2.1619 1.2419 0.0000\nC -1.2094 -0.6987 0.0000\nH -2.1619 -1.2419 0.0000\nC 0.0000 -1.3975 0.0000\nH 0.0000 -2.4839 0.0000\nC 1.2094 -0.6987 0.0000\nH 2.1619 -1.2419 0.0000\nC 1.2094 0.6987 0.0000\nH 2.1619 1.2419 0.0000\n"
}

CHARGE_MAP = {'O': -2, 'C': 0, 'H': 0, 'Na': 1, 'Cl': -1}  

TARGET_SIZE = 50
TARGET_SHAPE = 'sc'

CUBOID_THRESHOLD = 0.3

_pattern = ['C']*24 + ['H']*24
print(f'DEBUG: _pattern {_pattern}')

PATTERN = {'BEN': _pattern}

##########################################################################################    
#                                Generate Ice Supercell                                               
##########################################################################################    

unit_cell = read(CIF_FILE)
# plot_atoms(unit_cell)

P = find_optimal_cell_shape(
        cell=unit_cell.cell, 
        target_size=TARGET_SIZE,
        target_shape=TARGET_SHAPE
    )

supercell = make_supercell(unit_cell, P)
plot_atoms(supercell)

write(f'{PDB_FILE}', supercell)

# qm_ids = get_centeroid_region(
#             filename=PDB_FILE,
#             patterns=PATTERN,
#             cuboid_threshold=CUBOID_THRESHOLD,
#             debug = True
#         )

#print(f'DEBUG: qm_ids {qm_ids}')

# NOTE: Draft.
collections = identify_connectivity_pdb(
    filename=PDB_FILE,
    bonds_length=[('C', 'C', 1.6), ('C', 'H', 1.2)],
    debug=False
)
# print(f'DEBUG: collections {collections}')

qm_ids = process_pdb_advanced(
    filename = PDB_FILE,
    mol_residues = {'BEN': [collections, 12]},
    qm_resname = 'LIG',
    qm_threshold = 0.1,
    debug=True,
    clear_unmatched = True
)

# process_pdb(
#     filename=PDB_FILE,
#     patterns=PATTERN,
#     qm_ids=qm_ids,
#     qm_resname='LIG'
# )

# minimum_image_unwrap(filename=PDB_FILE)

defect, up_candidate_qm = del_atoms_pdb(      
    filename=PDB_FILE,
    output_filename=DEFECT_PDB_FILE,
    delete_indecies=qm_ids[0], # Remove the first index collection. 
    qm_resname='LIG'
)

print(f'DEBUG: deleting atoms at indices: {qm_ids[0]}')
print(f'DEBUG: Updated qm list {up_candidate_qm}')
print(f'DEBUG: Removed atoms {defect}')
print(f'DEBUG: atoms removed count: {len(defect)}')

# plot_atoms(read(filename=DEFECT_PDB_FILE))

# ##########################################################################################    
# #                                 COMPUTE FORMATION ENERGY                                          
# ##########################################################################################    
# Construct XYZ geometry string directly from the extracted unrelaxed atoms (Option A)
n_atoms = len(defect)
defect_xyz = f"{n_atoms}\n\n"
for symbol, pos in zip(defect.get_chemical_symbols(), defect.get_positions()):
    defect_xyz += f"{symbol} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n"

print("DEBUG: Derived real chemical potential geometry from extracted defect atoms.")

# Compute μ(C6H6) using the EXACT same unrelaxed geometry found inside the supercell
mu = calc_chemical_potential(
    species='CUSTOM', 
    basis_set='6-31G', 
    geometries={'CUSTOM': defect_xyz},
    dispersion = True
)

# Compute formation energy
E_f = calc_formation_energy( 
    filename_perf=PDB_FILE,
    filename_defect=DEFECT_PDB_FILE,
    qm_resname='LIG',
    chemical_potentials = {'BEN': (1, mu)},
    dispersion = True,
    debug=True,
    pe_cutoff=PE_CUTOFF,
    charge_map=CHARGE_MAP
)

# E_f.plot_band_diagram()