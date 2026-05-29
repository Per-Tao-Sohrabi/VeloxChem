from methods import *

PATH = 'benzene' 
GENERAL_FILE_NAME = 'benzene'
PDB_FILE = f'{PATH}/{GENERAL_FILE_NAME}.pdb'
CIF_FILE = f'{PATH}/{GENERAL_FILE_NAME}.cif'

DEFECT_PDB_FILE = f'{PATH}/defect_{GENERAL_FILE_NAME}.pdb'

TARGET_SIZE = 50
TARGET_SHAPE = 'sc'

# Change to make user input a molecule object. 
GEOMETRIES = {
    'H2O': "3\n\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.469\nH 0.0 -0.757 -0.469\n",
    'Na':  "1\n\nNa 0.0 0.0 0.0\n",
    'Cl':  "1\n\nCl 0.0 0.0 0.0\n",
    'NaCl': "2\n\nNa 0.0 0.0 0.0\nCl 2.36 0.0 0.0\n",
    'C6H6': "12\n\nC 0.0000 1.3975 0.0000\nH 0.0000 2.4839 0.0000\nC -1.2094 0.6987 0.0000\nH -2.1619 1.2419 0.0000\nC -1.2094 -0.6987 0.0000\nH -2.1619 -1.2419 0.0000\nC 0.0000 -1.3975 0.0000\nH 0.0000 -2.4839 0.0000\nC 1.2094 -0.6987 0.0000\nH 2.1619 -1.2419 0.0000\nC 1.2094 0.6987 0.0000\nH 2.1619 1.2419 0.0000\n"
}

CHARGE_MAP = {'O': -2, 'C': 0, 'H': 0, 'Na': 1, 'Cl': -1}  

PE_CUTOFF = 16.0
CUBOID_THRESHOLD = 0.15
BASIS_SET = '6-31G**'
XC_FUNCTIONAL = 'B3LYP'
DISPERSION = True

DEBUG = True

def benzene(
    path=PATH, 
    general_file_name=GENERAL_FILE_NAME, 
    pdb_file=PDB_FILE, 
    cif_file=CIF_FILE, 
    defect_pdb_file=DEFECT_PDB_FILE, 
    target_size=TARGET_SIZE, 
    target_shape=TARGET_SHAPE, 
    pe_cutoff=PE_CUTOFF, 
    cuboid_threshold=CUBOID_THRESHOLD, 
    basis_set=BASIS_SET, 
    xc_functional=XC_FUNCTIONAL, 
    dispersion=DISPERSION, 
    debug=DEBUG, 
    charge_map=CHARGE_MAP
):
    # pyrefly: ignore [missing-import]
    from ase.build import bulk
    from ase.build import find_optimal_cell_shape
    from ase.build import make_supercell
    from ase.spacegroup import crystal
    from ase.io import read, write
    from ase.visualize.plot import plot_atoms
    import os

    ##########################################################################################    
    #                                Generate Benzene Supercell                                               
    ##########################################################################################    
    unit_cell = read(cif_file)
    # plot_atoms(unit_cell)

    P = find_optimal_cell_shape(
            cell=unit_cell.cell, 
            target_size=target_size,
            target_shape=target_shape
        )

    supercell = make_supercell(unit_cell, P)
    plot_atoms(supercell)

    write(f'{pdb_file}', supercell)

    collections = identify_connectivity_pdb( # Joins atoms into molecules. 
        filename=pdb_file,
        bonds_length=[('C', 'C', 1.6), ('C', 'H', 1.2)],
        debug=debug
    )

    qm_ids = process_pdb_advanced( # Assigns qm sites. 
        filename = pdb_file,
        mol_residues = {'BEN': [collections, 12]},
        qm_resname = 'LIG',
        qm_threshold = 0.1,
        debug=debug,
        clear_unmatched = True
    )

    defect, up_candidate_qm = del_atoms_pdb( # Removes unassigned atoms. 
        filename=pdb_file,
        output_filename=defect_pdb_file,
        delete_indecies=qm_ids[0], # Remove the first index collection. 
        qm_resname='LIG'
    )

    print(f'DEBUG: deleting atoms at indices: {qm_ids[0]}')
    print(f'DEBUG: Updated qm list {up_candidate_qm}')
    print(f'DEBUG: Removed atoms {defect}')
    print(f'DEBUG: atoms removed count: {len(defect)}')

    # plot_atoms(read(filename=defect_pdb_file))

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
        basis_set=basis_set, 
        geometries={'CUSTOM': defect_xyz},
        dispersion = dispersion,
        xcfun = xc_functional,
    )

    # Compute formation energy
    E_f = calc_formation_energy( 
        filename_perf=pdb_file,
        filename_defect=defect_pdb_file,
        qm_resname='LIG',
        chemical_potentials = {'BEN': (1, mu)},
        dispersion = dispersion,
        debug=debug,
        pe_cutoff=pe_cutoff,
        charge_map=charge_map,
        xcfun = xc_functional,
        basis_set = basis_set
    )
    
    return E_f

if __name__ == '__main__':
    benzene(
            basis_set='6-31G**', 
            cuboid_threshold=0.15, 
            target_size=30, 
            target_shape='sc',
            pe_cutoff=12, 
            dispersion=True, 
            xc_functional='B3LYP', 
            debug=True
        )