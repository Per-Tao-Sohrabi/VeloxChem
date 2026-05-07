from methods import *

##########################################################################################    
#                                SET pathS & VARIABLES                                               
##########################################################################################    
PATH = 'tempfiles'  
UNIT_CELL_DIMENSIONS = (2,1,1)
GENERAL_FILE_NAME = f'1hx{UNIT_CELL_DIMENSIONS[0]}{UNIT_CELL_DIMENSIONS[1]}{UNIT_CELL_DIMENSIONS[2]}'
PDB_FILE = f'{PATH}/{GENERAL_FILE_NAME}.pdb'
DEFECT_PDB_FILE = f'{PATH}/defect_{GENERAL_FILE_NAME}.pdb'
PE_CUTOFF = 16.0

def ice(path=PATH, unit_cell_dimensions=UNIT_CELL_DIMENSIONS, general_file_name=GENERAL_FILE_NAME, pdb_file=PDB_FILE, defect_pdb_file=DEFECT_PDB_FILE, pe_cutoff=PE_CUTOFF):
    from ase.build import bulk
    from ase.spacegroup import crystal
    from ase.build import find_optimal_cell_shape
    from ase.build import make_supercell
    from ase.visualize.plot import plot_atoms
    from ase.io import read, write
    import os

    ##########################################################################################    
    #                                 

    ##########################################################################################    
    #                                   FIND QM REGION                                           
    ##########################################################################################    
    candidate_qm_water = get_centeroid_region(  # Read from PDB to maintain coordiate standards.
        filename=pdb_file, 
        cuboid_threshold = 0.2, 
        patterns={'WAT':['O', 'H', 'H']}
        )
    print(candidate_qm_water)

    ##########################################################################################    
    #                                      PROCESS PDB                                              
    ##########################################################################################    

    # RENAME RESIDUES
    process_pdb( 
        filename=pdb_file,
        patterns={'WAT': [' O', ' H', ' H']},
        qm_ids=candidate_qm_water,
        qm_resname='LIG'
    )

    # Unwrap minimum image convention.
    minimum_image_unwrap(pdb_file)

    ##########################################################################################    
    #                               GENERATE DEFECT & DEFECT PDB                                          
    ##########################################################################################    
    # defect_ice_block = ice_block
    # print(candidate_qm_water)
    indecies = candidate_qm_water['WAT']                # Arbitrarily select the first three atoms to be deleted
    target = indecies[:3]                               # to create a defect (missing water molecule).
    print(f'Target: {target}')
    defect, up_candidate_qm_water = del_atoms_pdb(      
        filename=pdb_file,
        output_filename=defect_pdb_file,
        delete_indecies=target,
        qm_resname='LIG'
    )

    print('ATOMS deleted')
    print(f'Defect {defect}')
    print(f'New qm candidate list {up_candidate_qm_water}')

    plot_atoms(read(filename=defect_pdb_file))

    ##########################################################################################    
    #                                 COMPUTE FORMATION ENERGY                                          
    ##########################################################################################    
    # Compute μ(H₂O) at same level of theory (HF/6-31G)
    mu_h2o = calc_chemical_potential_h2o(basis_set='6-31G') # The relative energy of the defect is calculated using the chemical potential of the missing species (H2O)

    # Compute formation energy
    E_f = calc_formation_energy( 
        filename_perf=pdb_file,
        filename_defect=defect_pdb_file,
        qm_resname='LIG',
        chemical_potentials = {'H2O': (1, mu_h2o)},  # Temporary variable implemented before I figure out how to generalize the chemcial potential generation. 
        pe_cutoff=16.0,
        npe_cutoff=None,
        charge_map=CHARGE_MAP
    )

if __name__ == '__main__':
    ice()
