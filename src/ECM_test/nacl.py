from methods import *

##########################################################################################    
#                                SET pathS & VARIABLES                                               
##########################################################################################  
PATH = 'tempfiles'
# . . . 
GENERAL_FILE_NAME_NACL = 'nacl'
DEFECT_FILE_NAME_NACL = f'defect_{GENERAL_FILE_NAME_NACL}'
CIF_FILE_NACL = f'{PATH}/{GENERAL_FILE_NAME_NACL}.cif'
PDB_FILE_NACL = f'{PATH}/{GENERAL_FILE_NAME_NACL}.pdb'
DEFECT_PDB_NACL = f'{PATH}/{DEFECT_FILE_NAME_NACL}.pdb'

TARGET_SIZE = 50
PE_CUTOFF = 16.0

CHARGE_MAP = {'O': 2, 'C': 0, 'H': -1, 'Na': 1, 'Cl': -1}  
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
}

def nacl(path=PATH, general_file_name_nacl=GENERAL_FILE_NAME_NACL, defect_file_name_nacl=DEFECT_FILE_NAME_NACL, cif_file_nacl=CIF_FILE_NACL, pdb_file_nacl=PDB_FILE_NACL, defect_pdb_nacl=DEFECT_PDB_NACL, target_size=TARGET_SIZE, pe_cutoff=PE_CUTOFF, charge_map=CHARGE_MAP, atomic_numbers=ATOMIC_NUMBERS):
    from ase.build import bulk
    from ase.spacegroup import crystal
    from ase.build import find_optimal_cell_shape
    from ase.build import make_supercell
    from ase.visualize.plot import plot_atoms
    from ase.io import read, write
    import os

    print(f"{'#'*100}")
    print(f"{'Starting NaCl defect calculations'}")
    print(f"{'#'*100}")

    #from scipy._lib.cobyqa.subsolvers.optim import qr_normal_byrd_omojokun

    # nacl = crystal(
    #     symbols='NaCl', 
    #     basis=[[0,0,0]],#, [0,1,0]],
    #     spacegroup=227,
    #     cellpar=[1,1,1, 90, 90, 90],
    #     size=(2,2,2)
    #     ))

    ##########################################################################################    
    #                                   GENERATE NACL SUPERCELL                                       
    ##########################################################################################  
    print(f"{'#'*100}")
    print(f"{'Generating NaCl supercell'}")
    print(f"{'#'*100}")
    # GENERATE NACL UNIT CELL
    nacl = bulk('NaCl', 'rocksalt', a=5.64)

    # plot_atoms(nacl)

    # GENERATE NACL SUPER CELL
    P = find_optimal_cell_shape(
        cell = nacl.cell,
        target_shape='sc',
        target_size=50
        )

    supercell_nacl = make_supercell(nacl, P)

    plot_atoms(supercell_nacl)
    os.system(f'mkdir {path}')
    os.system(f'touch {cif_file_nacl}')
    write(filename=cif_file_nacl, images=supercell_nacl, format='cif')
    os.system(f'touch {pdb_file_nacl}')
    write(filename=pdb_file_nacl, images=supercell_nacl)

    ##########################################################################################    
    #                                   FIND QM REGION                                           
    ##########################################################################################  
    print(f"{'#'*100}")
    print(f"{'Finding QM region'}")
    print(f"{'#'*100}")
    candidate_qm_nacl = get_centeroid_region(
        filename=pdb_file_nacl,
        patterns={'Na':['Na'], 'Cl': ['Cl']},
        print_ctrl=False,
        cuboid_threshold=0.2
    )
    print(f'Number of candidates: {len(candidate_qm_nacl['Na'])} {len(candidate_qm_nacl['Cl'])} = {candidate_qm_nacl}')

    # Writes supercell to a file
    write(filename=pdb_file_nacl, images=supercell_nacl)

    ##########################################################################################    
    #                                      PROCESS PDB                                              
    ########################################################################################## 
    print(f"{'#'*100}")
    print(f"{'Processing PDB'}")
    print(f"{'#'*100}")

    process_pdb(
        filename=pdb_file_nacl,
        patterns={'Na':['Na'], 'Cl':['Cl']},
        qm_ids = candidate_qm_nacl,
        qm_resname='LIG'
    )

    plot_atoms(atoms=(read(filename=pdb_file_nacl)))
    # unwrap pdb
    minimum_image_unwrap(filename=pdb_file_nacl)
    plot_atoms(atoms=(read(filename=pdb_file_nacl)))

    ##########################################################################################    
    #                                    FIND DEFECT                                          
    ##########################################################################################  
    print(f"{'#'*100}")
    print(f"{'Finding defect'}")
    print(f"{'#'*100}")

    # Find target pairs to be delted.
    all_qm = [i for v in candidate_qm_nacl.values() for i in v]
    targets = find_schottky_pair_qm(pdb_file_nacl, all_qm)

    print(f'Schottky pairs: {targets}')

    ##########################################################################################    
    #                               GENERATE DEFECT & DEFECT PDB                                          
    ##########################################################################################   
    print(f"{'#'*100}")
    print(f"{'Generating defect'}")
    print(f"{'#'*100}")

    defect, upd_qm = del_atoms_pdb(
        filename=pdb_file_nacl,
        output_filename=defect_pdb_nacl,
        delete_indecies=targets, 
        qm_resname='LIG'
    )


    ##########################################################################################   
    #                               COMPUTE FORMATION ENERGY                                          
    ##########################################################################################   
    print(f"{'#'*100}")
    print(f"{'Computing formation energy'}")
    print(f"{'#'*100}")

    # Calculate qm_charge and multiplicity
    supercell_nacl = read(f'{path}/{general_file_name_nacl}.cif')       # Read the perfect cell from CIF file to 

    #mu_nacl = calc_chemical_potential('NaCl', basis_set='6-31G')          # Chemical potential of NaCl as bonded molecule
    mu_na = calc_chemical_potential('Na', basis_set='6-31G')             # Calculate the chemical potential of Na
    mu_cl = calc_chemical_potential('Cl', basis_set='6-31G')             # Calculate the chemical potential of Cl

    q_nacl, m_nacl = calc_charge_multiplicity(filename=defect_pdb_nacl, qm_resname='LIG', charge_map=charge_map )
    print(f'Defect {defect}')

    E_f = calc_formation_energy(
        filename_perf=pdb_file_nacl,
        filename_defect=defect_pdb_nacl,
        qm_resname='LIG',
        charge_state=q_nacl,        # Charge of the defect
        #chemical_potentials={'NaCl': (1, mu_nacl)},
        chemical_potentials={'Na': (1, mu_na), 'Cl': (1, mu_cl)},
        charge_map=charge_map,
        pe_cutoff=16.0
    )
    # Expected: ~2.2-2.4 eV

if __name__ == '__main__':
    nacl()