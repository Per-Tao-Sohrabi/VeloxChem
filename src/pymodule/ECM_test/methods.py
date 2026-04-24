import os
from ase.io import read, write
from ase.build import find_optimal_cell_shape
from ase.build import make_supercell
from ase.visualize.plot import plot_atoms
import numpy as np
import veloxchem.ensembledriver
import veloxchem.ensembleparser

def generate_ice_block(path, cell_dimensions, unit_cell_filename, target_supercell_shape, target_supercell_size, write_to_pdb = True, plot = True):
    super_cell_filename = f'1hx{cell_dimensions[0]}{cell_dimensions[1]}{cell_dimensions[2]}_supercell'
    # Generate unit.
    os.system(f'mkdir {path}')
    # os.system('ls')
    os.system(f'genice2 --rep {cell_dimensions[0]} {cell_dimensions[1]} {cell_dimensions[2]} 1h --format cif > {path}/{unit_cell_filename}.cif')

    # Make Atoms obj. 
    ice = read(
        filename=(f'{path}/{unit_cell_filename}.cif')
        )

    # print(np.asarray(cell))
    P = find_optimal_cell_shape(
        cell=ice.cell, 
        target_size=target_supercell_size,
        target_shape=target_supercell_shape
        )

    ice_block = make_supercell(prim=ice,P=P,)
    
    if write_to_pdb:
        pdb_file = f'{path}/{super_cell_filename}.pdb'
        write(f'{pdb_file}', ice_block)
    
    if plot:
        plot_atoms(ice_block)

def get_space_dimensions(filename = None, ):
    
    ice_block = read(f'{filename}')
    np_ice_block = np.array(ice_block)

    x = 0
    y = 0
    z = 0
    for i in np_ice_block:
        ix, iy, iz = i.position
        if ix > x:
            x = ix
        if iy > y:
            y = iy
        if iz > z:
            z = iz
        print(ix)
    print(x, y, z)
    return [x, y, z]

def get_centeroid_region(filename = None, cuboid_threshold = None, print_status=True):
    
    unit = read(f'{filename}')
    np_supercell = np.array(unit)

    x, y, z = get_space_dimensions(filename)
    midpoint = [x/2, y/2, z/2]
    x_position_threshold = [ (midpoint[0]-x*cuboid_threshold), (midpoint[0]+x*cuboid_threshold) ]
    y_position_threshold = [ (midpoint[1]-y*cuboid_threshold), (midpoint[1]+y*cuboid_threshold) ]
    z_position_threshold = [ (midpoint[2]-z*cuboid_threshold), (midpoint[2]+z*cuboid_threshold) ]

    candidate_qm = [] # list of indecies of atoms in the same moelcule.
    if print_status:
        print(f'1 \t IDENTIFYING CANDIDATE MOLECULES FOR THE QM REGION \n')
        print(f'\n')
        #print(f' ATOMS ARRAY: \n\n {ice_block_ar}')
        print(f'\n')
        print(f'2 \t SEARCH CONFIG: ')
        print(f'\n')
        print(f'\t Lenghts:')
        print(f'\n')
        print(f'\t\t x axis: {x} \n \t\t y axis: {y}, \n \t\t z axis {z}')
        print(f'\n')
        print(f'\t Threshold configuration')
        print(f'\n')
        print(f'\t Cuboid threshold parameters: {100*cuboid_threshold} % (for all axes).')
        print(f'\n')
        print(f'\t Thresholds:')
        print(f'\n')
        print(f'\t\t x axis: {x_position_threshold} \n \t\t y axis: {y_position_threshold}, \n \t\t z axis {z_position_threshold}')

        print(f'\n')
        print(f'3 \t IDENTIFYING CANDIDATE ATOMS')
    c = -1

    for i in np_supercell:
        c =+ 1
        #print(f'checking Atom {i} with coordinates {i.position}...')
        if(i.symbol != 'O'):
            #print(f'REJECTED: Not O type atom.')
            continue
        ix, iy, iz = i.position

        if(ix < x_position_threshold[0] or ix > x_position_threshold[1]):
            #print(f'REJECTED: x coords out of range')
            continue
        else:
            if(iy < y_position_threshold[0] or iy > y_position_threshold[1]):
                continue
                #print(f'REJECTED: y coords out of range')
            else:
                if(iz < z_position_threshold[0] or iz > z_position_threshold[1]):
                    continue
                    #print(f'REJECTED: z coords out of range')
                else:
                    candidate_qm.append(i.index)
                    candidate_qm.append(i.index+1)
                    candidate_qm.append(i.index+2)
                    if print_status:
                        print(f'ACCEPTED: Added candidate molecule with atom indeceies {c}, {c+1}, {c+2}')

    print(f'\n')
    print(f'RESULTS: Candidate atoms list:\n')
    print(f'\tNr. of candidates: {len(candidate_qm)}\n')
    for i in candidate_qm:
        print(f'\t{i} ') #{i}\n')
    return candidate_qm

def process_pdb(filename , atoms_per_mol, target_filename , global_resname , qm_resname, qm_indecies):
    with open(filename) as f:
        lines = f.readlines()

    atom_count = 0
    out = []
    for line in lines:
        if line.startswith('ATOM'):
            curr_resname = global_resname
            mol_idx = atom_count//atoms_per_mol
            res_num = (mol_idx % 9999) + 1 # PDB resid wraps at 9999 
            if atom_count in qm_indecies:
                curr_resname = qm_resname
            line = line[:17] + f'{curr_resname:>3s}' + line[20:22] + f'{res_num:4d}' + line[26:]
            atom_count +=1
            out.append(line)
    with open(target_filename, 'w') as f:
        f.writelines(out)
    
def del_atoms_pdb(filename, output_filename, atom_indecies):
    with open(filename) as f:
        lines = f.readlines()

    atom_count = 0
    out = []
    for line in lines:
        if line.startswith('ATOM'):
            if atom_count not in atom_indecies:
                out.append(line)
            atom_count +=1
    with open(output_filename, 'w') as f:
        f.writelines(out)
    
def calc_energy(filename, qm_resname, pe_cutoff=6.0, npe_cutoff=None):
    ep = veloxchem.ensembleparser.EnsembleParser()   
    ed = veloxchem.ensembledriver.EnsembleDriver()
    ensemble = ep.structures(
        trajectory_file = f"{filename}",
        num_snapshots = None, 
        qm_region = f"resname {qm_resname}", 
        pe_cutoff = pe_cutoff,
        npe_cutoff=npe_cutoff
    )

    ed.set_env_models(pe_model = 'SEP', npe_model='tip3p')

    scf_results = ed.compute(ensemble, basis_set = '6-31G')
    return scf_results