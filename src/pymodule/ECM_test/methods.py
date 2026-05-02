from jax.lax import switch
from tracemalloc import stop
import os
from typing import Literal
from ase.io import read, write
from ase.build import find_optimal_cell_shape
from ase.build import make_supercell
from ase.visualize.plot import plot_atoms
import numpy as np
import veloxchem.ensembledriver
import veloxchem.ensembleparser
import pandas as pd

#from pymodule.ECM_test.tempfiles.main import PDB_FILE, QM_RESNAME
    
PossiblePhases = Literal['1h', '1c']    
def generate_ice_block(path, phase: PossiblePhases, cell_dimensions, unit_cell_filename, target_supercell_shape, target_supercell_size, write_to_pdb = True, plot = True):
    super_cell_filename = f'{phase}x{cell_dimensions[0]}{cell_dimensions[1]}{cell_dimensions[2]}_supercell'
    # Generate unit.
    os.system(f'mkdir {path}')
    # os.system('ls')
    os.system(f'genice2 --rep {cell_dimensions[0]} {cell_dimensions[1]} {cell_dimensions[2]} {phase} --format cif > {path}/{unit_cell_filename}.cif')

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

# TODO: Generalize this function to output more geometry parameters if necessary
def get_space_dimensions(filename = None, ):
    
    supercell = read(f'{filename}')
    np_supercell = np.array(supercell)

    x = 0
    y = 0
    z = 0
    for i in np_supercell:
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

# TODO: Simplify to output all indecies in QM region. 
def get_centeroid_region(filename = None, cuboid_threshold = None, print_ctrl=True, target_atom_symbol='O', residue_len=1):
    '''
    Param:
        filename: Takes path string of pdb or cif file. 
        cuboid_threshold: Decimal margin from center to the edge of the cuboid QM region.
        print_ctrl: Set true to print real-time deduction status.
        target_atom_symbol=List of strings or just string of a single atom symbol that 
    '''
    supercell = read(f'{filename}')
    np_supercell = np.array(supercell)

    x, y, z = get_space_dimensions(filename)
    
    midpoint = [x/2, y/2, z/2]
    
    x_position_threshold = [ (midpoint[0]-x*cuboid_threshold), (midpoint[0]+x*cuboid_threshold) ]
    y_position_threshold = [ (midpoint[1]-y*cuboid_threshold), (midpoint[1]+y*cuboid_threshold) ]
    z_position_threshold = [ (midpoint[2]-z*cuboid_threshold), (midpoint[2]+z*cuboid_threshold) ]

    candidate_qm = [] # list of indecies of atoms in the same moelcule.
    
    if print_ctrl:
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
    skip_count = 0
    for i in np_supercell:              # Access each individual atom. 
        c += 1                          # Count up
        if skip_count > 0: 
            skip_count -= 1
            continue
        if(i.symbol not in target_atom_symbol):
            continue
        ix, iy, iz = i.position         # Isolate atom coordinates
        if(ix < x_position_threshold[0] or ix > x_position_threshold[1]):           # Validate position
            continue
        else:
            if(iy < y_position_threshold[0] or iy > y_position_threshold[1]):       # Validate position
                continue
            else:
                if(iz < z_position_threshold[0] or iz > z_position_threshold[1]):   # Validate position
                    continue
                else:

                    # 1. Validate                                                
                    for r in range(residue_len):                # Ensure that suffiecnt atoms are marked.
                        candidate_qm.append(i.index + r + 1)    # Append main indecies and indecies chained through the residue in one go.
                        skip_count = residue_len - 1            # Do not append the indecies in the residue chain.
                    if print_ctrl:
                        print(f'ACCEPTED: Added candidate molecule with atom indeceies {c}, {c+1}, {c+2}')
    print(f'\n')
    print(f'RESULTS: Candidate atoms list:\n')
    print(f'\tNr. of candidates: {len(candidate_qm)}\n')
    for i in candidate_qm:
        print(f'\t{i} ') #{i}\n')
    return candidate_qm

def process_pdb(filename, patterns, qm_ids=None, qm_resname=None):
    """
    Process a PDB file, assigning residue names and unique residue numbers
    to atoms matching specified patterns.

    Args:
        filename:   Path to the PDB file.
        patterns:   Dict mapping residue names to atom symbol patterns.
                    e.g. {'WAT': [' O', ' H', ' H'], 'ION': [' Na']}
        qm_ids:     Optional list of line indices whose residue name
                    should be overridden with qm_resname.
        qm_resname: Residue name for QM-region molecules (e.g. 'LIG').
    """
    if qm_ids is None:
        qm_ids = []

    print(f'Processing {filename}')
    with open(filename) as f:
        lines = f.readlines()

    # Identify ATOM data region
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.startswith('ATOM'):
            if start_idx is None:
                start_idx = i
            end_idx = i

    if start_idx is None:
        print('No ATOM lines found.')
        return

    print(f'Data row index limits: Start, {start_idx}; End, {end_idx}')

    # Extract atom symbols from the ATOM region
    symbols = [line[14:16] for line in lines[start_idx:end_idx + 1]]

    # Find all matching segments for every pattern
    matches = []  # (line_index, pattern_length, resname)
    for resname, pattern in patterns.items():
        for idx in _find_pattern(symbols, pattern, offset=start_idx):
            matches.append((idx, len(pattern), resname))

    # Sort by line index so residue numbers are assigned in file order
    matches.sort(key=lambda x: x[0])
    print(f'Found {len(matches)} matching segments.')

    # Assign residue names and unique residue numbers
    res_num = 2  # start at 2 so unmatched atoms stay at 1
    for seg_idx, pat_len, resname in matches:
        res_num_str = f'{res_num:>4}'  # right-justified, 4 chars (PDB cols 23-26)

        residue_name = resname
        if seg_idx in qm_ids and qm_resname is not None:
            residue_name = qm_resname

        for i in range(pat_len):
            line = lines[seg_idx + i]
            # cols: [:17] record+serial+name | [17:20] resname | [20:22] blank+chain | [22:26] resseq | [26:] rest
            lines[seg_idx + i] = line[:17] + residue_name + line[20:22] + res_num_str + line[26:]

        res_num += 1

    with open(filename, 'w') as f:
        f.writelines(lines)
    print(f'Done. Assigned {res_num - 2} residues.')


def _find_pattern(symbols, pattern, offset=0):
    """
    Find all non-overlapping occurrences of pattern in symbols.
    Returns a list of absolute line indices (adjusted by offset).
    """
    matches = []
    i = 0
    while i <= len(symbols) - len(pattern):
        if symbols[i:i + len(pattern)] == pattern:
            matches.append(i + offset)
            i += len(pattern)  # skip past this match to avoid overlap
        else:
            i += 1
    return matches


    # atom_count = 0
    # out = []
    # for line in lines:
    #     if line.startswith('ATOM') or line.startswith('HETATM'):
    #         # 1) Determine residue name
    #         curr_resname = global_resname  # may be None

    #         if atom_type_resnames and atom_count not in qm_indecies:
    #             # Extract atom name from cols 12-16 (reliable in ASE PDBs)
    #             atom_name = line[12:16].strip()
    #             # Also try element from cols 76-78 as fallback
    #             atom_symbol = line[76:78].strip() if len(line) >= 78 else ''

    #             # Try matching by symbol first, then by atom name
    #             if atom_symbol in atom_type_resnames:
    #                 curr_resname = atom_type_resnames[atom_symbol]
    #             elif atom_name in atom_type_resnames:
    #                 curr_resname = atom_type_resnames[atom_name]

    #         # 2) QM atoms always override
    #         if atom_count in qm_indecies:
    #             curr_resname = qm_resname

    #         # 3) Safety check
    #         if curr_resname is None:
    #             raise ValueError(
    #                 f"Could not determine residue name for atom {atom_count}: {line.strip()}\n"
    #                 f"Set global_resname or provide atom_type_resnames mapping."
    #             )

    #         mol_idx = atom_count // atoms_per_mol
    #         res_num = (mol_idx % 9999) + 1

    #         # Fix atom name to uppercase to match PE database expectations
    #         atom_name = line[12:16].strip().upper()
    #         line = line[:12] + f'{atom_name:>4s}' + line[16:]
    #         line = line[:17] + f'{curr_resname:>3s}' + line[20:22] + f'{res_num:4d}' + line[26:]
    #         atom_count += 1
    #         out.append(line)
    # with open(target_filename, 'w') as f:
    #     f.writelines(out)

def minimum_image_unwrap(filename):
    """
    Unwrap molecules split across periodic boundaries in a PDB file.
    For each residue, shifts all atoms to be within half a cell length
    of the first atom in that residue (the minimum image convention).
    
    Must be called AFTER process_pdb has assigned correct residue numbers.
    Modifies the file in place.
    """
    with open(filename) as f:
        lines = f.readlines()

    # Extract cell dimensions from the CRYST1 record
    cell = None
    for line in lines:
        if line.startswith('CRYST1'):
            cell = [float(line[6:15]), float(line[15:24]), float(line[24:33])]
            break

    if cell is None:
        print('No CRYST1 record found. Cannot unwrap.')
        return

    # Group ATOM lines by (chain ID, residue number)
    residues = {}
    for i, line in enumerate(lines):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            key = line[21:26]  # chain ID + residue seq number
            if key not in residues:
                residues[key] = []
            residues[key].append(i)

    # For each residue, unwrap atoms relative to the first atom
    n_fixed = 0
    for key, atom_indices in residues.items():
        if len(atom_indices) < 2:
            continue

        # Reference position = first atom in the residue (e.g. the Oxygen)
        ref_line = lines[atom_indices[0]]
        ref_pos = [float(ref_line[30:38]), float(ref_line[38:46]), float(ref_line[46:54])]

        for idx in atom_indices[1:]:
            line = lines[idx]
            pos = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            fixed = False

            for ax in range(3):
                diff = pos[ax] - ref_pos[ax]
                if diff > cell[ax] / 2:
                    pos[ax] -= cell[ax]
                    fixed = True
                elif diff < -cell[ax] / 2:
                    pos[ax] += cell[ax]
                    fixed = True

            if fixed:
                lines[idx] = line[:30] + f'{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}' + line[54:]
                n_fixed += 1

    with open(filename, 'w') as f:
        f.writelines(lines)

    print(f'Unwrapped {n_fixed} atoms across {len(residues)} residues in {filename}.')
    
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
    
# Atomic numbers for common elements, used to count electrons in QM region
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
}

def calc_charge_and_multiplicity(filename, qm_resname, charge_map):
    """
    Calculate the net formal charge and spin multiplicity of the QM region
    by reading the PDB and identifying QM atoms by their residue name.

    This approach is robust to atom deletions (defect cells) because it
    relies on residue labels baked into the PDB, not on atom indices.

    Args:
        filename:     Path to the PDB file.
        qm_resname:   Residue name used for QM region (e.g. 'LIG').
        charge_map:   Dict mapping atom symbols to formal charges,
                      e.g. {'Na': 1, 'Cl': -1, 'O': -2, 'H': 1}

    Returns:
        tuple: (qm_charge, qm_multiplicity)
            - qm_charge (int): Net formal charge of the QM region.
            - qm_multiplicity (int): Spin multiplicity (1 = singlet, 2 = doublet, ...).
              Assumes lowest-spin ground state: even electrons → 1, odd → 2.
    """
    qm_charge = 0
    total_electrons = 0

    with open(filename) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                resname = line[17:20].strip()
                if resname == qm_resname:
                    # Extract element symbol from cols 76-78 (standard PDB)
                    symbol = line[76:78].strip()
                    qm_charge += charge_map.get(symbol, 0)
                    total_electrons += ATOMIC_NUMBERS.get(symbol, 0)

    # Actual electron count = nuclear electrons minus net charge
    # (positive charge means fewer electrons, negative means more)
    actual_electrons = total_electrons - qm_charge

    # Lowest-spin assumption: even e⁻ → singlet (1), odd e⁻ → doublet (2)
    qm_multiplicity = 1 if actual_electrons % 2 == 0 else 2

    return qm_charge, qm_multiplicity

def calc_energy(filename, qm_resname, pe_cutoff=6.0, npe_cutoff=None, qm_charge=0, qm_multiplicity=1):
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

    # TODO: Testa att sätta ed.xcfun = 'B3LYP'

    scf_results = ed.compute(ensemble, basis_set = '6-31G', qm_charge=qm_charge, qm_multiplicity=qm_multiplicity)
    return scf_results