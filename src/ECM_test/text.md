METHODOLOGY

Each test is conducted in two stages, wherein the first stage pertains to model construction and preparation. The second stage pertains to the actual energy calculation, where models constructed and extracted in the first stage are used to calculate the formation energy according to (but with some simplification) the E_f equation described by de Walle and Neugebauer (2004) (equation 1) (Van de Walle and Neugebauer, 2004). For context, the following section will first describe 

STRUCTURE GENERATION USING ASE AND GENICE2

The Atomic Simulation Environemt (ASE) is a general package which provides a broad range of computational tools that allow for the generation and analysis of various molecular structures and properties. In the current investigation ASE was heavily used in the construction of NaCl models and the intermediate modification of both NaCl models and Ice 1h models prior to the analysis using the ensemble parser. Inorder to capture the intriecacies of the proton dissorider observed across ice 1h lattices another package called Genice2 was used. 

ICE CRYSTALS

Generation and modification of the ice crystals for the current study all consisted of five steps: (1) supercell generation using Genice2, stored as .cif and .pdb files; (2) identification of a centroid region to be specified as a the QM region; (3) modification of the corresponding .pdb file for compatiability with the ensamble parser; (4) defect generation and generation of a corresponding .pdb file; (5) compute formation energy. 

NaCl

NaCl rocksalt crystal models were generated using the ASE.build module. Herein ase.build.bulk, ase.build.find_optimal_cell_shape, and ase.build.make_supercell were used to generate an ase.Atoms object for the rocksalt supercell. Steps 2-5 for the ice model as described above was applied for NaCl.  

IDENTIFYING CENTROID REGION

A method for identifying particular atoms in a central cuboid region of the bulk ase.Atoms objects associated with a .pdb file was developed (get_centroid_region) and implemented for both ice and NaCl. The function takes three parameter that include: filename – the string path to the target .pdb file; patterns – a dictionary of the target residue names mapped to their respective atomic symbol sequences (e.g., `{'WAT': ['O', 'H', 'H']}` for water or `{'Na': ['Na'], 'Cl': ['Cl']}` for sodium chloride); and cuboid_threshold – a decimal margin defining the spatial extent of the cuboid QM region relative to the center of the supercell. The function returns a dictionary mapping the residue names to lists of candidate atom indices falling within this central region.

PDB MODIFICATION FOR ENSEMBLE PARSER

To ensure compatibility with the VeloxChem ensemble parser, the generated `.pdb` files require explicit residue assignments and boundary corrections. The `process_pdb` function was developed to parse the PDB file, search for the defined atomic symbol patterns, and assign unique residue numbers. Crucially, the atoms identified as candidates for the QM region in the previous step are assigned a specific residue name (e.g., 'LIG'), which distinguishes them from the surrounding environment during the embedding setup. Following residue assignment, the `minimum_image_unwrap` function applies the minimum image convention to unwrap molecules (such as water) that are split across periodic boundaries, ensuring that all atoms of a single residue remain contiguous within the simulation cell.

DEFECT GENERATION

Defect structures are introduced by selectively deleting atoms from the previously identified QM region. For the ice 1h model, a simple molecular vacancy is created by removing a single water molecule (three contiguous atoms) from the 'LIG' region. For the NaCl rocksalt model, a Schottky defect is simulated by identifying a nearest-neighbor sodium and chloride atom pair within the QM region using the `find_schottky_pair_qm` function, and subsequently removing both to maintain charge neutrality. The `del_atoms_pdb` function handles the removal of these targeted atom indices from the PDB file, sequentially reindexes the remaining atoms, updates the QM candidate list, and generates a new defect `.pdb` file.

ENERGY CALCULATION

The final stage of the methodology involves calculating the defect formation energy utilizing the VeloxChem suite. The `calc_energy_tot` function employs the `EnsembleParser` to load the structural data and define the QM region based on the assigned 'LIG' residue name. The environment is treated using a multiscale embedding approach: a polarizable embedding (PE) layer utilizing the Spherical Expansion of the Potential (SEP) model within a specified cutoff radius (e.g., 16.0 Å), and optionally a non-polarizable embedding (NPE) layer (e.g., TIP3P for water). A restricted Hartree-Fock (HF) SCF calculation with a 6-31G basis set is then performed via the `EnsembleDriver` to obtain the total energy. The formal charge and spin multiplicity of the QM region are dynamically determined by the `calc_charge_multiplicity` function to ensure correct electronic structure treatment for both perfect and defect cells.

To complete the formation energy calculation ($E_f$), the chemical potentials ($\mu_i$) of the removed species must be accounted for. The `calc_chemical_potential` function calculates the isolated reference energies for the required species (e.g., an H2O molecule for ice, or individual open-shell Na and Cl atoms for the Schottky defect) at the same level of theory (HF/6-31G). The final formation energy is then computed by the `calc_formation_energy` function, which subtracts the perfect supercell energy from the defect supercell energy and adds the chemical potential compensation for the removed species, yielding the final defect formation energy.
