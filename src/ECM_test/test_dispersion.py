import sys
import os

# Ensure the correct path
sys.path.append(os.getcwd())
from acetic_acid import acetic_acid

print("=== RUNNING WITHOUT DISPERSION ===")
E_f_no_disp = acetic_acid(basis_set='6-31G**', cuboid_threshold=0.2, target_size=50, target_shape='sc', pe_cutoff=16, pe_model='SEP', npe_model=None, polarizable=False, dispersion=False, xc_functional='B3LYP', debug=False)
print(f"Formation energy WITHOUT dispersion: {E_f_no_disp} Ha")
