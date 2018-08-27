'''
Computing the CCSD correlation energy using an RHF reference
References used:
    - http://github.com/CrawfordGroup/ProgrammingProjects
    - Stanton:1991:4334
    - https://github.com/psi4/psi4numpy
'''

import numpy as np
import psi4
from helper_cc import *
import time

psi4.core.clean()

# Set memory
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
np.set_printoptions(precision=12, threshold=np.inf, linewidth=200, suppress=True)
numpy_memory = 2

# Set Psi4 options
mol = psi4.geometry("""                                                 
0 1
O
H 1 1.1
H 1 1.1 2 104 
noreorient
symmetry c1
""")

psi4.set_options({'basis': '6-31g', 'scf_type': 'pk',
                  'freeze_core': 'false', 'e_convergence': 1e-10,
                  'd_convergence': 1e-10, 'save_jk': 'true'})

# Set for CCSD
E_conv = 1e-8
R_conv = 1e-7
maxiter = 40
compare_psi4 = False

# Set for LPNO
#local=True
local=False
pno_cut = 0.0

# Compute RHF energy with psi4
psi4.set_module_options('SCF', {'E_CONVERGENCE': 1e-8})
psi4.set_module_options('SCF', {'D_CONVERGENCE': 1e-8})
e_scf, wfn = psi4.energy('SCF', return_wfn=True)
print('SCF energy: {}\n'.format(e_scf))
print('Nuclear repulsion energy: {}\n'.format(mol.nuclear_repulsion_energy()))

# Create Helper_CCenergy object
hcc = HelperCCEnergy(local, pno_cut, wfn) 

ccsd_e = hcc.do_CC(local=False, e_conv=1e-8, r_conv =1e-7, maxiter=20, start_diis=0)

#psi4_ccsd_e = psi4.energy('CCSD', e_convergence=1e-8, r_convergence=1e-7)
#print('Psi4 CCSD energy: {}'.format(psi4_ccsd_e))
