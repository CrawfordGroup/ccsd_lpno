'''
Computing the CCSD correlation energy using an RHF reference
References used:
    - http://github.com/CrawfordGroup/ProgrammingProjects
    - Stanton:1991:4334
    - https://github.com/psi4/psi4numpy
'''

import numpy as np
import psi4
from ndot import ndot
from helper_cc import *
import time

psi4.core.clean()

# Set memory
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
np.set_printoptions(precision=12, linewidth=400, suppress=True)
numpy_memory = 2

# Set Psi4 options
mol = psi4.geometry("""                                                 
H
H 1 R
H 1 D 2 P
H 3 R 1 P 2 T
H 3 D 4 P 1 X
H 5 R 3 P 4 T
H 5 D 6 P 3 X
H 7 R 5 P 6 T    

R = 0.75 
D = 1.5
P = 90.0 
T = 60.0 
X = 180.0

no_reorient
no_com
symmetry c1
""")

psi4.set_options({'basis': 'aug-cc-pVDZ', 'scf_type': 'pk',
                  'freeze_core': 'false', 'e_convergence': 1e-10,
                  'd_convergence': 1e-10, 'save_jk': 'true'})

# Set for CCSD
E_conv = 1e-8
R_conv = 1e-7
maxiter = 40
compare_psi4 = False

# Set for LPNO
local=True
#local=False
e_cut = 1e-4
pno_cut = 1e-7

# Compute RHF energy with psi4
e_scf, wfn = psi4.energy('SCF', return_wfn=True)
print('SCF energy: {}\n'.format(e_scf))

# Create Helper_CCenergy object
hcc = HelperCCEnergy(local, pno_cut, wfn) 

t_ia = hcc.t_ia
t_ijab = hcc.t_ijab

old_e = hcc.old_e
# Iterate until convergence
for i in range(maxiter):
    tau_t = hcc.make_taut(t_ia, t_ijab)
    tau = hcc.make_tau(t_ia, t_ijab)
    new_tia, new_tijab = hcc.update_ts(local, tau, tau_t, t_ia, t_ijab)
    new_e = hcc.corr_energy(new_tia, new_tijab)
    rms = np.linalg.norm(new_tia - t_ia)
    rms += np.linalg.norm(new_tijab - t_ijab)
    print('{}\t\t\t {}\t\t\t{}\t\t\t{}'.format(i, new_e, abs(new_e - old_e), rms))
    if(abs(new_e - old_e) < E_conv and abs(rms) < R_conv):
        print('Convergence reached.\n CCSD Correlation energy: {}\n'.format(new_e))
        break
    t_ia = new_tia
    t_ijab = new_tijab
    old_e = new_e
