'''
Computing the CCSD correlation energy using an RHF reference
References used:
    - http://github.com/CrawfordGroup/ProgrammingProjects
    - Stanton:1991:4334
    - https://github.com/psi4/psi4numpy
'''

import numpy as np
import psi4
import ccsd_lpno
from psi4 import constants as pc 

psi4.core.clean()

# Set memory
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
np.set_printoptions(precision=12, threshold=np.inf, linewidth=200, suppress=True)

def test_ccsd():
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
    symmetry c1
    """)

    ''' 0 1
    O
    H 1 1.1
    H 1 1.1 2 104 
    noreorient
    symmetry c1'''
    
    psi4.set_options({'basis': 'cc-pVDZ', 'scf_type': 'pk',
                      'e_convergence': 1e-10,
                      #'freeze_core': 'false', 'e_convergence': 1e-10,
                      'd_convergence': 1e-10, 'save_jk': 'true'})

    # Set for CCSD
    E_conv = 1e-8
    R_conv = 1e-8
    maxiter = 40
    compare_psi4 = False

    # Set for LPNO
    #local=True
    local=None
    pno_cut = 0.0

    # Compute RHF energy with psi4
    psi4.set_module_options('SCF', {'E_CONVERGENCE': 1e-10})
    psi4.set_module_options('SCF', {'D_CONVERGENCE': 1e-10})
    e_scf, wfn = psi4.energy('SCF', return_wfn=True)
    print('SCF energy: {}\n'.format(e_scf))
    print('Nuclear repulsion energy: {}\n'.format(mol.nuclear_repulsion_energy()))

    # Create Helper_CCenergy object
    hcc = ccsd_lpno.HelperCCEnergy(wfn, local=local, pno_cut=pno_cut) 

    ccsd_e = hcc.do_CC(local=local, e_conv=E_conv, r_conv=R_conv, maxiter=40, start_diis=0)

    print('CCSD correlation energy: {}'.format(ccsd_e))

    psi4_ccsd_e = psi4.energy('CCSD', e_convergence=E_conv, r_convergence=R_conv)
    print('Psi4 CCSD energy: {}'.format(psi4_ccsd_e))
    psi4.compare_values(e_scf+ccsd_e, psi4_ccsd_e, 8, "CCSD Energy")

if __name__=="__main__":
    test_ccsd()
