'''
Computing the CCSD-LR dipole polarizability  using an RHF reference
References used:
    - http://github.com/CrawfordGroup/ProgrammingProjects
    - Stanton:1991:4334
    - https://github.com/psi4/psi4numpy
'''

import numpy as np
import psi4
import ccsd_lpno

def test_optrot():
    psi4.core.clean()

    # Set memory
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('test_optrot_out.dat', False)
    np.set_printoptions(precision=12, threshold=np.inf, linewidth=200, suppress=True)

    # Set Psi4 options
    mol = psi4.geometry("""                                                 
     O     -0.028962160801    -0.694396279686    -0.049338350190                                                                  
     O      0.028962160801     0.694396279686    -0.049338350190                                                                  
     H      0.350498145881    -0.910645626300     0.783035421467                                                                  
     H     -0.350498145881     0.910645626300     0.783035421467                                                                  
    symmetry c1        
    """)

    psi4.set_options({'basis': 'sto-3g', 'scf_type': 'pk',
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

    # Set for polarizability calculation
    typ = 'polar'
    omega_nm = 589.0

    # Compute RHF energy with psi4
    psi4.set_module_options('SCF', {'E_CONVERGENCE': 1e-10})
    psi4.set_module_options('SCF', {'D_CONVERGENCE': 1e-10})
    e_scf, wfn = psi4.energy('SCF', return_wfn=True)
    print('SCF energy: {}\n'.format(e_scf))
    print('Nuclear repulsion energy: {}\n'.format(mol.nuclear_repulsion_energy()))

    # Create Helper_CCenergy object
    optrot_lg = ccsd_lpno.do_linresp(wfn, omega_nm, mol, method='optrot', gauge='length', local=local, pno_cut=pno_cut) 
    optrot_mvg = ccsd_lpno.do_linresp(wfn, omega_nm, mol, method='optrot', gauge='velocity', local=local, pno_cut=pno_cut) 

    # Comaprison with Psi4
    psi4.set_options({'d_convergence': 1e-10})
    psi4.set_options({'e_convergence': 1e-10})
    psi4.set_options({'r_convergence': 1e-10})
    psi4.set_options({'omega': [589, 'nm'],
                      'gauge': 'both'})  
    psi4.properties('ccsd', properties=['rotation'])
    psi4.compare_values(optrot_lg, psi4.core.variable("CCSD SPECIFIC ROTATION (LEN) @ 589NM"), \
     5, "CCSD SPECIFIC ROTATION (LENGTH GAUGE) 589 nm") #TEST
    psi4.compare_values(optrot_mvg, psi4.core.variable("CCSD SPECIFIC ROTATION (MVG) @ 589NM"), \
     5, "CCSD SPECIFIC ROTATION (MODIFIED VELOCITY GAUGE) 589 nm") #TEST

