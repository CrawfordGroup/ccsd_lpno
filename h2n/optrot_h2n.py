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

cutoffs = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

optrot_lg_list = {}
optrot_mvg_list = {}

for cut in cutoffs:
    psi4.core.clean()

    # Set memory
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('optrot.dat', False)
    np.set_printoptions(precision=12, threshold=np.inf, linewidth=200, suppress=True)

    # Set Psi4 options
    geom = ccsd_lpno.mollib.mollib["h2_7"]
    mol = psi4.geometry(geom)

    psi4.set_options({'basis': 'aug-cc-pvdz', 'scf_type': 'pk',
                      'freeze_core': 'false', 'e_convergence': 1e-10,
                      'd_convergence': 1e-10, 'save_jk': 'true'})

    # Set for CCSD
    E_conv = 1e-8
    R_conv = 1e-7
    maxiter = 40
    compare_psi4 = False

    # Set for linear response calculation
    omega_nm = 589

    # Compute RHF energy with psi4
    psi4.set_module_options('SCF', {'E_CONVERGENCE': 1e-10})
    psi4.set_module_options('SCF', {'D_CONVERGENCE': 1e-10})
    e_scf, wfn = psi4.energy('SCF', return_wfn=True)
    print('SCF energy: {}\n'.format(e_scf))
    print('Nuclear repulsion energy: {}\n'.format(mol.nuclear_repulsion_energy()))

    no_vir = wfn.nmo() - wfn.doccpi()[0] - wfn.frzcpi()[0]
    # Set for LPNO
    localize=True
    #local=False
    pert=True
    pno_cut = cut

    # Do the linear response calculation
    optrot_lg = ccsd_lpno.do_linresp(wfn, omega_nm, mol, method='optrot', gauge='length', localize=localize, pert=pert, pno_cut=pno_cut) 
    optrot_mvg = ccsd_lpno.do_linresp(wfn, omega_nm, mol, method='optrot', gauge='velocity', localize=localize, pert=pert, pno_cut=pno_cut) 
    
    optrot_lg_list['{}'.format(cut)] = optrot_lg
    optrot_mvg_list['{}'.format(cut)] = optrot_mvg

print("List of optical rotations (LG, {} nm): {}".format(omega_nm, optrot_lg_list))
print("List of optical rotations (MVG, {} nm): {}".format(omega_nm, optrot_mvg_list))
