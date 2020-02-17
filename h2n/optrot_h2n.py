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
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--j", default='output.json', type=str, help="Output json filename") 
parser.add_argument("--m", default='h2o2', type=str, help="Mollib molecule name") 
args = parser.parse_args()

#cutoffs = [1e-10, 5e-9, 5e-8, 5e-7, 5e-6, 5e-5]
cutoffs =[0]

optrot_lg_list = {}
optrot_mvg_list = {}

for cut in cutoffs:
    psi4.core.clean()

    # Set memory
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('optrot.dat', False)
    np.set_printoptions(precision=12, threshold=np.inf, linewidth=200, suppress=True)

    # Set Psi4 options
    geom = ccsd_lpno.mollib.mollib["{}".format(args.m)]
    mol = psi4.geometry(geom)

    psi4.set_options({'basis': '6-31g', 'scf_type': 'pk',
                      'freeze_core': 'false', 'e_convergence': 1e-12,
                      'd_convergence': 1e-12, 'save_jk': 'true'})

    # Set for CCSD
    E_conv = 1e-10
    R_conv = 1e-8
    maxiter = 40
    compare_psi4 = False

    # Set for linear response calculation
    omega_nm = 589

    # Compute RHF energy with psi4
    psi4.set_module_options('SCF', {'E_CONVERGENCE': 1e-12})
    psi4.set_module_options('SCF', {'D_CONVERGENCE': 1e-12})
    e_scf, wfn = psi4.energy('SCF', return_wfn=True)
    print('SCF energy: {}\n'.format(e_scf))
    print('Nuclear repulsion energy: {}\n'.format(mol.nuclear_repulsion_energy()))

    no_vir = wfn.nmo() - wfn.doccpi()[0] - wfn.frzcpi()[0]
    # Set for LPNO
    localize=True
    #local=False
    pert='mu'
    pno_cut = cut

    # Do the linear response calculation
    optrot_lg = ccsd_lpno.do_linresp(wfn, omega_nm, mol, method='optrot', gauge='length', localize=localize, pert=pert, pno_cut=pno_cut) 
    t0 = time.time()
    #optrot_mvg = ccsd_lpno.do_linresp(wfn, omega_nm, mol, method='optrot', gauge='velocity', localize=localize, pert=pert, pno_cut=pno_cut, e_cut=1e-4) 
    t1 = time.time()
    print("Total time: {}".format(t1 - t0))
    optrot_lg_list['{}'.format(cut)] = optrot_lg
    #optrot_mvg_list['{}'.format(cut)] = optrot_mvg

#optrot_data = {}
#optrot_data['LG'] = optrot_lg_list
#optrot_data['MVG'] = optrot_mvg_list
#with open("{}".format(args.j), "w") as write_file:
#    json.dump(optrot_data, write_file, indent=4)

print("List of optical rotations (LG, {} nm): {}".format(omega_nm, optrot_lg_list))
#print("List of optical rotations (MVG, {} nm): {}".format(omega_nm, optrot_mvg_list))
