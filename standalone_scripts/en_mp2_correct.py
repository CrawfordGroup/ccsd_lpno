'''
This script uses the CCSD-LPNO response code
to compute the MP2-level energy correction
to the PNO/PNO++ method by including the
external (truncated) space
'''

import numpy as np
import psi4
import ccsd_lpno
import argparse
import time
import json

parser = argparse.ArgumentParser()
parser.add_argument("--j", default='output.json', type=str, help="Output json filename") 
parser.add_argument("--m", default='h2_2', type=str, help="Molecule from mollib")
args = parser.parse_args()

cutoffs = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 5e-9, 5e-8, 5e-7, 5e-6, 5e-5, 5e-4]
#cutoffs =[0, 1e-6]

mp2_en_list = {}

for cut in cutoffs:
    psi4.core.clean()

    # Set memory
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('optrot.dat', False)
    np.set_printoptions(precision=12, threshold=np.inf, linewidth=200, suppress=True)

    # Set Psi4 options
    geom = ccsd_lpno.mollib.mollib["{}".format(args.m)]
    mol = psi4.geometry(geom)

    psi4.set_options({'basis': 'aug-cc-pvdz', 'scf_type': 'pk',
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
    #localize=True
    local=True
    pert='mu'
    pno_cut = cut

    local = ccsd_lpno.HelperLocal(wfn.doccpi()[0], no_vir)
    hcc = ccsd_lpno.HelperCCEnergy(wfn, local=local, pert=pert, pno_cut=pno_cut)

    mp2_en_list['{}'.format(cut)] = hcc.pno_correct
    #optrot_mvg_list['{}'.format(cut)] = optrot_mvg

with open("{}".format(args.j), "w") as write_file:
    json.dump(mp2_en_list, write_file, indent=4)

print("List of MP2 energy corrections: {}".format(mp2_en_list))
