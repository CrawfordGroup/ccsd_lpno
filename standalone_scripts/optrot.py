'''
This script uses psi4 to compute reference
values for CCSD Correlation energy,
dipole polarizability and optical rotation
'''

import numpy as np
import psi4
import argparse
import time
import ccsd_lpno

parser = argparse.ArgumentParser()
parser.add_argument("--m", default='h2_2', type=str, help="Molecule from mollib")
args = parser.parse_args()

mp2_en_list = {}

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
psi4.set_options({'omega':[589, 'nm'], 'gauge': 'both'})
psi4.properties('ccsd', properties=['rotation', 'polarizability'])

print("Correlation energy: {}".format(psi4.core.variable("CCSD CORRELATION ENERGY")))
print("Optical rotation (LG): {}".format(psi4.core.variable("CCSD SPECIFIC ROTATION (LEN) @ 589NM")))
print("Optical rotation (MVG): {}".format(psi4.core.variable("CCSD SPECIFIC ROTATION (MVG) @ 589NM")))
print("Polarizability: {}".format(psi4.core.variable("CCSD DIPOLE POLARIZABILITY @ 589NM")))
