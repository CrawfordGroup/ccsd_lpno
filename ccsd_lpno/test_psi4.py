
import numpy as np
import psi4

import time

psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
np.set_printoptions(precision=8, linewidth=400, suppress=True)
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

psi4.set_options({'basis': 'aug-cc-pVDZ', 'scf_type': 'pk', 'mp2_type': 'conv',
                  'freeze_core': 'true', 'e_convergence': 1e-10,
                  'd_convergence': 1e-10, 'save_jk': 'true'})

e_scf = psi4.energy('scf')
e_mp2 = psi4.energy('mp2')
print('MP2 energy: {}\n'.format(e_mp2 - e_scf))
