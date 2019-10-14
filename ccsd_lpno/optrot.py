'''
Computing the optical rotation using CCLR theory
and and RHF reference
References:
    - http://github.com/CrawfordGroup/ProgrammingProjects
    - Stanton:1991:4334
    - https://github.com/psi4/psi4numpy
    - Crawford:2006:227
'''

import numpy as np
import psi4
import ccsd_lpno
from local import *
from psi4 import constants as pc 

psi4.core.clean()

# Set memory
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
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

# Compute RHF energy with psi4
psi4.set_module_options('SCF', {'E_CONVERGENCE': 1e-10})
psi4.set_module_options('SCF', {'D_CONVERGENCE': 1e-10})
e_scf, wfn = psi4.energy('SCF', return_wfn=True)
print('SCF energy: {}\n'.format(e_scf))
print('Nuclear repulsion energy: {}\n'.format(mol.nuclear_repulsion_energy()))

no_vir = wfn.nmo() - wfn.doccpi()[0] - wfn.frzcpi()[0]

local = HelperLocal(wfn.doccpi()[0], no_vir)
pert=True
pno_cut = 0.0

# Create Helper_CCenergy object
hcc = ccsd_lpno.HelperCCEnergy(wfn, local=local, pert=pert, pno_cut=pno_cut) 
ccsd_e = hcc.do_CC(local=local, e_conv=1e-10, r_conv =1e-10, maxiter=40, start_diis=0)

print('CCSD correlation energy: {}'.format(ccsd_e))

# Create HelperCCHbar object
hbar = ccsd_lpno.HelperHbar(hcc, ccsd_e)
print('Finished building Hbar matrix elements.')

# Create HelperLamdba object
lda = ccsd_lpno.HelperLambda(hcc, hbar)
pseudo_e = lda.iterate(local=local, e_conv=1e-8, r_conv =1e-10, maxiter=30)

# Set the frequency in hartrees
omega_nm = 589.0
omega = (pc.c * pc.h * 1e9) / (pc.hartree2J * omega_nm)

### Length gauge OR calculation
### Form of linear response function: <<mu;L>>

# Get the perturbation mu
dipole_array = hcc.mints.ao_dipole()

# Get the angular momentum L
angular_momentum = hcc.mints.ao_angular_momentum()

# Create HelperPert objects for both
Mu = {}
pert1 = {}
L = {}
pert2 = {}

# Rosenfeld tensor
beta = {}
betap = {}
beta_new = {}

i=0
for string in ['X', 'Y', 'Z']:
    Mu[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(dipole_array[i]))
    pert1[string] = ccsd_lpno.HelperPert(hcc, hbar, lda, Mu[string], omega)
    L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
    pert2[string] = ccsd_lpno.HelperPert(hcc, hbar, lda, L[string], omega)

    i+=1
    for hand in ['right', 'left']:
        pseudoresponse1 = pert1[string].iterate(hand, r_conv=1e-10, local=local)
    for hand in ['right', 'left']:
        pseudoresponse2 = pert2[string].iterate(hand, r_conv=1e-10, local=local)

print('Rosenfeld tensor:')
for string1 in ['X', 'Y', 'Z']:
    for string2 in ['X', 'Y', 'Z']:
        beta[string1+string2] = ccsd_lpno.HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
        betap[string1+string2] = ccsd_lpno.HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

        beta_new[string1+string2] = 0.5 * (beta[string1+string2] - betap[string1+string2])
        print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

trace = 0.0
for string in ['XX','YY','ZZ']:
    trace += beta_new[string]
trace /= 3.0

# Calculation of the specific rotation
Mass = 0
for atom in range(mol.natom()):
    Mass += mol.mass(atom)
h_bar = pc.h / (2.0 * np.pi)
prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
# Have to multiply with omega for length gauge
optrot_lg = prefactor * trace * omega



### Velocity gauge OR calculation
### Form of linear response function: <<p;L>>

# Get the perturbation P
p_array = hcc.mints.ao_nabla()

# Get the angular momentum L
angular_momentum = hcc.mints.ao_angular_momentum()

# Create HelperPert objects for both
P = {}
pert1 = {}
L = {}
pert2 = {}

# Rosenfeld tensor
beta = {}
betap = {}
beta_new = {}

i=0
for string in ['X', 'Y', 'Z']:
    P[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(p_array[i]))
    pert1[string] = ccsd_lpno.HelperPert(hcc, hbar, lda, P[string], omega)
    L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
    pert2[string] = ccsd_lpno.HelperPert(hcc, hbar, lda, L[string], omega)

    i+=1
    for hand in ['right', 'left']:
        pseudoresponse1 = pert1[string].iterate(hand, r_conv=1e-10, local=local)
    for hand in ['right', 'left']:
        pseudoresponse2 = pert2[string].iterate(hand, r_conv=1e-10, local=local)

print('Rosenfeld tensor:')
for string1 in ['X', 'Y', 'Z']:
    for string2 in ['X', 'Y', 'Z']:
        beta[string1+string2] = ccsd_lpno.HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
        betap[string1+string2] = ccsd_lpno.HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

        beta_new[string1+string2] = 0.5 * (beta[string1+string2] + betap[string1+string2])
        print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

trace = 0.0
for string in ['XX','YY','ZZ']:
    trace += beta_new[string]
trace /= 3.0

# Calculation of the specific rotation
Mass = 0
for atom in range(mol.natom()):
    Mass += mol.mass(atom)
h_bar = pc.h / (2.0 * np.pi)
prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
optrot_vg = prefactor * trace
# So velocity gauge is / omega

print("Specific rotation value (velocity gauge): {} deg/ [dm (gm/ml)]".format(optrot_vg))

### Modified velocity gauge OR calculation
### Form of linear response function: <<p;L>> - <<p;L>>_0
### Using the velocity gauge OR value and subtracting the static value
omega = 0.0

# Get the perturbation P
p_array = hcc.mints.ao_nabla()

# Get the angular momentum L
angular_momentum = hcc.mints.ao_angular_momentum()

# Create HelperPert objects for both
P = {}
pert1 = {}
L = {}
pert2 = {}

# Rosenfeld tensor
beta = {}
betap = {}
beta_new = {}

i=0
for string in ['X', 'Y', 'Z']:
    P[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(p_array[i]))
    pert1[string] = ccsd_lpno.HelperPert(hcc, hbar, lda, P[string], omega)
    L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
    pert2[string] = ccsd_lpno.HelperPert(hcc, hbar, lda, L[string], omega)

    i+=1
    for hand in ['right', 'left']:
        pseudoresponse1 = pert1[string].iterate(hand, r_conv=1e-10, local=local)
    for hand in ['right', 'left']:
        pseudoresponse2 = pert2[string].iterate(hand, r_conv=1e-10, local=local)

print('Rosenfeld tensor:')
for string1 in ['X', 'Y', 'Z']:
    for string2 in ['X', 'Y', 'Z']:
        beta[string1+string2] = ccsd_lpno.HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
        betap[string1+string2] = ccsd_lpno.HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

        beta_new[string1+string2] = 0.5 * (beta[string1+string2] + betap[string1+string2])
        print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

trace = 0.0
for string in ['XX','YY','ZZ']:
    trace += beta_new[string]
trace /= 3.0

# Calculation of the specific rotation
Mass = 0
for atom in range(mol.natom()):
    Mass += mol.mass(atom)
h_bar = pc.h / (2.0 * np.pi)
prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
optrot_diff = prefactor * trace

optrot_mvg = optrot_vg - optrot_diff
print("Specific rotation value (length gauge): {} deg/ [dm (gm/ml)]".format(optrot_lg))

print("Specific rotation value (modified velocity gauge): {} deg/ [dm (gm/ml)]".format(optrot_mvg))
