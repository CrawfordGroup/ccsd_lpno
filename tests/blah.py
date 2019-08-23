'''
Computing the CCSD-LR dipole polarizability  using an RHF reference
References used:
    - http://github.com/CrawfordGroup/ProgrammingProjects
    - Stanton:1991:4334
    - https://github.com/psi4/psi4numpy
'''

import numpy as np
import psi4
from helper_cc import *
from cc_hbar import *
from cc_lambda import *
from cc_pert import *
from psi4 import constants as pc 

psi4.core.clean()

# Set memory
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
np.set_printoptions(precision=12, threshold=np.inf, linewidth=200, suppress=True)

# Set Psi4 options
mol = psi4.geometry("""                                                 
0 1
O
H 1 1.1
H 1 1.1 2 104 
noreorient
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
omega_nm = 589.0

# Compute RHF energy with psi4
psi4.set_module_options('SCF', {'E_CONVERGENCE': 1e-10})
psi4.set_module_options('SCF', {'D_CONVERGENCE': 1e-10})
e_scf, wfn = psi4.energy('SCF', return_wfn=True)
print('SCF energy: {}\n'.format(e_scf))
print('Nuclear repulsion energy: {}\n'.format(mol.nuclear_repulsion_energy()))

# Create Helper_CCenergy object
hcc = HelperCCEnergy(local, pno_cut, wfn) 

ccsd_e = hcc.do_CC(local=False, e_conv=1e-10, r_conv =1e-10, maxiter=40, start_diis=0)

print('CCSD correlation energy: {}'.format(ccsd_e))
# Create HelperCCHbar object
hbar = HelperHbar(hcc, ccsd_e)

#Hoo = hbar.Hoo
#Hvv = hbar.Hvv
#Hov = hbar.Hov
#
#Hoooo = hbar.Hoooo
#Hvvvv = hbar.Hvvvv
#Hvovv = hbar.Hvovv
#Hooov = hbar.Hooov
#Hovvo = hbar.Hovvo
#Hovov = hbar.Hovov
#Hvvvo = hbar.Hvvvo
#Hovoo = hbar.Hovoo
#
#body = np.load('2body.npz')
#
#print('Hoooo matches:\t\t{}'.format(np.allclose(body['arr_0'], Hoooo, atol=1e-07)))
#print('Hvvvv matches:\t\t{}'.format(np.allclose(body['arr_1'], Hvvvv, atol=1e-07)))
#print('Hvovv matches:\t\t{}'.format(np.allclose(body['arr_2'], Hvovv, atol=1e-07)))
#print('Hooov matches:\t\t{}'.format(np.allclose(body['arr_3'], Hooov, atol=1e-07)))
#print('Hovvo matches:\t\t{}'.format(np.allclose(body['arr_4'], Hovvo, atol=1e-07)))
#print('Hovov matches:\t\t{}'.format(np.allclose(body['arr_5'], Hovov, atol=1e-07)))
#print('Hvvvo matches:\t\t{}'.format(np.allclose(body['arr_6'], Hvvvo, atol=1e-07)))
#print('Hovoo matches:\t\t{}'.format(np.allclose(body['arr_7'], Hovoo, atol=1e-07)))

# Create HelperLamdba object
lda = HelperLambda(hcc, hbar)
pseudo_e = lda.iterate(e_conv=1e-8, r_conv =1e-10, maxiter=30)

#print('l1 matches:\t\t{}'.format(np.allclose(lambda1['arr_0'], lda.l_ia, atol=1e-07)))
#print('l2 matches:\t\t{}'.format(np.allclose(lambda2['arr_0'], lda.l_ijab, atol=1e-07)))

# Set the frequency in hartrees
omega = (pc.c * pc.h * 1e9) / (pc.hartree2J * omega_nm)

# Get the perturbation A for Xs and Ys
dipole_array = hcc.mints.ao_dipole()

# Create HelperPert object and solve for xs and ys
Mu = {}
pert = {}
hresp = {}
polar = {}

i=0
for string in ['X', 'Y', 'Z']:
    Mu[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(dipole_array[i]))
    pert[string] = HelperPert(hcc, hbar, lda, Mu[string], omega)

    #exs = np.load('exs_MU_{}.npz'.format(string))
    #print('Checking if init X1s are same: {}'.format(np.allclose(pert[string].x_ia, exs['arr_0'])))
    #print('Checking if init X2s are same: {}'.format(np.allclose(pert[string].x_ijab, exs['arr_1'])))

    i += 1
    for hand in ['right', 'left']:
        pseudoresponse = pert[string].iterate(hand, r_conv=1e-10)


for string in ['X', 'Y', 'Z']:
    for string2 in ['X', 'Y', 'Z']:
        hresp[string+string2] = HelperResp(lda, pert[string], pert[string2])
        polar[string+string2] = hresp[string+string2].linear_resp()
    
print('Polarizability tensor:')
for string in ['X', 'Y', 'Z']:
    for string2 in ['X', 'Y', 'Z']:
        if string != string2:
            polar[string+string2+'_new'] = 0.5 * (polar[string+string2] + polar[string2+string])
        else:
            polar[string+string2+'_new'] = polar[string+string2]
        print("{} {}: {}\n".format(string, string2, polar[string+string2+'_new']))

trace = polar['XX'] + polar['YY'] + polar['ZZ']
isotropic_polar = trace / 3.0

print("Isotropic CCSD Dipole polarizability at {} nm: {} a.u.".format(omega_nm, isotropic_polar))
    #pn_x1 = np.load('pn_x{}.npy'.format('MU_'+string+'1'))
    #pn_x2 = np.load('pn_x{}.npy'.format('MU_'+string+'2'))
    #pn_y1 = np.load('pn_y{}.npy'.format('MU_'+string+'1'))
    #pn_y2 = np.load('pn_y{}.npy'.format('MU_'+string+'2'))
    #print('x{} matches:\t\t{}'.format(string+'1', np.allclose(pn_x1, pert[string].x_ia, atol=1e-07)))
    #print('x{} matches:\t\t{}'.format(string+'2', np.allclose(pn_x2, pert[string].x_ijab, atol=1e-07)))
    #print('y{} matches:\t\t{}'.format(string+'1', np.allclose(pn_y1, pert[string].y_ia, atol=1e-07)))
    #print('y{} matches:\t\t{}'.format(string+'2', np.allclose(pn_y2, pert[string].y_ijab, atol=1e-07)))

#psi4_ccsd_e = psi4.energy('CCSD', e_convergence=1e-8, r_convergence=1e-7)
#print('Psi4 CCSD energy: {}'.format(psi4_ccsd_e))
