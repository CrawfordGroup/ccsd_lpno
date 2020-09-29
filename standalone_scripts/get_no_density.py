'''
This script computes the FVNO and FVNO++ densities,
diagonalizes them to obtain NOs, and then plots the
orbital spatial extents or quadrupole moment integrals
'''

import psi4
import numpy as np
import argparse
import ccsd_lpno
from opt_einsum import contract
import matplotlib.pyplot as plt

psi4.core.clean()

# get mol as argument
parser = argparse.ArgumentParser()
parser.add_argument("--m", default='h2_2', type=str, help="Molecule from mollib")
args = parser.parse_args()

geom = ccsd_lpno.mollib.mollib["{}".format(args.m)]
mol = psi4.geometry(geom)

psi4.set_options({'basis': 'aug-cc-pVDZ', 'scf_type': 'pk',
                  'freeze_core': 'false', 'e_convergence': 1e-12,
                  'd_convergence': 1e-12, 'save_jk': 'true'})
psi4.set_options({'omega':[589, 'nm']})

E, wf = psi4.energy('scf', return_wfn=True)
prop, ccwf = psi4.properties('ccsd', properties=['dipole'], return_wfn=True, ref_wfn=wf)
mints = psi4.core.MintsHelper(wf.basisset())

no_occ = wf.doccpi()[0]
no_vir = wf.nmo() - wf.doccpi()[0] - wf.frzcpi()[0]
hcc_pno = ccsd_lpno.HelperCCEnergy(wf, local=None, pert=None) 

scf_density = wf.Da_subset("MO").to_array()
print("SCF density: {}".format(scf_density))
#np.save('scf_density.npy', scf_density)

# create FVNO density
t_ijab = 2.0 * hcc_pno.t_ijab - hcc_pno.t_ijab.swapaxes(2,3)
mp2_density = 2.0 * contract('ijac,ijbc->ab', t_ijab, hcc_pno.t_ijab)
print("MP2 density: {}".format(mp2_density))
np.save('h2o2_mp2_density.npy', mp2_density)

# diagonalize MP2 density to get NOs
eps, Q = np.linalg.eigh(mp2_density)

print("FVNO Occupations",eps)
np.save('h2o2_fvno_occ_nonlocal.npy', eps)
# get AO quadrupole, XX+YY+ZZ
omega = np.asarray(mints.ao_quadrupole()[0])
for i in [3,5]:
    omega += np.asarray(mints.ao_quadrupole()[i])
print("Shape of omega: {}".format(omega.shape))

C = hcc_pno.C_arr
omega_canon = contract('uj,vi,uv', C, C, omega)
print("Shape of omega_canon: {}".format(omega_canon[no_occ:, no_occ:].shape))
print("Orbital energies: {}".format(hcc_pno.eps_vir))
eps_vir = hcc_pno.eps_vir
np.save('omega_canon_h2o2.npy', omega_canon)
np.save('eps_canon_h2o2.npy', eps_vir)

CQ = C.copy()
CQ[:, no_occ:] = np.dot(C[:,no_occ:], Q)

omega_fvno = contract('Aa,ab,bB->AB', CQ.T, omega, CQ)
np.save('omega_fvno_h2o2.npy', omega_fvno)
np.save('eps_fvno_h2o2.npy', eps)

# get pnopp helperCC object
local_pnopp = ccsd_lpno.HelperLocal(no_occ, no_vir)
hcc_pnopp = ccsd_lpno.HelperCCEnergy(wf, local=local_pnopp, local_occ=False, pert='mu')
(denom_ia, denom_ijab) = hcc_pnopp.denom_tuple

# check that the occupied orbitals aren't localized
#if np.allclose(hcc_pno.C_arr[:, :no_occ], hcc_pnopp.C_arr[:, :no_occ], atol=1e-6):
#    print("Localizing occupieds!")
#    exit()

# prepare perturbation
dipole_array = hcc_pnopp.mints.ao_dipole()
dirn = ['X','Y','Z']
A_list = {}
for i in range(3):
    A_list[dirn[i]] = np.einsum('uj,vi,uv', hcc_pnopp.C_arr, hcc_pnopp.C_arr, np.asarray(dipole_array[i]))

# prepare FVNO++ density
fvnopp_density = np.zeros((no_vir, no_vir))
X_guess_ia = {}
X_guess_ijab = {}
for A in A_list.values():
    Avvoo = contract('ijeb,ae->abij', hcc_pnopp.t_ijab, A[no_occ:, no_occ:])
    Avvoo -= contract('mjab,mi->abij', hcc_pnopp.t_ijab, A[:no_occ, :no_occ])
    Abar = Avvoo.swapaxes(0,2).swapaxes(1,3)
    Abar += Avvoo.swapaxes(0,2).swapaxes(1,3)
    Abar_ia = A[:no_occ, no_occ:]

    # A1_bar?
    X_guess_ia[i] = Abar_ia.copy() 
    X_guess_ia[i] /= denom_ia 
    X_guess_ijab[i] = Abar.copy()
    X_guess_ijab[i] /= denom_ijab

    temp = 2.0 * X_guess_ijab[i] - X_guess_ijab[i].swapaxes(2,3)
    fvnopp_density += 2.0 * contract('ijac,ijbc->ab', temp, X_guess_ijab[i])
    fvnopp_density += contract('ia,ib->ab', X_guess_ia[i], X_guess_ia[i])

eps_pnopp, vnopp = np.linalg.eigh(fvnopp_density)

CQpp = C.copy()
CQpp[:, no_occ:] = np.dot(C[:,no_occ:], vnopp)

omega_fvnopp = contract('Aa,ab,bB->AB', CQpp.T, omega, CQpp)
np.save('omega_fvnopp_h2o2.npy', omega_fvnopp)
np.save('eps_fvnopp_h2o2.npy', eps_pnopp)

'''
no_occ = 9
omega_canon = np.load('omega_canon_h2o2.npy')
omega_fvno = np.load('omega_fvno_h2o2.npy')
omega_fvnopp = np.load('omega_fvnopp_h2o2.npy')
eps_vir = np.load('eps_canon_h2o2.npy')
eps = np.load('eps_fvno_h2o2.npy')
eps_pnopp = np.load('eps_fvnopp_h2o2.npy')
'''
plt.rcParams["font.size"] = "14"
fig, ax1 = plt.subplots(dpi=200)
l1 = ax1.plot(np.diag(np.abs(omega_canon[no_occ:, no_occ:])), eps_vir, label="MO", color='tab:red')
#plt.plot(range(no_vir), np.flip(np.diag(np.abs(omega_canon[no_occ:, no_occ:]))), label="MO", color='tab:blue')
ax2 = ax1.twinx()
#plt.plot(range(no_vir), np.diag(np.abs(omega_fvno[no_occ:, no_occ:])), label="NO", color='tab:red')
l2 = ax2.plot(np.diag(np.abs(omega_fvno[no_occ:, no_occ:])), eps, label="NO", color='tab:blue')
#ax2.set_ylim(0, 0.0002)
l3 = ax2.plot(np.diag(np.abs(omega_fvnopp[no_occ:, no_occ:])), eps_pnopp, label="NO++", color='tab:orange')
ax2.set_yscale("log")
ax1.set_ylabel('Orbital energy / $E_h$')
ax2.set_ylabel('log(occupation number)')
ax1.set_xlabel('Orbital spatial extent' )
handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.87, 0.89))
#plt.legend([l1, l2, l3], ["MO", "NO", "NO++"])
plt.savefig('orbital_extent_h2o2', bbox_inches='tight')
plt.show()
