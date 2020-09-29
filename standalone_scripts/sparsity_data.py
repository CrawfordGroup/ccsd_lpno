'''
This script uses psi4 to compute reference
values for CCSD Correlation energy,
dipole polarizability and optical rotation
'''

import numpy as np
import psi4
import argparse
import ccsd_lpno
from opt_einsum import contract
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

parser = argparse.ArgumentParser()
parser.add_argument("--m", default='h2_7', type=str, help="Molecule from mollib")
args = parser.parse_args()

mp2_en_list = {}

psi4.core.clean()

# Set memory
psi4.set_memory('2 GB')
psi4.core.set_output_file('sparsity_data.out', False)
np.set_printoptions(precision=12, threshold=np.inf, linewidth=200, suppress=True)

'''
# Set Psi4 options
geom = ccsd_lpno.mollib.mollib["{}".format(args.m)]
mol = psi4.geometry(geom)


psi4.set_options({'basis': 'cc-pvdz', 'scf_type': 'pk',
                  'freeze_core': 'false', 'e_convergence': 1e-12,
                  'd_convergence': 1e-12, 'save_jk': 'true'})
psi4.set_options({'omega':[589, 'nm'], 'gauge': 'both'})

scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
This was trying to get the converged CCSD T amplitudes from Psi4 and then
rotate them into the PNO basis, which doesn't work
e, wfn = psi4.energy('ccsd', return_wfn=True, ref_wfn=scf_wfn)
t2 = wfn.get_amplitudes()["tIjAb"].to_array()
print("Shape of T2 array: {}".format(t2.shape))

for ij in range(no_occ*no_occ):
    Q_compute = local.Q_list[ij]
    trans_t2[ij] = contract('Aa,ab,bB->AB', Q_compute.T, new_t[ij], Q_compute)

no_occ = scf_wfn.doccpi()[0]
no_vir = scf_wfn.nmo() - scf_wfn.doccpi()[0] - scf_wfn.frzcpi()[0]
local = ccsd_lpno.HelperLocal(no_occ, no_vir)
#This section to compute T2 and X2 for each method, arrays are saved on 
# line 83 of file 'linresp.py'
omega_nm = 589
polar = ccsd_lpno.do_linresp(scf_wfn, omega_nm, mol, pno_cut=0, method='polar', gauge='length', localize=True, pert='mu')

'''
# This section for T2 amplitudes, computed using H2_7
new_t = np.load('T2_can.npy')
trans_t2 = np.load('T2_pno.npy')
transpp_t2 = np.load('T2_pnopp.npy')

# This section for X2 amplitudes, computed using H2_7
new_x = np.load('X2_can_y.npy')
trans_x2 = np.load('X2_pno_y.npy')
transpp_x2 = np.load('X2_pnopp_y.npy')

t_pno_untrans = np.load('T2_pno_untrans.npy')
t_pnopp_untrans = np.load('T2_pnopp_untrans.npy')

print("T2 can and pno_untrans are the same: {}".format(np.allclose(new_t, t_pno_untrans)))
print("T2 can and pnopp_untrans are the same: {}".format(np.allclose(new_t, t_pnopp_untrans)))
print("T2 pno_untrans and pnopp_untrans are the same: {}".format(np.allclose(t_pno_untrans, t_pnopp_untrans)))

t2_canonical_flat = np.ravel(new_t)
t2_pno_flat = np.ravel(trans_t2)
t2_pnopp_flat = np.ravel(transpp_t2)

x2_canonical_flat = np.ravel(new_x)
x2_pno_flat = np.ravel(trans_x2)
x2_pnopp_flat = np.ravel(transpp_x2)

# This section is setting the smallest magnitude elements to 0 so that they don't
# show up on the graph
can_zero = abs(t2_canonical_flat) < 1e-10
pno_zero = abs(t2_pno_flat) < 1e-10
pnopp_zero = abs(t2_pnopp_flat) < 1e-10

canx_zero = abs(x2_canonical_flat) < 1e-10
pnox_zero = abs(x2_pno_flat) < 1e-10
pnoppx_zero = abs(x2_pnopp_flat) < 1e-10

# Since it's a log plot, the elements are set to 1, since log(1) = 0
x2_canonical_flat[canx_zero] = 1
x2_pno_flat[pnox_zero] = 1
x2_pnopp_flat[pnoppx_zero] = 1

t2_canonical_flat[can_zero] = 1
t2_pno_flat[pno_zero] = 1
t2_pnopp_flat[pnopp_zero] = 1


weights = np.zeros_like(t2_canonical_flat) + 1./len(t2_canonical_flat)

fig, axs = plt.subplots(3, sharex=True, sharey=True)
axs[0].hist(np.log10(abs(t2_canonical_flat)), bins=200, weights=weights, histtype='step', label='Unperturbed')
axs[0].hist(np.log10(abs(x2_canonical_flat)), bins=200, weights=weights, histtype='step', label='Perturbed')
axs[1].hist(np.log10(abs(t2_pno_flat)), bins=200, weights=weights, histtype='step', label='Unperturbed')
axs[1].hist(np.log10(abs(x2_pno_flat)), bins=200, weights=weights, histtype='step', label='Perturbed')
axs[2].hist(np.log10(abs(t2_pnopp_flat)), bins=200, weights=weights, histtype='step', label='Unperturbed')
axs[2].hist(np.log10(abs(x2_pnopp_flat)), bins=200, weights=weights, histtype='step', label='Perturbed')
axs[0].set_title('Canonical')
axs[1].set_title('PNO')
axs[2].set_title('PNO++')
plt.legend()
plt.xlim(-10, -2)
plt.ylim(0, 0.02)
plt.show()
