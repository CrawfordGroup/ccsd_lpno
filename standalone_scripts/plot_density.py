'''
This script file plots density matrices,
first in the PNO basis and then in the PNO++
basis
'''

import psi4
import numpy as np
import argparse
import ccsd_lpno
from opt_einsum import contract
from matplotlib import pyplot as plt
from matplotlib import colors

parser = argparse.ArgumentParser()
parser.add_argument("--m", default='h2_7', type=str, help="Molecule from mollib")
parser.add_argument("--load_t2", default=None, type=str, help="File to load t2 from")
parser.add_argument("--load_pno", default=None, type=str, help="File to load pno t2 from")
args = parser.parse_args()

if args.load_t2 is None:
    psi4.core.clean()
    geom = ccsd_lpno.mollib.mollib["{}".format(args.m)]
    mol = psi4.geometry(geom)

    psi4.set_options({'basis': 'cc-pvdz', 'scf_type': 'pk',
                      'freeze_core': 'false', 'e_convergence': 1e-12,
                      'd_convergence': 1e-12, 'save_jk': 'true'})
    psi4.set_options({'omega':[589, 'nm']})

    scf_e, scf_wfn  = psi4.energy('scf', return_wfn=True)
    e, wfn = psi4.energy('ccsd', return_wfn=True, ref_wfn=scf_wfn)
    t2 = wfn.get_amplitudes()["tIjAb"].to_array()  
    print("Shape of T2 array: {}".format(t2.shape))

    no_occ = scf_wfn.doccpi()[0]
    no_vir = scf_wfn.nmo() - scf_wfn.doccpi()[0] - scf_wfn.frzcpi()[0]
    local_pno = ccsd_lpno.HelperLocal(no_occ, no_vir)
    hcc_pno = ccsd_lpno.HelperCCEnergy(scf_wfn, local=local_pno, pert=None) 

    max_ij = np.argmax(np.abs(hcc_pno.e_ij))
    print("Max_ij here: {}".format(max_ij))
    Q_compute_pno = local_pno.Q_list[max_ij]
    t2 = t2[max_ij // no_occ][max_ij % no_occ]
    t2_pno = contract('Aa,ab,bB->AB', Q_compute_pno.T, t2, Q_compute_pno)
    np.save('h2_7_t2.npy', t2)
    np.save('h2_7_t2_pno.npy', t2_pno)
else:
    t2 = np.load(args.load_t2)
    t2_pno = np.load(args.load_pno)

t2 = np.abs(t2)
t2 = np.log(t2)
t2_zero_idx = t2 < -10
t2[t2_zero_idx] = -10
t2_pno = np.abs(t2_pno)
t2_pno = np.log(t2_pno)
t2_pno_zero_idx = t2_pno < -10
t2_pno[t2_pno_zero_idx] = -10
#print(t2.shape)
#print(t2)
#print(t2_pno)
fig, saxs = plt.subplots(1,2)
ax = plt.axes([0,0,1,1], frameon=False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
images = []
images.append(saxs[1].imshow(t2_pno,interpolation='nearest'))
images.append(saxs[0].imshow(t2,interpolation='nearest'))
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)
fig.colorbar(images[1])

def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())


for im in images:
    im.callbacksSM.connect('changed', update)

saxs[0].set_title('Canonical MOs')
saxs[1].set_title('PNOs')
plt.savefig("both.png",transparent=True, bbox_inches='tight')
plt.show()

