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
                      'freeze_core': 'false', 'e_convergence': 1e-10,
                      'd_convergence': 1e-10, 'save_jk': 'true'})
    psi4.set_options({'omega':[589, 'nm']})

    scf_e, scf_wfn  = psi4.energy('scf', return_wfn=True)
    hcc_mo = ccsd_lpno.HelperCCEnergy(scf_wfn, local=None, pert=None) 
    ccsd_e = hcc_mo.do_CC()
    contrib_mo = 2.0 * contract('ijab,ijab->ij', hcc_mo.MO[:hcc_mo.no_occ, :hcc_mo.no_occ, hcc_mo.no_occ:, hcc_mo.no_occ:], hcc_mo.t_ijab)
    contrib_mo -= contract('ijba,ijab->ij', hcc_mo.MO[:hcc_mo.no_occ, :hcc_mo.no_occ, hcc_mo.no_occ:, hcc_mo.no_occ:], hcc_mo.t_ijab)
    print("MO contributions:\n{}".format(contrib_mo))

    no_occ = scf_wfn.doccpi()[0]
    no_vir = scf_wfn.nmo() - scf_wfn.doccpi()[0] - scf_wfn.frzcpi()[0]
    local_pno = ccsd_lpno.HelperLocal(no_occ, no_vir)
    hcc_pno = ccsd_lpno.HelperCCEnergy(scf_wfn, local=local_pno, pert=None) 
    ccsd_e = hcc_pno.do_CC(local=local_pno)
    #contrib_pno = hcc_pno.e_ij
    #contrib_pno = contract('ijab,ijab->ij', hcc_pno.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], hcc_pno.t_ijab)
    MO_pno = hcc_pno.MO[:no_occ, :no_occ, no_occ:, no_occ:].copy()
    for i in range(no_occ):
        for j in range(no_occ):
            ij = i + no_occ*j
            Q_compute_pno = local_pno.Q_list[ij]
            print(local_pno.Q_list[ij].shape)
            MO_pno[i][j] = contract('Aa,ab,bB->AB', Q_compute_pno.T, MO_pno[i][j], Q_compute_pno)
            hcc_pno.t_ijab[i][j] = contract('Aa,ab,bB->AB', Q_compute_pno.T, hcc_pno.t_ijab[i][j], Q_compute_pno)
    contrib_pno = 2.0 * contract('ijab,ijab->ij', MO_pno, hcc_pno.t_ijab)
    contrib_pno -= contract('ijba,ijab->ij', MO_pno, hcc_pno.t_ijab)
    print("PNO contributions:\n{}".format(contrib_mo))

    #max_ij = np.argmax(np.abs(hcc_pno.e_ij))
    #print("Max_ij here: {}".format(max_ij))
    #Q_compute_pno = local_pno.Q_list[max_ij]
    #t2 = t2[max_ij // no_occ][max_ij % no_occ]
    #t2_pno = contract('Aa,ab,bB->AB', Q_compute_pno.T, t2, Q_compute_pno)
    np.save('{}_contrib.npy'.format(args.m), contrib_mo)
    np.save('{}_contrib_pno.npy'.format(args.m), contrib_pno)
else:
    contrib_mo = np.load(args.load_t2)
    contrib_pno = np.load(args.load_pno)

t2 = np.abs(contrib_mo)
t2 = np.log(t2)
t2_zero_idx = t2 < -12
t2[t2_zero_idx] = -12
t2_pno = np.abs(contrib_pno)
t2_pno = np.log(t2_pno)
t2_pno_zero_idx = t2_pno < -12
t2_pno[t2_pno_zero_idx] = -12
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

#def update(changed_image):
#    for im in images:
#        if (changed_image.get_cmap() != im.get_cmap()
#                or changed_image.get_clim() != im.get_clim()):
#            im.set_cmap(changed_image.get_cmap())
#            im.set_clim(changed_image.get_clim())

#for im in images:
#    im.callbacksSM.connect('changed', update)

cbar_ax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
fig.colorbar(images[0], cax=cbar_ax)
saxs[0].set_title('Canonical MOs')
saxs[1].set_title('PNOs')
plt.savefig("{}_contrib.png".format(args.m),transparent=True, bbox_inches='tight')
plt.show()

