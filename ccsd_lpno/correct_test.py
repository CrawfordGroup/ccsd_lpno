import ccsd_lpno
import psi4
import numpy as np
from opt_einsum import contract

def full_ext_linresp(t_ijab, A_list, B_list, denom):
    no_occ = t_ijab.shape[0]
    no_vir = t_ijab.shape[2]
    D_ia = denom[0]
    D_ijab = denom[1]

    #Using MP2 T2s, create guess X's, Y's, L's
    t_ia = np.zeros((no_occ, no_vir))
    l_ia = np.zeros((no_occ, no_vir))
    l_ijab = t_ijab
    
    dirn_list = ['X','Y','Z']
    total = 0.0
    for dirn in dirn_list:
        A = A_list[dirn]
        # Make guess Xs
        x_ia = A[no_occ:, :no_occ].copy()
        x_ia += 2.0 * contract('miea,me->ai', t_ijab, A[:no_occ, no_occ:])
        x_ia -= contract('imea,me->ai', t_ijab, A[:no_occ, no_occ:])
        x_ia = x_ia.swapaxes(0,1) / D_ia
        y_ia = x_ia.copy()

        Avvoo = contract('ijeb,ae->abij', t_ijab, A[no_occ:, no_occ:])
        Avvoo -= contract('mjab,mi->abij', t_ijab, A[:no_occ, :no_occ])
        Abar = Avvoo.swapaxes(0,2).swapaxes(1,3)
        Abar += Abar.swapaxes(0,1).swapaxes(2,3)
        # X_ijab = Abar_ijab / Hbar_ii + Hbar_jj - Hbar_aa _ Hbar_bb
        x_ijab = Abar.copy()
        x_ijab /= D_ijab
        y_ijab = x_ijab.copy()


        for dirn1 in dirn_list:
            B = B_list[dirn1]
            linresp = 0.0 
            # <0| B_bar X1 |0>
            linresp += 2.0 * contract('ia,ia->', B[:no_occ, no_occ:], x_ia)
            # <0| L1 B_bar X1 |0>
            linresp += contract('ca,ia,ic->', B[no_occ:, no_occ:], x_ia, l_ia) #*
            linresp -= contract('ik,ia,ka->', B[:no_occ, :no_occ], x_ia, l_ia) #*
            # <0| L2 B_bar X1 |0>
            temp = contract('ijeb,me->mbij', t_ijab, B[:no_occ, no_occ:])
            temp1 = -1.0 * contract('miab,me->abei', t_ijab, B[:no_occ, no_occ:])
            linresp += contract('bcaj,ia,ijbc->', temp1, x_ia, l_ijab)
            linresp -= 0.5 * contract('kbij,ka,ijab->', temp, x_ia, l_ijab)
            linresp -= 0.5 * contract('kaji,kb,ijab->', temp, x_ia, l_ijab)
            # <0| Y1 B_bar |0>
            temp2 = B[no_occ:, :no_occ].copy()
            temp2 += 2.0 * contract('miea,me->ai', t_ijab, B[:no_occ, no_occ:])
            temp2 -= contract('imea,me->ai', t_ijab, B[:no_occ, no_occ:])
            linresp += contract('ai,ia->', temp2, y_ia)
            #print("term {} {}: {}".format(dirn, dirn1, linresp))
            # <0| L1 B_bar X2 |0>
            linresp += 2.0 * contract('jb,ijab,ia->', B[:no_occ, no_occ:], x_ijab, l_ia)
            linresp -= contract('jb,ijba,ia->', B[:no_occ, no_occ:], x_ijab, l_ia)
            # <0| L2 B_bar X2 |0>
            linresp -= 0.5 * contract('ki,kjab,ijab->', B[:no_occ, :no_occ], x_ijab, l_ijab)
            linresp -= 0.5 * contract('kj,kiba,ijab->', B[:no_occ, :no_occ], x_ijab, l_ijab)
            linresp += 0.5 * contract('ac,ijcb,ijab->', B[no_occ:, no_occ:], x_ijab, l_ijab)
            linresp += 0.5 * contract('bc,ijac,ijab->', B[no_occ:, no_occ:], x_ijab, l_ijab)
            #print("Polar2 : {}".format(linresp))
            # <0| Y2 B_bar |0>
            temp3 = contract('ijeb,ae->abij', t_ijab, B[no_occ:, no_occ:])
            temp3 -= contract('mjab,mi->abij', t_ijab, B[:no_occ, :no_occ])
            linresp += 0.5 * contract('abij,ijab->', temp3, y_ijab)
            linresp += 0.5 * contract('baji,ijab->', temp3, y_ijab)

            #print("Singles contribution: {}".format(singles_val))
            #print("Doubles contribution: {}".format(doubles_val))

            linresp *= -1.0
            print("Here's the contribution for A_{} B_{}: {}".format(dirn, dirn1, linresp))
            if dirn == dirn1:
                total += linresp

    return total    

if __name__ == "__main__":
    psi4.core.clean()

    psi4.core.set_output_file('correct_psi4.dat', False)
    np.set_printoptions(precision=12, linewidth=200, suppress=True)

    geom = ccsd_lpno.mollib.mollib["h2o2"]
    mol = psi4.geometry(geom)

    psi4.set_options({'basis': '6-31g', 'scf_type': 'pk',
                      'freeze_core': 'false', 'e_convergence': 1e-12,
                      'd_convergence': 1e-12, 'save_jk': 'true'})
    
    psi4.set_module_options('SCF', {'E_CONVERGENCE': 1e-12})
    psi4.set_module_options('SCF', {'D_CONVERGENCE': 1e-12})
    e_scf, wfn = psi4.energy('SCF', return_wfn=True)
    print('SCF energy: {}\n'.format(e_scf))
    no_vir = wfn.nmo() - wfn.doccpi()[0] - wfn.frzcpi()[0]

    pert='mu'
    pno_cut=0
    local = ccsd_lpno.HelperLocal(wfn.doccpi()[0], no_vir)
    hcc = ccsd_lpno.HelperCCEnergy(wfn, local=local, pert=pert, pno_cut=pno_cut)
    
    # Prepare the perturbations
    A_list = {}
    B_list = {}
    dirn_list = ['X','Y','Z']
    dipole_array = hcc.mints.ao_dipole()
    angular_momentum = hcc.mints.ao_angular_momentum()
    i = 0
    for dirn in dirn_list:
        A_list[dirn] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(dipole_array[i]))
        B_list[dirn] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
        i += 1
    
    #print("Denom tuple: {}".format(hcc.denom_tuple))
    lin_resp_value = full_ext_linresp(hcc.t_ijab, A_list, B_list, hcc.denom_tuple)
    print("The trace: {}".format(lin_resp_value))
