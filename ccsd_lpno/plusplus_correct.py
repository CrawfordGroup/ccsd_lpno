import ccsd_lpno
import psi4
import numpy as np
import pprint
from opt_einsum import contract

def compute_correction(t_ijab, A_list, B_list, denom, Q_trunc_list, S_list):
    no_occ = t_ijab.shape[0]
    no_vir = t_ijab.shape[2]
    D_ia = denom[0]
    D_ijab = denom[1]

    # Convert T2s, L2s to trunc PNO basis
    new_t = np.reshape(t_ijab, (self.no_occ*self.no_occ, self.no_vir, self.no_vir))
    trans_t2 = []
    for ij in range(self.no_occ * self.no_occ):
        trans_t = contract('Aa,ab,bB->AB', Q_trunc_list[ij].T, new_t[ij], Q_trunc_list[ij])
        t2_list.append(trans_t)

    l2_list = t2_list.copy() 
    new_tijab = np.reshape(trans_t2, (self.no_occ, self.no_occ))
    new_lijab = new_tijab.copy()
    
    #Using MP2 T2s, create guess X's, Y's for a given direction
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

        #transform x1s and y1s
        x1_list = []
        y1_list = []
        for i in range(self.no_occ):
            tmp_Q = Q_trunc_list[i*self.no_occ+i]
            x1_list.append(contract('Aa,a->A', tmp_Q.T, x_ia[i]))
        y1_list = x1_list.copy()

        Avvoo = contract('ijeb,ae->abij', t_ijab, A[no_occ:, no_occ:])
        Avvoo -= contract('mjab,mi->abij', t_ijab, A[:no_occ, :no_occ])
        Abar = Avvoo.swapaxes(0,2).swapaxes(1,3)
        Abar += Abar.swapaxes(0,1).swapaxes(2,3)
        # X_ijab = Abar_ijab / Hbar_ii + Hbar_jj - Hbar_aa _ Hbar_bb
        x_ijab = Abar.copy()
        x_ijab /= D_ijab
        y_ijab = x_ijab.copy()

        #transform x2s and y2s
        new_x = np.reshape(x_ijab, (self.no_occ*self.no_occ, self.no_vir, self.no_vir))
        x2_list = []
        y2_list = []
        for ij in range(self.no_occ * self.no_occ):
            x2_list.append(contract('Aa,ab,bB->AB', Q_trunc_list[ij].T, new_x[ij], Q_trunc_list[ij]))
        y2_list = x2_list.copy()
        new_xijab = np.reshape(x2_list, (self.no_occ, self.no_occ))
        new_yijab = new_xijab.copy()

        for dirn1 in dirn_list:
            B = B_list[dirn1]
            B_oo = B[:self.no_occ, :self.no_occ]
            # transform B_vo and B_ov
            B_vo_list = []
            B_ov_list = []
            for i in range(self.no_occ):
                tmp_Q = Q_trunc_list[i*self.no_occ+i]
                B_vo_list.append(contract('Aa,a->A', tmp_Q.T, B[self.no_occ:, i]))
                B_ov_list.append(contract('Aa,a->A', tmp_Q.T, B[i, self.no_occ:]))
            
            B_vv_list = []
            for ij in range(self.no_occ * self.no_occ):
                B_vv_list.append(contract('Aa,ab,bB->AB', Q_trunc_list[ij].T, B[no_occ:, no_occ:], Q_trunc_list[ij]))

            linresp = 0.0
            # y1 B_bar
            # y_i^a_ii b_{a_ii}_i
            linresp += contract('ia,ia->', y1_list, b_vo_list)
            # y_i^a_ii (2 S_{a_ii a_mi} t_mi^{e_mi a_mi} S_{e_mi e_mm} b_{m e_mm} - S_{a_ii a_im} t_im^{e_im a_im} S_{e_im e_mm} b_{m e_mm})
            for i in range(self.no_occ):
                ii = i * self.no_occ + i
                for m in range(self.no_occ):
                    mi = m * self.no_occ + i
                    im = i * self.no_occ + m
                    mm = m * self.no_occ + m
                    linresp += 2.0 * contract('a,ab,cb,cd,d->', y1_list[i], S_list[ii][mi], t2_list[mi], S_list[mi][mm], B_ov_list[m])
                    linresp -= contract('a,ab,cb,cd,d->', y1_list[i], S_list[ii][im], t2_list[mi], S_list[mi][mm], B_ov_list[m])
            # y2 B_bar
            # 0.5 y_ij^{a_ij b_ij} t_ij^{e_ij b_ij} b_{a_ij e_ij}
            for ij in range(self.no_occ * self.no_occ):
                linresp += 0.5 * contract('ab,eb,ae->', y2_list(ij), t2_list[ij], b_vv_list[ij])
            # 0.5 y_ij^{a_ij b_ij} S_{a_ij a_ji} t_ji^{e_ji a_ji} b_{b_ji e_ji} S_{b_ji b_ij}
            # - 0.5 y_ij^{a_ij b_ij} S_{a_ij a_mj} t_mj^{a_mj b_mj} S_{b_mj b_ij} b_{mi}
            # - 0.5 y_ij^{a_ij b_ij} S_{a_ij a_mi} t_mi^{b_mi a_mi} S_{b_mi, b_ij} b_mj
            for i in range(self.no_occ):
                for j in range(self.no_occ):
                    ij = i * self.no_occ + j
                    linresp += 0.5 * contract('ab,ac,ec,de,db->', y2_list[ij], S_list[ij][ji], t2_list[ji], b_vv_list[ji], S_list[ji][ij])
                    for m in range(self.no_occ):
                        mj = m * self.no_occ + j
                        mi = m * self.no_occ + i
                        linresp -= 0.5 * contract('ab,ac,cd,db->', y2_list[ij], S_list[ij][mj], t2_list[mj], S_list[mj][ij]) * B[m][i]
                        linresp -= 0.5 * contract('ab,ac,dc,db->', y2_list[ij], S_list[ij][mi], t2_list[mi], S_list[mi][ij]) * B[m][j]
            # B_bar X1
            # x_i^a b_ia
            linresp += 2.0 * contract('ia,ia->', x1_list, B_ov_list)
            # L2 Bbar X1
            # - x_i^{a_ii} l_ij^{b_ij c_ij} S_{b_ij b_mj} S_{c_ij c_mj} t_mj^{b_mj c_mj} b_{m a_mm} S_{a_mm a_ii}
            for i in range(self.no_occ):
                ii = i * self.no_occ + i
                for j in range(self.no_occ):
                    ij = i * self.no_occ + j
                    for m in range(self.no_occ):
                        mj = m * self.no_occ + j
                        mm = m * self.no_occ + m
                        linresp -= contract('a,bc,bd,ce,de,f,fa->', x1_list[i], l2_list[ij], S_list[ij][mj], S_list[ij][mj], t2_list[mj], B_ov_list[m], S_list[mm][ii])
            # - 0.5 x_i^{a_ii} S_{a_ii a_kj} l_kj^{a_kj b_kj} t_kj^{e_kj b_kj} S_{a_kj a_ii} S_{e_kj e_ii} b_{e e_ii}
            # - 0.5 x_i^{a_ii} S_{a_ii a_kj} l_kj^{b_kj a_kj} S_{b_kj b_jk} t_jk^{e_jk b_jk} S_{e_jk e_ii} b_{i e_ii}
            for i in range(self.no_occ):
                for j in range(self.no_occ):
                    for k in range(self.no_occ):

            # <0| L2 B_bar X1 |0>
            temp = contract('ijeb,me->mbij', t_ijab, B[:no_occ, no_occ:])
            temp1 = -1.0 * contract('miab,me->abei', t_ijab, B[:no_occ, no_occ:])
            linresp -= 0.5 * contract('kbij,ka,ijab->', temp, x_ia, l_ijab)
            linresp += contract('bcaj,ia,ijbc->', temp1, x_ia, l_ijab)
            linresp -= 0.5 * contract('kaji,kb,ijab->', temp, x_ia, l_ijab)
            # <0| L2 B_bar X2 |0>
            linresp -= 0.5 * contract('ki,kjab,ijab->', B[:no_occ, :no_occ], x_ijab, l_ijab)
            linresp -= 0.5 * contract('kj,kiba,ijab->', B[:no_occ, :no_occ], x_ijab, l_ijab)
            linresp += 0.5 * contract('ac,ijcb,ijab->', B[no_occ:, no_occ:], x_ijab, l_ijab)
            linresp += 0.5 * contract('bc,ijac,ijab->', B[no_occ:, no_occ:], x_ijab, l_ijab)

            linresp *= -1.0
            print("Here's the contribution for A_{} B_{}: {}".format(dirn, dirn1, linresp))
            if dirn == dirn1:
                total += linresp

    return total    

if __name__ == "__main__":
    psi4.core.clean()

    psi4.core.set_output_file('plusplus_correct.dat', False)
    np.set_printoptions(precision=12, linewidth=200, suppress=True)

    geom = ccsd_lpno.mollib.mollib["h2_2"]
    mol = psi4.geometry(geom)

    psi4.set_options({'basis': 'sto-3g', 'scf_type': 'pk',
                      'freeze_core': 'false', 'e_convergence': 1e-12,
                      'd_convergence': 1e-12, 'save_jk': 'true'})
    
    psi4.set_module_options('SCF', {'E_CONVERGENCE': 1e-12})
    psi4.set_module_options('SCF', {'D_CONVERGENCE': 1e-12})
    e_scf, wfn = psi4.energy('SCF', return_wfn=True)
    print('SCF energy: {}\n'.format(e_scf))
    no_vir = wfn.nmo() - wfn.doccpi()[0] - wfn.frzcpi()[0]

    pert='mu'
    pno_cut=1e-1
    local = ccsd_lpno.HelperLocal(wfn.doccpi()[0], no_vir)
    ppno_correct=True
    hcc = ccsd_lpno.HelperCCEnergy(wfn, local=local, pert=pert, pno_cut=pno_cut, ppno_correction=ppno_correct)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(local.Q_trunc_list)
    S_list = local.build_overlaps(local.Q_trunc_list)

    pp.pprint(S_list)
    #print("List of overlaps: {}".format(S_list))
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

    lin_resp_value = compute_correction(hcc.t_ijab, A_list, B_list, hcc.denom_tuple, local.Q_trunc_list, S_list)
    print("The trace: {}".format(lin_resp_value))
