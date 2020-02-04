import ccsd_lpno
import psi4
import numpy as np
import argparse
import json
from opt_einsum import contract
from psi4 import constants as pc

def compute_correction(t_ijab, A_list, B_list, denom, Q_trunc_list, S_list):
    no_occ = t_ijab.shape[0]
    no_vir = t_ijab.shape[2]
    D_ia = denom[0]
    D_ijab = denom[1]

    # Convert T2s, L2s to trunc PNO basis
    new_t = np.reshape(t_ijab, (no_occ*no_occ, no_vir, no_vir))
    t2_list = []
    for ij in range(no_occ * no_occ):
        trans_t = contract('Aa,ab,bB->AB', Q_trunc_list[ij].T, new_t[ij], Q_trunc_list[ij])
        t2_list.append(trans_t)

    l2_list = t2_list.copy() 
    #new_tijab = np.reshape(trans_t2, (no_occ, no_occ))
    #new_lijab = new_tijab.copy()
    
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

        # Transform x1s and y1s
        x1_list = []
        y1_list = []
        for i in range(no_occ):
            tmp_Q = Q_trunc_list[i*no_occ+i]
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

        # Transform x2s and y2s
        new_x = np.reshape(x_ijab, (no_occ*no_occ, no_vir, no_vir))
        x2_list = []
        y2_list = []
        for ij in range(no_occ * no_occ):
            x2_list.append(contract('Aa,ab,bB->AB', Q_trunc_list[ij].T, new_x[ij], Q_trunc_list[ij]))
        y2_list = x2_list.copy()
        #new_xijab = np.reshape(x2_list, (no_occ, no_occ))
        #new_yijab = new_xijab.copy()

        for dirn1 in dirn_list:
            B = B_list[dirn1]
            B_oo = B[:no_occ, :no_occ]

            # Transform B_vo and B_ov
            B_vo_list = []
            B_ov_list = []
            for i in range(no_occ):
                tmp_Q = Q_trunc_list[i*no_occ+i]
                B_vo_list.append(contract('Aa,a->A', tmp_Q.T, B[no_occ:, i]))
                B_ov_list.append(contract('Aa,a->A', tmp_Q.T, B[i, no_occ:]))
            
            B_vv_list = []
            for ij in range(no_occ * no_occ):
                B_vv_list.append(contract('Aa,ab,bB->AB', Q_trunc_list[ij].T, B[no_occ:, no_occ:], Q_trunc_list[ij]))

            linresp = 0.0
            # B_bar X1
            # x_i^a b_ia
            for i in range(no_occ):
                linresp += 2.0 * contract('a,a->', x1_list[i], B_ov_list[i])
            # L2 Bbar X1
            # - x_i^{a_ii} l_ij^{b_ij c_ij} S_{b_ij b_mj} S_{c_ij c_mj} t_mj^{b_mj c_mj} b_{m a_mm} S_{a_mm a_ii}
            for i in range(no_occ):
                ii = i * no_occ + i
                for j in range(no_occ):
                    ij = i * no_occ + j
                    for m in range(no_occ):
                        mj = m * no_occ + j
                        mm = m * no_occ + m
                        linresp -= contract('a,bc,bd,ce,de,f,fa->', x1_list[i], l2_list[ij], S_list[ij][mj], S_list[ij][mj], t2_list[mj], B_ov_list[m], S_list[mm][ii])
            # - 0.5 x_i^{a_ii} S_{a_ii a_kj} l_kj^{a_kj b_kj} t_kj^{e_kj b_kj} S_{e_kj e_ii} b_{i e_ii}
            # - 0.5 x_i^{a_ii} S_{a_ii a_kj} l_kj^{b_kj a_kj} S_{b_kj b_jk} t_jk^{e_jk b_jk} S_{e_jk e_ii} b_{i e_ii}
            for i in range(no_occ):
                ii = i * no_occ + i
                for j in range(no_occ):
                    for k in range(no_occ):
                        kj = k * no_occ + j
                        jk = j * no_occ + k
                        linresp -= 0.5 * contract('a,ab,bc,ec,ef,f->', x1_list[i], S_list[ii][kj], l2_list[kj], t2_list[kj], S_list[kj][ii], B_ov_list[i])
                        linresp -= 0.5 * contract('a,ab,cb,cd,ed,ef,f->', x1_list[i], S_list[ii][kj], l2_list[kj], S_list[kj][jk], t2_list[jk], S_list[jk][ii], B_ov_list[i])
            # Y1 B_bar
            # y_i^a_ii b_{a_ii}_i
            for i in range(no_occ):
                linresp += contract('a,a->', y1_list[i], B_vo_list[i])
            # y_i^a_ii (2 S_{a_ii a_mi} t_mi^{e_mi a_mi} S_{e_mi e_mm} b_{m e_mm} - S_{a_ii a_im} t_im^{e_im a_im} S_{e_im e_mm} b_{m e_mm})
            for i in range(no_occ):
                ii = i * no_occ + i
                for m in range(no_occ):
                    mi = m * no_occ + i
                    im = i * no_occ + m
                    mm = m * no_occ + m
                    linresp += 2.0 * contract('a,ab,cb,cd,d->', y1_list[i], S_list[ii][mi], t2_list[mi], S_list[mi][mm], B_ov_list[m])
                    linresp -= contract('a,ab,cb,cd,d->', y1_list[i], S_list[ii][im], t2_list[im], S_list[im][mm], B_ov_list[m])
            #print("term {} {}: {}".format(dirn, dirn1, linresp))
            # L2 B_bar X2
            # 0.5 * x_ij^{a_ij b_ij} (l_ij^{c_ij b_ij} b_{c_ij a_ij} + l_ij^{a_ij c_ij} b_{c_ij b_ij})
            # - 0.5 * x_ij^{a_ij b_ij} S_{a _ij a_kj} S_{b_ij b_kj} l_kj^{a_kj b_kj} b_ik
            # - 0.5 * x_ij^{a_ij b_ij} S_{a _ij a_ik} S_{b_ij b_ik} l_ik^{a_ik b_ik} b_jk
            for i in range(no_occ):
                for j in range(no_occ):
                    ij = i * no_occ + j
                    linresp += 0.5 * contract('ab,cb,ca->', x2_list[ij], l2_list[ij], B_vv_list[ij])
                    linresp += 0.5 * contract('ab,ac,cb->', x2_list[ij], l2_list[ij], B_vv_list[ij])
                    for k in range(no_occ):
                        kj = k * no_occ + j
                        jk = j * no_occ + k
                        ik = i * no_occ + k
                        linresp -= 0.5 * contract('ab,ac,bd,cd->', x2_list[ij], S_list[ij][kj], S_list[ij][kj], l2_list[kj]) * B_oo[i][k]
                        linresp -= 0.5 * contract('ab,ac,bd,cd->', x2_list[ij], S_list[ij][ik], S_list[ij][ik], l2_list[ik]) * B_oo[j][k]

            # Y2 B_bar
            # 0.5 y_ij^{a_ij b_ij} t_ij^{e_ij b_ij} b_{a_ij e_ij}
            for ij in range(no_occ * no_occ):
                linresp += 0.5 * contract('ab,eb,ae->', y2_list[ij], t2_list[ij], B_vv_list[ij])
            # 0.5 y_ij^{a_ij b_ij} S_{a_ij a_ji} t_ji^{e_ji a_ji} b_{b_ji e_ji} S_{b_ji b_ij}
            # - 0.5 y_ij^{a_ij b_ij} S_{a_ij a_mj} t_mj^{a_mj b_mj} S_{b_mj b_ij} b_{mi}
            # - 0.5 y_ij^{a_ij b_ij} S_{a_ij a_mi} t_mi^{b_mi a_mi} S_{b_mi, b_ij} b_mj
            for i in range(no_occ):
                for j in range(no_occ):
                    ij = i * no_occ + j
                    ji = j * no_occ + i
                    linresp += 0.5 * contract('ab,ac,ec,de,db->', y2_list[ij], S_list[ij][ji], t2_list[ji], B_vv_list[ji], S_list[ji][ij])
                    for m in range(no_occ):
                        mj = m * no_occ + j
                        mi = m * no_occ + i
                        linresp -= 0.5 * contract('ab,ac,cd,db->', y2_list[ij], S_list[ij][mj], t2_list[mj], S_list[mj][ij]) * B_oo[m][i]
                        linresp -= 0.5 * contract('ab,ac,dc,db->', y2_list[ij], S_list[ij][mi], t2_list[mi], S_list[mi][ij]) * B_oo[m][j]
                        
            linresp *= -1.0
            print("Here's the contribution for A_{} B_{}: {}".format(dirn, dirn1, linresp))
            if dirn == dirn1:
                total += linresp

    return total    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--j", default='output.json', type=str, help="Output json filename")
    parser.add_argument("--m", default='h2_2', type=str, help="Molecule from mollib")
    parser.add_argument("--gauge", default='both', type=str, help="Gauge for OR calculation")
    args = parser.parse_args()

    psi4.core.clean()

    psi4.core.set_output_file('plusplus_correct.dat', False)
    np.set_printoptions(precision=12, linewidth=200, suppress=True)

    geom = ccsd_lpno.mollib.mollib["{}".format(args.m)]
    mol = psi4.geometry(geom)

    psi4.set_options({'basis': 'aug-cc-pVDZ', 'scf_type': 'pk',
                      'freeze_core': 'false', 'e_convergence': 1e-12,
                      'd_convergence': 1e-12, 'save_jk': 'true'})
    
    psi4.set_module_options('SCF', {'E_CONVERGENCE': 1e-12})
    psi4.set_module_options('SCF', {'D_CONVERGENCE': 1e-12})
    e_scf, wfn = psi4.energy('SCF', return_wfn=True)
    print('SCF energy: {}\n'.format(e_scf))
    no_vir = wfn.nmo() - wfn.doccpi()[0] - wfn.frzcpi()[0]
    
    optrot_lg_list = {}
    optrot_mvg_list = {}
    polar_list = {}
    # Setting cutoffs and wavelength in nm
    cutoffs = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 5e-9, 5e-8, 5e-7, 5e-6, 5e-5, 5e-4]
    omega_nm = 589

    for cut in cutoffs:
        pert='mu'
        pno_cut=cut
        local = ccsd_lpno.HelperLocal(wfn.doccpi()[0], no_vir)
        ppno_correct=True
        hcc = ccsd_lpno.HelperCCEnergy(wfn, local=local, pert=pert, pno_cut=pno_cut, ppno_correction=ppno_correct)
        S_list = local.build_overlaps(local.Q_trunc_list)
        #print("List of overlaps: {}".format(S_list))
        
        # Length gauge
        if args.gauge == 'polar':
            # Prepare the perturbations
            A_list = {}
            dirn_list = ['X','Y','Z']
            dipole_array = hcc.mints.ao_dipole()
            i = 0
            for dirn in dirn_list:
                A_list[dirn] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(dipole_array[i]))
                i += 1

            #print("Denom tuple: {}".format(hcc.denom_tuple))
            lin_resp_value = compute_correction(hcc.t_ijab, A_list, A_list, hcc.denom_tuple, local.Q_trunc_list, S_list)
            
            trace = lin_resp_value / 3.0
            print("The trace: {}".format(lin_resp_value))
            print("Polarizability correction (LG) ({} nm): {}".format(omega_nm, trace))
            polar_list['{}'.format(cut)] = trace

        if args.gauge == 'both':
            # Prepare the perturbations
            A_list = {}
            B_list = {}
            dirn_list = ['X','Y','Z']
            dipole_array = hcc.mints.ao_dipole()
            angular_momentum = hcc.mints.ao_angular_momentum()
            i = 0
            for dirn in dirn_list:
                A_list[dirn] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(dipole_array[i]))
                B_list[dirn] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                i += 1

            #print("Denom tuple: {}".format(hcc.denom_tuple))
            lin_resp_value = compute_correction(hcc.t_ijab, A_list, B_list, hcc.denom_tuple, local.Q_trunc_list, S_list)
            lin_resp_value -= compute_correction(hcc.t_ijab, B_list, A_list, hcc.denom_tuple, local.Q_trunc_list, S_list)
            lin_resp_value *= 0.5
            
            trace = lin_resp_value / 3.0
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            hbar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * hbar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            omega = (pc.c * pc.h * 1e9) / (pc.hartree2J * omega_nm) 
            optrot_lg = prefactor * trace * omega
            print("The trace: {}".format(lin_resp_value))
            print("Optical rotation correction (LG) ({} nm): {}".format(omega_nm, optrot_lg))
            optrot_lg_list['{}'.format(cut)] = optrot_lg

            p_array = hcc.mints.ao_nabla()
            i = 0
            for dirn in dirn_list:
                A_list[dirn] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(p_array[i]))
                B_list[dirn] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                i += 1
                
            lin_resp_value = compute_correction(hcc.t_ijab, A_list, B_list, hcc.denom_tuple, local.Q_trunc_list, S_list)
            lin_resp_value += compute_correction(hcc.t_ijab, B_list, A_list, hcc.denom_tuple, local.Q_trunc_list, S_list)
            lin_resp_value *= 0.5

            trace = lin_resp_value / 3.0
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            hbar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * hbar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            optrot_vg = prefactor * trace
            print("The trace: {}".format(lin_resp_value))
            print("Optical rotation correction (MVG) ({} nm): {}".format(omega_nm, optrot_vg))
            optrot_mvg_list['{}'.format(cut)] = optrot_vg

    optrot_data = {}
    #optrot_data['LG'] = optrot_lg_list
    #optrot_data['MVG'] = optrot_mvg_list
    optrot_data['polar'] = polar_list
    with open("{}".format(args.j), "w") as write_file:
        json.dump(optrot_data, write_file, indent=4)

    #print("List of optical rotations (LG, {} nm): {}".format(omega_nm, optrot_lg_list))
    #print("List of optical rotations (MVG, {} nm): {}".format(omega_nm, optrot_mvg_list))
    print("List of polarizabilities (LG, {} nm): {}".format(omega_nm, polar_list))
