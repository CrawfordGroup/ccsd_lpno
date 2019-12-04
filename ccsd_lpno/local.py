
import numpy as np
import psi4
from opt_einsum import contract


class HelperLocal(object):
    def __init__(self, no_occ, no_vir):
        # init
        self.no_occ = no_occ
        self.no_vir = no_vir

    def form_density(self, t_ijab):
        # Create Tij and Ttij
        T_ij = t_ijab.copy().reshape((self.no_occ * self.no_occ, self.no_vir, self.no_vir))
        #print('T matrix [0]:\n{}'.format(T_ij[0]))
        Tt_ij = 2.0 * T_ij.copy() 
        Tt_ij -= T_ij.swapaxes(1, 2)
        #print("X here: {}".format(T_ij))
        #print("Xt here: {}".format(Tt_ij))

        # Form pair densities
        D = np.zeros((self.no_occ * self.no_occ, self.no_vir, self.no_vir))
        for ij in range(self.no_occ * self.no_occ):
            i = ij // self.no_occ
            j = ij % self.no_occ
            D[ij] = contract('ab,bc->ac', T_ij[ij], Tt_ij[ij].T) + contract('ab,bc->ac', T_ij[ij].T, Tt_ij[ij])
            D[ij] *= 2.0 / (1.0 + int(i==j))
            D[ij] += D[ij].T
            D[ij] *= 0.5
        #print("Density matrix [0, 0]: \n{}".format(self.D[0]))
        return D

    def form_semicanonical(self, Q_list, F_vir):
        # Get semicanonical transforms
            # transform F_vir to PNO basis
            # Diagonalize F_pno, get L
            # save virtual orb. energies
        eps_pno_list = [] 
        L_list = []
        # For each ij, F_pno is pno x pno dimension
        for ij in range(self.no_occ * self.no_occ):
            tmp1 = Q_list[ij]
            F_pno = contract('pa,ab,bq->pq', tmp1.swapaxes(0, 1), F_vir, tmp1)
            eps_pno, L = np.linalg.eigh(F_pno)
            eps_pno_list.append(eps_pno)
            L_list.append(L)
            #if ij == 0:
            #    print("Here's L: {}".format(L))
        return L_list, eps_pno_list

    def build_PNO_lists(self, pno_cut, D, str_pair_list=None):
        # Diagonalize pair densities to get PNOs (Q) and occ_nos
        occ_nos = np.zeros((self.no_occ * self.no_occ, self.no_vir))
        self.Q = np.zeros((self.no_occ * self.no_occ, self.no_vir, self.no_vir))
        #print("Average density: {}".format(D[0]))
        #print("numpy version: {}". format(np.__version__))
        for ij in range(self.no_occ * self.no_occ):
            occ_nos[ij], self.Q[ij] = np.linalg.eigh(D[ij])
            #if ij == 0:
            #    print("Here's occ nums and Q: {}\n{}".format(occ_nos[ij], Q[ij]))

        #Q_full = np.load('Q_full.npy')
        #print("The two Qs are the same: {}".format(np.allclose(Q_full, Q)))
        # Truncate each set of pnos by occ no
        self.s_pairs = np.zeros(self.no_occ * self.no_occ)
        Q_list = []
        avg = 0.0
        sq_avg = 0.0
        for ij in range(self.no_occ * self.no_occ):
            survivors = occ_nos[ij] > pno_cut
            #if ij == 0:
            #    print("Survivors[0]:\n{}".format(survivors))
            for a in range(self.no_vir):
                if survivors[a] == True:
                    self.s_pairs[ij] += 1
            avg += self.s_pairs[ij]
            sq_avg += self.s_pairs[ij] * self.s_pairs[ij]
            rm_pairs = self.no_vir - int(self.s_pairs[ij])
            Q_list.append(self.Q[ij, :, rm_pairs:])
            #if ij == 5:
            #    print("Here's occ nums and Q: {}\n{}".format(occ_nos[ij], Q[ij]))
            #    print("Here's occ nums and Q: {}\n{}".format(occ_nos[ij], Q[ij, :, rm_pairs:]))
        
        print("Tcut_PNO : {}".format(pno_cut))
        print("Total no. of PNOs: {}".format(avg))
        print("T2 ratio: {}".format(sq_avg/(self.no_occ * self.no_occ * self.no_vir * self.no_vir)))
        avg = avg/(self.no_occ * self.no_occ)
        print('Occupation numbers [0]:\n {}'.format(occ_nos[0]))
        print("Numbers of surviving PNOs:\n{}".format(self.s_pairs))
        print('Average number of PNOs:\n{}'.format(avg))

        return Q_list

    def pseudoresponse(self, z_ijab):
        Avvoo = contract('ijeb,ae->abij', self.t_ijab, self.A[self.no_occ:, self.no_occ:])
        Avvoo -= contract('mjab,mi->abij', self.t_ijab, self.A[:self.no_occ, :self.no_occ])
        pertbar_ijab = Avvoo.swapaxes(0,2).swapaxes(1,3)
        temp = pertbar_ijab + pertbar_ijab.swapaxes(0, 1).swapaxes(2, 3)
        presp = 2.0 * contract('ijab,ijab->', z_ijab, temp)
        presp -= contract('ijba,ijab->', z_ijab, temp)

    def init_PNOs(self, pno_cut, t_ijab, F_vir, pert=None, str_pair_list=None, A_list=None, Hbar=None):

        if pert:
            print('Pert switch on. Initializing pert PNOs')

            Hbar_ii = Hbar[:self.no_occ]
            Hbar_aa = Hbar[self.no_occ:]

            X_guess = {}
            i = 0
            D = np.zeros((self.no_occ * self.no_occ, self.no_vir, self.no_vir))
            denom = Hbar_ii.reshape(-1, 1, 1, 1) + Hbar_ii.reshape(-1, 1, 1) - Hbar_aa.reshape(-1, 1) - Hbar_aa

            for A in A_list.values():
                # Build guess Abar
                # Abar_ijab = P_ij^ab (t_ij^eb A_ae - t_mj^ab A_mi)
                Avvoo = contract('ijeb,ae->abij', t_ijab, A[self.no_occ:, self.no_occ:])
                Avvoo -= contract('mjab,mi->abij', t_ijab, A[:self.no_occ, :self.no_occ])
                Abar = Avvoo.swapaxes(0,2).swapaxes(1,3)
                Abar += Abar.swapaxes(0,1).swapaxes(2,3)

                # Build guess X's
                # X_ijab = Abar_ijab / Hbar_ii + Hbar_jj - Hbar_aa _ Hbar_bb
                X_guess[i] = Abar.copy()
                X_guess[i] /= denom

                D += self.form_density(X_guess[i])
                i += 1
            #print("X_guess [0]: {}".format(X_guess[0]))
            D /= 3.0
            #print('Average density: {}'.format(D))
            # Identify weak pairs using MP2 pseudoresponse
            # requires the building of the guess Abar matrix and guess X's
            # Todo

            if pert == 'mu' or pert == 'l':
                self.Q_list = self.build_PNO_lists(pno_cut, D, str_pair_list=str_pair_list)
            if pert == 'mu+unpert' or pert == 'l+unpert':
                D_unpert = self.form_density(t_ijab)
                self.Q_list = self.combine_PNO_lists(pno_cut, D, D_unpert, str_pair_list=str_pair_list)
        else:
            print('Pert switch off. Initializing ground PNOs')
            D = self.form_density(t_ijab)
            self.Q_list = self.build_PNO_lists(pno_cut, D, str_pair_list=str_pair_list)
        self.L_list, self.eps_pno_list = self.form_semicanonical(self.Q_list, F_vir)

    def increment(self, Ria, Rijab, F_occ): 
    #def increment(self, Rijab, F_occ): 
        # Q[i, b, a] is diff from Q[i, i, b, a]!
        # Update T1s
        new_tia = np.zeros((self.no_occ, self.no_vir))
        for i in range(self.no_occ):
            tmp_Q = self.Q_list[i*self.no_occ+i]
            tmp_L = self.L_list[i*self.no_occ+i]
            # Transform Rs using Q
            #R1Q = contract('b,ba->a', Ria[i], tmp_Q)
            R1Q = contract('ab,b->a', tmp_Q.T, Ria[i])
            # Transform RQs using L
            #R1QL = contract('b,ba->a', R1Q, tmp_L)
            R1QL = contract('ab,b->a', tmp_L.T, R1Q)
            tmp_e = self.eps_pno_list[i*self.no_occ+i]
            # Use vir orb. energies from semicanonical
            d1_QL = np.zeros(tmp_e.shape[0])
            for a in range(tmp_e.shape[0]):
                d1_QL[a] = F_occ[i, i] - tmp_e[a]
            T1QL = R1QL / d1_QL
            # Back transform to TQs
            T1Q = contract('ab,b->a', tmp_L, T1QL)
            # Back transform to Ts
            new_tia[i] += contract('ab,b->a', tmp_Q, T1Q)
       
        # Update T2s
        new_tijab = np.zeros((self.no_occ, self.no_occ, self.no_vir, self.no_vir))
        for ij in range(self.no_occ * self.no_occ):
            tmp1 = self.Q_list[ij]
            # Transform Rs using Q
            R2Q = contract('ca,ab,bd->cd', tmp1.T, Rijab[ij // self.no_occ, ij % self.no_occ], tmp1)
            # Transform RQs using L
            tmp2 = self.L_list[ij]
            R2QL = contract('ca,ab,bd->cd', tmp2.T, R2Q, tmp2)
            # Use vir orb. energies from semicanonical
            tmp3 = self.eps_pno_list[ij]
            d2_QL = np.zeros((tmp3.shape[0], tmp3.shape[0]))
            for a in range(tmp3.shape[0]):
                for b in range(tmp3.shape[0]):
                    d2_QL[a, b] = F_occ[ij // self.no_occ, ij // self.no_occ ] + F_occ[ij % self.no_occ, ij % self.no_occ] - tmp3[a] - tmp3[b]
            #print('denom in semi-canonical PNO basis:\n{}\n'.format(d_QL.shape))
            T2QL = R2QL / d2_QL
            # Back transform to TQs
            T2Q = contract('ca,ab,bd->cd', tmp2, T2QL, tmp2.T)
            # Back transform to Ts
            new_tijab[ij // self.no_occ, ij % self.no_occ] += contract('ca,ab,bd->cd', tmp1, T2Q, tmp1.T)
            
        return new_tia, new_tijab
        #return new_tijab

    def PNO_correction(self, t_ijab, MO):
        total = 0
        new_MO = np.reshape(MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], (self.no_occ*self.no_occ, self.no_vir, self.no_vir))
        new_t = np.reshape(t_ijab, (self.no_occ*self.no_occ, self.no_vir, self.no_vir))
        for ij in range(self.no_occ * self.no_occ):
            rm_pairs = self.no_vir - int(self.s_pairs[ij])
            print("rm_pairs: {}".format(rm_pairs))
            if rm_pairs == 0:
                continue
            Q_compute = self.Q[ij, :, :rm_pairs]
            trans_MO = contract('Aa,ab,bB->AB', Q_compute.T, new_MO[ij], Q_compute)
            trans_t = contract('Aa,ab,bB->AB', Q_compute.T, new_t[ij], Q_compute)
            print("Shapes: {} \n{}".format(trans_MO.shape, trans_t.shape))
            total += 2.0 * contract('ab,ab->', trans_MO, trans_t)
            total -= contract('ba,ab->', trans_MO, trans_t)
        return total

    def combine_PNO_lists(self, pno_cut, D, D_unpert, str_pair_list=None):
        try:
            print("Unpert PNO cutoff: {}".format(pno_cut[0]))
            Q_pert = self.build_PNO_lists(pno_cut[0], D, str_pair_list=str_pair_list)
            print("Pert PNO cutoff: {}".format(pno_cut[1]))
            Q_unpert = self.build_PNO_lists(pno_cut[1], D_unpert, str_pair_list=str_pair_list)
            Q_list = []
        except:
            print("PNO cut is not a list with the right dimensions.")
        for ij in range(self.no_occ * self.no_occ):
            print("Shapes of Q_pert[{}] and Q_unpert[{}]: ({},{}) and ({},{})".format(ij, ij, len(Q_pert[ij]), len(Q_pert[ij][0]), len(Q_unpert[ij]), len(Q_unpert[ij][0]))) 
            Q_combined = np.hstack((Q_pert[ij], Q_unpert[ij]))
            print("Shape of Q_combined[{}]: {}".format(ij, Q_combined.shape))
            print("Norm list [{}]: {}".format(ij, np.linalg.norm(Q_combined, axis=0)))
            Q_ortho, trash = np.linalg.qr(Q_combined)
            print("Norm list [{}]: {}".format(ij, np.linalg.norm(Q_ortho, axis=0)))
            Q_list.append(Q_ortho)

        return Q_list
