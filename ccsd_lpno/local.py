
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

    def build_PNO_lists(self, pno_cut, D, F_vir, str_pair_list=None):
            # Diagonalize pair densities to get PNOs (Q) and occ_nos
            occ_nos = np.zeros((self.no_occ * self.no_occ, self.no_vir))
            Q = np.zeros((self.no_occ * self.no_occ, self.no_vir, self.no_vir))
            print("Average density: {}".format(D))
            for ij in range(self.no_occ * self.no_occ):
                occ_nos[ij], Q[ij] = np.linalg.eigh(D[ij])

            Q = np.load('Q_full.npy')
            #print("The two Qs are the same: {}".format(np.allclose(Q_full, Q)))
            # Truncate each set of pnos by occ no
            s_pairs = np.zeros(self.no_occ * self.no_occ)
            Q_list = []
            avg = 0.0
            for ij in range(self.no_occ * self.no_occ):
                survivors = occ_nos[ij] > pno_cut
                if ij == 0:
                    print("Survivors[0]:\n{}".format(survivors))
                for a in range(self.no_vir):
                    if survivors[a] == True:
                        s_pairs[ij] += 1
                avg += s_pairs[ij]
                rm_pairs = self.no_vir - int(s_pairs[ij])
                Q_list.append(Q[ij, :, :int(s_pairs[ij])])
                if ij == 0:
                    print("Here's occ nums and Q: {}\n{}".format(occ_nos[ij], Q[ij, :, :rm_pairs]))
            
            avg = avg/(self.no_occ * self.no_occ)
            print('Occupation numbers [0]:\n {}'.format(occ_nos[0]))
            print("Numbers of surviving PNOs:\n{}".format(s_pairs))
            print('Average number of PNOs:\n{}'.format(avg))

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
                if ij == 0:
                    print("Here's L: {}".format(L))

            #print("The two Ls are the same: {}".format(np.allclose(L_full, L_list)))
            #print('Q x L:\n{}\n'.format(Q @ L))
            return Q_list, L_list, eps_pno_list

    def pseudoresponse(self, z_ijab):
        Avvoo = contract('ijeb,ae->abij', self.t_ijab, self.A[self.no_occ:, self.no_occ:])
        Avvoo -= contract('mjab,mi->abij', self.t_ijab, self.A[:self.no_occ, :self.no_occ])
        pertbar_ijab = Avvoo.swapaxes(0,2).swapaxes(1,3)
        temp = pertbar_ijab + pertbar_ijab.swapaxes(0, 1).swapaxes(2, 3)
        presp = 2.0 * contract('ijab,ijab->', z_ijab, temp)
        presp -= contract('ijba,ijab->', z_ijab, temp)

    def init_PNOs(self, pno_cut, t_ijab, F_vir, pert=False, str_pair_list=None, A_list=None, Hbar=None):

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
                Abar += Avvoo.swapaxes(0,3).swapaxes(1,2)
                #Abar = contract('ijeb,ae->ijab', t_ijab, A[self.no_occ:, self.no_occ:])
                #Abar -= contract('mjab,mi->ijab', t_ijab, A[:self.no_occ, :self.no_occ])
                #Abar += contract('jiea,be->ijab', t_ijab, A[self.no_occ:, self.no_occ:])
                #Abar -= contract('miba,mj->ijab', t_ijab, A[:self.no_occ, :self.no_occ])

                # Build guess X's
                # X_ijab = Abar_ijab / Hbar_ii + Hbar_jj - Hbar_aa _ Hbar_bb
                X_guess[i] = Abar.copy()
                X_guess[i] /= denom
                D += self.form_density(X_guess[i])
                #print("Density: {}".format(D))
                i += 1
            D /= 3.0
            # Identify weak pairs using MP2 pseudoresponse
            # requires the building of the guess Abar matrix and guess X's
            # Todo

            self.Q_list, self.L_list, self.eps_pno_list = self.build_PNO_lists(pno_cut, D, F_vir, str_pair_list=str_pair_list)

        else:
            print('Pert switch off. Initializing ground PNOs')
            D = self.form_density(t_ijab)
            self.Q_list, self.L_list, self.eps_pno_list = self.build_PNO_lists(pno_cut, D, F_vir, str_pair_list=str_pair_list)

    def increment(self, Ria, Rijab, F_occ): 
    #def increment(self, Rijab, F_occ): 
        # Q[i, b, a] is diff from Q[i, i, b, a]!
        # Update T1s
        '''new_tia = np.zeros((self.no_occ, self.no_vir))
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
       '''
        # Update T2s
        new_tijab = np.zeros((self.no_occ, self.no_occ, self.no_vir, self.no_vir))
        for ij in range(self.no_occ * self.no_occ):
            tmp1 = self.Q_list[ij]
            # Transform Rs using Q
            R2Q = contract('ac,ab,bd->cd', tmp1, Rijab[ij // self.no_occ, ij % self.no_occ], tmp1)
            # Transform RQs using L
            tmp2 = self.L_list[ij]
            R2QL = contract('ac,ab,bd->cd', tmp2, R2Q, tmp2)
            # Use vir orb. energies from semicanonical
            tmp3 = self.eps_pno_list[ij]
            d2_QL = np.zeros((tmp3.shape[0], tmp3.shape[0]))
            for a in range(tmp3.shape[0]):
                for b in range(tmp3.shape[0]):
                    d2_QL[a, b] = F_occ[ij // self.no_occ, ij // self.no_occ ] + F_occ[ij % self.no_occ, ij % self.no_occ] - tmp3[a] - tmp3[b]
            #print('denom in semi-canonical PNO basis:\n{}\n'.format(d_QL.shape))
            T2QL = R2QL / d2_QL
            # Back transform to TQs
            T2Q = contract('ca,ab,db->cd', tmp2, T2QL, tmp2)
            # Back transform to Ts
            new_tijab[ij // self.no_occ, ij % self.no_occ] += contract('ca,ab,db->cd', tmp1, T2Q, tmp1)
            
        #return new_tia, new_tijab
        return new_tijab
