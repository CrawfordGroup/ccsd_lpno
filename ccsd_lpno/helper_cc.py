'''
HelperCCEnergy class definition and function definitions
For RHF CCSD calculations
'''

import numpy as np
import psi4
from . import diis
from .diis import *
from opt_einsum import contract

class HelperCCEnergy(object):
    def __init__(self, local, pno_cut, rhf_wfn, memory=2):
        # Hardcoding weak pair cutoff for now
        e_cut = 1e-4

        # Set energy and wfn from Psi4
        self.wfn = rhf_wfn

        # Get orbital coeffs from wfn
        C = self.wfn.Ca()
        C.print_out()
        self.C_arr = C.to_array()
        self.C_occ = self.wfn.Ca_subset("AO", "ACTIVE_OCC")
        basis = self.wfn.basisset()
        self.no_occ = self.wfn.doccpi()[0]
        self.no_mo = self.wfn.nmo()
        self.no_fz = self.wfn.frzcpi()[0]
        self.no_occ -= self.no_fz
        self.eps = np.asarray(self.wfn.epsilon_a())
        self.J = self.wfn.jk().J()[0].to_array()
        self.K = self.wfn.jk().K()[0].to_array()

        # Get No. of virtuals
        self.no_vir = self.no_mo - self.no_occ - self.no_fz
        print("Checking dimensions of orbitals: no_occ: {}\t no_fz: {}\t no_active: {}\t no_vir: {}\t total: {}".format(self.no_fz+self.no_occ, self.no_fz, self.no_occ, self.no_vir, self.no_occ+self.no_fz+self.no_vir))

        self.mints = psi4.core.MintsHelper(self.wfn.basisset())

        self.H = np.asarray(self.mints.ao_kinetic()) + np.asarray(self.mints.ao_potential())

        self.I = self.mints.ao_eri()
        # Get localized occupied orbitals
        # Make MO integrals
        # Build Fock matrix
        if local:
            local = HelperLocal(ccsd)
            self.MO = local.MO
            self.F = local.F
        else:    
            self.MO_nfz = np.asarray(self.mints.mo_eri(C, C, C, C))
            self.MO = self.MO_nfz[self.no_fz:, self.no_fz:, self.no_fz:, self.no_fz:]
            self.F = self.H + 2.0 * self.J - self.K
            self.F_nfz = contract('uj, vi, uv', C, C, self.F)
            self.F = self.F_nfz[self.no_fz:, self.no_fz:]

        #test = self.H + 2.0 * self.J - self.K
        #test = contract('uj, vi, uv', C, C, test)

        # Need to change ERIs to physicist notation
        self.MO = self.MO.swapaxes(1, 2)

        # Need F_occ and F_vir separate (will need F_vir for semi-canonical basis later)
        self.F_occ = self.F[:self.no_occ, :self.no_occ]
        self.F_vir = self.F[self.no_occ:, self.no_occ:]

        #print("MO basis F_vir:\n{}\n".format(self.F_vir))
        print("MO basis F_occ:\n{}\n".format(self.F_occ))

        # Once localized, the occupied orbital energies are no longer
        # equivalent to the diagonal of the Fock matrix
        self.eps_occ = np.diag(self.F_occ)
        self.eps_vir = np.diag(self.F_vir)

        # init T1s
        self.t_ia = np.zeros((self.no_occ, self.no_vir))

        # init T2s
        # note that occ.transpose(col) - vir(row) gives occ x vir matrix of differences
        self.d_ia = self.eps_occ.reshape(-1, 1) - self.eps_vir
        self.d_ijab = self.eps_occ.reshape(-1, 1, 1, 1) + self.eps_occ.reshape(-1, 1, 1) - self.eps_vir.reshape(-1, 1) - self.eps_vir
        self.t_ijab = self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:].copy()
        # T2s matching!
        #print("T2s[0,0]:\{}".format(self.t_ijab[0,0]))
        self.t_ijab /= self.d_ijab
        #print("denoms[0,0]:\{}".format(self.d_ijab[0,0]))

        if local:
            # Initialize PNOs
            print('Local switch on. Initializing PNOs.')
            
            # Identify weak pairs using MP2 pair corr energy
            e_ij = contract('ijab,ijab->ij', self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], self.t_ijab)
            self.mp2_e  = self.corr_energy(self.t_ia, self.t_ijab)
            #print('MP2 correlation energy: {}\n'.format(self.mp2_e))
            #print('Pair corr energy matrix:\n{}'.format(e_ij))
            str_pair_list = abs(e_ij) > e_cut
            #print('Strong pair list:\n{}'.format(str_pair_list.reshape(1,-1)))
            
            # Create Tij and Ttij
            T_ij = self.t_ijab.copy().reshape((self.no_occ * self.no_occ, self.no_vir, self.no_vir))
            #print('T matrix [0]:\n{}'.format(T_ij[0]))
            Tt_ij = 2.0 * T_ij.copy() 
            Tt_ij -= T_ij.swapaxes(1, 2)


            # Form pair densities
            self.D = np.zeros((self.no_occ * self.no_occ, self.no_vir, self.no_vir))
            for ij in range(self.no_occ * self.no_occ):
                i = ij // self.no_occ
                j = ij % self.no_occ
                self.D[ij] = contract('ab,bc->ac', T_ij[ij], Tt_ij[ij].T) + contract('ab,bc->ac', T_ij[ij].T, Tt_ij[ij])
                self.D[ij] *= 2.0 / (1.0 + int(i==j))
            #print("Density matrix [0, 0]: \n{}".format(self.D[0]))

            # Diagonalize pair densities to get PNOs (Q) and occ_nos
            self.occ_nos = np.zeros((self.no_occ * self.no_occ, self.no_vir))
            self.Q = np.zeros((self.no_occ * self.no_occ, self.no_vir, self.no_vir))
            for ij in range(self.no_occ * self.no_occ):
                    self.occ_nos[ij], self.Q[ij] = np.linalg.eigh(self.D[ij])

            # Truncate each set of pnos by occ no
            self.s_pairs = np.zeros(self.no_occ * self.no_occ)
            self.Q_list = []
            avg = 0.0
            for ij in range(self.no_occ * self.no_occ):
                survivors = self.occ_nos[ij] > pno_cut
                if ij == 0:
                    print("Survivors[0]:\n{}".format(survivors))
                for a in range(self.no_vir):
                    if survivors[a] == True:
                        self.s_pairs[ij] += 1
                avg += self.s_pairs[ij]
                rm_pairs = self.no_vir - int(self.s_pairs[ij])
                self.Q_list.append(self.Q[ij, :, rm_pairs:])
            
            avg = avg/(self.no_occ * self.no_occ)
            print('Occupation numbers [0]:\n {}'.format(self.occ_nos[0]))
            print("Numbers of surviving PNOs:\n{}".format(self.s_pairs))
            print('Average number of PNOs:\n{}'.format(avg))

            # Get semicanonical transforms
                # transform F_vir to PNO basis
                # Diagonalize F_pno, get L
                # save virtual orb. energies
            self.eps_pno_list = [] 
            self.L_list = []
            # For each ij, F_pno is pno x pno dimension
            for ij in range(self.no_occ * self.no_occ):
                tmp1 = self.Q_list[ij]
                self.F_pno = contract('pa,ab,bq->pq', tmp1.swapaxes(0, 1), self.F_vir, tmp1)
                self.eps_pno, L = np.linalg.eigh(self.F_pno)
                self.eps_pno_list.append(self.eps_pno)
                self.L_list.append(L)
            #print('Q x L:\n{}\n'.format(Q @ L))


    # Make intermediates, Staunton:1991 eqns 3-11
    # Spin-adapted, every TEI term is modified to include
    # antisymmetrized term
    def make_taut(self, t_ia, t_ijab):
        tau_t = t_ijab + 0.5 * (contract('ia,jb->ijab', t_ia, t_ia))
        return tau_t


    def make_tau(self, t_ia, t_ijab):
        tau = t_ijab + (contract('ia,jb->ijab', t_ia, t_ia))
        return tau


    def make_Fae(self, taut, t_ia, t_ijab):
        Fae = self.F_vir.copy()
        #Fae[np.diag_indices_from(Fae)] = 0
        Fae -= 0.5 * contract('me,ma->ae', self.F[:self.no_occ, self.no_occ:], t_ia)
        Fae += 2.0 * contract('mf,mafe->ae', t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        Fae -= contract('mf,maef->ae', t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        Fae -= 2.0 * contract('mnaf,mnef->ae', taut, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        Fae += contract('mnaf,mnfe->ae', taut, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Fae


    def make_Fmi(self, taut, t_ia, t_ijab):
        Fmi = self.F_occ.copy()
        #Fmi[np.diag_indices_from(Fmi)] = 0
        Fmi += 0.5 * contract('ie,me->mi', t_ia, self.F[:self.no_occ, self.no_occ:])
        Fmi += 2.0 * contract('ne,mnie->mi', t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:])
        Fmi -= contract('ne,mnei->mi', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        Fmi += 2.0 * contract('inef,mnef->mi', taut, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        Fmi -= contract('inef,mnfe->mi', taut, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Fmi


    def make_Fme(self, t_ia, t_ijab):
        Fme = self.F[:self.no_occ, self.no_occ:].copy()
        Fme += 2.0 * contract('nf,mnef->me', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        Fme -= contract('nf,mnfe->me', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Fme


    def make_Wmnij(self, tau, t_ia, t_ijab):
        Wmnij = self.MO[:self.no_occ, :self.no_occ, :self.no_occ, :self.no_occ].copy()
        Wmnij += contract('je,mnie->mnij', t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:])
        Wmnij += contract('ie,mnej->mnij', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        Wmnij += contract('ijef,mnef->mnij', tau, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Wmnij


    def make_Wmbej(self, t_ia, t_ijab):
        Wmbej = self.MO[:self.no_occ, self.no_occ:, self.no_occ:, :self.no_occ].copy()
        Wmbej += contract('jf,mbef->mbej', t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        Wmbej -= contract('nb,mnej->mbej', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        tmp = 0.5 * t_ijab.copy() + contract('jf,nb->jnfb', t_ia, t_ia)
        Wmbej -= contract('jnfb,mnef->mbej', tmp, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        Wmbej += contract('njfb,mnef->mbej', t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        Wmbej -= 0.5 * contract('njfb,mnfe->mbej', t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Wmbej


    def make_Wmbje(self, t_ia, t_ijab):
        Wmbje = -1.0 * self.MO[:self.no_occ, self.no_occ:, :self.no_occ, self.no_occ:].copy()
        Wmbje -= contract('jf,mbfe->mbje', t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        Wmbje += contract('nb,mnje->mbje', t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:])
        tmp = 0.5 * t_ijab.copy() + contract('jf,nb->jnfb', t_ia, t_ia)
        Wmbje += contract('jnfb,mnfe->mbje', tmp, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Wmbje


    def make_Zmbij(self, tau):
        Zmbij = 0
        Zmbij += contract('mbef,ijef->mbij', self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:], tau)
        return Zmbij


    # Update T1 and T2 amplitudes
    def update_ts(self, local, tau, tau_t, t_ia, t_ijab):

        no_occ = t_ia.shape[0]
        # Build intermediates
        Fae = self.make_Fae(tau_t, t_ia, t_ijab)
        Fmi = self.make_Fmi(tau_t, t_ia, t_ijab)
        Fme = self.make_Fme(t_ia, t_ijab)

        Wmnij = self.make_Wmnij(tau, t_ia, t_ijab)
        Wmbej = self.make_Wmbej(t_ia, t_ijab)
        Wmbje = self.make_Wmbje(t_ia, t_ijab)
        Zmbij = self.make_Zmbij(tau)

        # Create residual T1s
        Ria = self.F[:no_occ, no_occ:].copy()
        Ria += contract('ie,ae->ia', t_ia, Fae)
        Ria -= contract('ma,mi->ia', t_ia, Fmi)
        Ria += 2.0 * contract('imae,me->ia', t_ijab, Fme)
        Ria -= contract('imea,me->ia', t_ijab, Fme)
        Ria -= contract('nf,naif->ia', t_ia, self.MO[:no_occ, no_occ:, :no_occ, no_occ:])
        Ria += 2.0 * contract('nf,nafi->ia', t_ia, self.MO[:no_occ, no_occ:, no_occ:, :no_occ])
        Ria += 2.0 * contract('mief,maef->ia', t_ijab, self.MO[:no_occ, no_occ:, no_occ:, no_occ:])
        Ria -= contract('mife,maef->ia', t_ijab, self.MO[:no_occ, no_occ:, no_occ:, no_occ:])
        Ria -= 2.0 * contract('mnae,nmei->ia', t_ijab, self.MO[:no_occ, :no_occ, no_occ:, :no_occ])
        Ria += contract('mnae,nmie->ia', t_ijab, self.MO[:no_occ, :no_occ, :no_occ, no_occ:])

        # Create residual T2s
        Rijab = self.MO[:no_occ, :no_occ, no_occ:, no_occ:].copy()
        # Term 2
        tmp = contract('ijae,be->ijab', t_ijab, Fae)
        Rijab += tmp 
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)

        tmp = 0.5 * contract('ijae,mb,me->ijab', t_ijab, t_ia, Fme)
        Rijab -= tmp
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 3
        tmp = contract('imab,mj->ijab', t_ijab, Fmi)
        Rijab -= tmp
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        tmp = 0.5 * contract('imab,je,me->ijab', t_ijab, t_ia, Fme)
        Rijab -= tmp 
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 4
        Rijab += contract('mnab,mnij->ijab', tau, Wmnij)
        # Term 5
        Rijab += contract('ijef,abef->ijab', tau, self.MO[no_occ:, no_occ:, no_occ:, no_occ:])
        # Extra term since Wabef is not formed
        tmp = contract('ma,mbij->ijab', t_ia, Zmbij)
        Rijab -= tmp
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 6 # 1
        tmp = contract('imae,mbej->ijab', t_ijab, Wmbej)
        tmp -= contract('imea,mbej->ijab', t_ijab, Wmbej)
        Rijab += tmp
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
        tmp1 = contract('ie,ma,mbej->ijab', t_ia, t_ia, self.MO[:no_occ, no_occ:, no_occ:, :no_occ])
        Rijab -= tmp1
        Rijab -= tmp1.swapaxes(0, 1).swapaxes(2, 3)
        # Term 6 # 2
        tmp = contract('imae,mbej->ijab', t_ijab, Wmbej)
        tmp += contract('imae,mbje->ijab', t_ijab, Wmbje)
        Rijab += tmp
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 6 # 3
        tmp = contract('mjae,mbie->ijab', t_ijab, Wmbje)
        Rijab += tmp
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
        tmp1 = contract('ie,mb,maje->ijab', t_ia, t_ia, self.MO[:no_occ, no_occ:, :no_occ, no_occ:])
        Rijab -= tmp1
        Rijab -= tmp1.swapaxes(0, 1).swapaxes(2, 3)

        # Term 7
        tmp = contract('ie,abej->ijab', t_ia, self.MO[no_occ:, no_occ:, no_occ:, :no_occ])
        Rijab += tmp
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 8
        tmp = contract('ma,mbij->ijab', t_ia, self.MO[:no_occ, no_occ:, :no_occ, :no_occ])
        Rijab -= tmp
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        if local:
            # Q[i, b, a] is diff from Q[i, i, b, a]!

            # Update T1s
            new_tia = t_ia.copy()
            new_tia += Ria / self.d_ia
            '''for i in range(no_occ):
                tmp_Q = Q_list[i*no_occ+i]
                tmp_L = L_list[i*no_occ+i]
                # Transform Rs using Q
                R1Q = contract('b,ba->a', Ria[i], tmp_Q)
                # Transform RQs using L
                R1QL = contract('b,ba->a', R1Q, tmp_L)
                tmp3 = self.eps_pno_list[i*no_occ+i]
                # Use vir orb. energies from semicanonical
                d1_QL = np.zeros(tmp3.shape[0])
                for a in range(tmp3.shape[0]):
                    d1_QL[a] = self.F_occ[i, i] - tmp3[a]
                T1QL = R1QL / d1_QL
                # Back transform to TQs
                T1Q = contract('ab,b->a', tmp_L, T1QL)
                # Back transform to Ts
                new_tia[i] += contract('ab,b->a', tmp_Q, T1Q)
            '''
            # Update T2s
            new_tijab = t_ijab.copy()
            for ij in range(no_occ * no_occ):
                tmp1 = self.Q_list[ij]
                # Transform Rs using Q
                R2Q = contract('ac,ab,bd->cd', tmp1, Rijab[ij // no_occ, ij % no_occ], tmp1)
                # Transform RQs using L
                tmp2 = self.L_list[ij]
                R2QL = contract('ac,ab,bd->cd', tmp2, R2Q, tmp2)
                # Use vir orb. energies from semicanonical
                tmp3 = self.eps_pno_list[ij]
                d2_QL = np.zeros((tmp3.shape[0], tmp3.shape[0]))
                for a in range(tmp3.shape[0]):
                    for b in range(tmp3.shape[0]):
                        d2_QL[a, b] = self.F_occ[ij // no_occ, ij // no_occ ] + self.F_occ[ij % no_occ, ij % no_occ] - tmp3[a] - tmp3[b]
                #print('denom in semi-canonical PNO basis:\n{}\n'.format(d_QL.shape))
                T2QL = R2QL / d2_QL
                # Back transform to TQs
                T2Q = contract('ca,ab,db->cd', tmp2, T2QL, tmp2)
                # Back transform to Ts
                new_tijab[ij // no_occ, ij % no_occ] += contract('ca,ab,db->cd', tmp1, T2Q, tmp1)
        else:
            # Apply denominators
            new_tia =  t_ia.copy() 
            new_tia += Ria / self.d_ia
            new_tijab = t_ijab.copy() 
            new_tijab += Rijab / self.d_ijab

        return new_tia, new_tijab

    # Compute CCSD correlation energy
    def corr_energy(self, t_ia, t_ijab):
        no_occ = t_ia.shape[0] 
        E_corr = 2.0 * contract('ia,ia->', self.F[:no_occ, no_occ:], t_ia)
        tmp_tau = self.make_tau(t_ia, t_ijab)
        E_corr += 2.0 * contract('ijab,ijab->', self.MO[:no_occ, :no_occ, no_occ:, no_occ:], tmp_tau)
        E_corr -= contract('ijba,ijab->', self.MO[:no_occ, :no_occ, no_occ:, no_occ:], tmp_tau)
        return E_corr

    def do_CC(self, local=False, e_conv=1e-8, r_conv=1e-7, maxiter=40, max_diis=8, start_diis=0):
        self.old_e = self.corr_energy(self.t_ia, self.t_ijab)
        print('Iteration\t\t Correlation energy\tDifference\tRMS\nMP2\t\t\t {}'.format(self.old_e))
    # Set up DIIS
        diis = HelperDIIS(self.t_ia, self.t_ijab, max_diis)

    # Iterate until convergence
        for i in range(maxiter):
            tau_t = self.make_taut(self.t_ia, self.t_ijab)
            tau = self.make_tau(self.t_ia, self.t_ijab)
            new_tia, new_tijab = self.update_ts(local, tau, tau_t, self.t_ia, self.t_ijab)
            new_e = self.corr_energy(new_tia, new_tijab)
            rms = np.linalg.norm(new_tia - self.t_ia)
            rms += np.linalg.norm(new_tijab - self.t_ijab)
            print('CC Iteration: {:3d}\t {:2.12f}\t{:1.12f}\t{:1.12f}\tDIIS Size: {}'.format(i, new_e, abs(new_e - self.old_e), rms, diis.diis_size))
            if(abs(new_e - self.old_e) < e_conv and abs(rms) < r_conv):
                print('Convergence reached.\n CCSD Correlation energy: {}\n'.format(new_e))
                self.t_ia = new_tia
                self.t_ijab = new_tijab
                break
            # Update error vectors for DIIS
            diis.update_err_list(new_tia, new_tijab)
            # Extrapolate using DIIS
            if(i >= start_diis):
                new_tia, new_tijab = diis.extrapolate(new_tia, new_tijab)

            self.t_ia = new_tia
            self.t_ijab = new_tijab
            self.old_e = new_e

        return new_e
