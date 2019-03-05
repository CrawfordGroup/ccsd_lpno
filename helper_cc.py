'''
HelperCCEnergy class definition and function definitions
For RHF CCSD calculations
'''

import numpy as np
import psi4
from ndot import ndot
from diis import *

class HelperCCEnergy(object):
    def __init__(self, local, pno_cut, rhf_wfn, memory=2):
        # Hardcoding weak pair cutoff for now
        e_cut = 1e-4

        # Set energy and wfn from Psi4
        self.wfn = rhf_wfn

        # Get orbital coeffs from wfn
        C = self.wfn.Ca()
        C.print_out()
        c_arr = C.to_array()
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

        mints = psi4.core.MintsHelper(self.wfn.basisset())

        self.H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

        I = mints.ao_eri()
        # Get localized occupied orbitals
        # Make MO integrals
        if local:
            Local = psi4.core.Localizer.build("BOYS", basis, self.C_occ)
            Local.localize()
            new_C_occ = Local.L
            nc_arr = C.to_array()
            nco_arr = new_C_occ.to_array()
            print("Checking dimensions of localized active occupied:\nShape of local array: {}\nShape of C: {}\n".format(nco_arr.shape, nc_arr.shape))
            nc_arr[:, self.no_fz:(self.no_fz + self.no_occ)] = nco_arr[:,:]
            new_C = psi4.core.Matrix.from_array(nc_arr)
            print("Shape of new MO coeff matrix: {}".format(new_C.shape))
            self.MO_nfz = np.asarray(mints.mo_eri(new_C, new_C, new_C, new_C))
            self.MO = self.MO_nfz[self.no_fz:, self.no_fz:, self.no_fz:, self.no_fz:]
            print("Checking size of ERI tensor: {}".format(self.MO.shape))
            print("Checking size of ERI_nfz tensor: {}".format(self.MO_nfz.shape))
        else:    
            self.MO_nfz = np.asarray(mints.mo_eri(C, C, C, C))
            self.MO = self.MO_nfz[self.no_fz:, self.no_fz:, self.no_fz:, self.no_fz:]


        # Build Fock matrix
        if local:
            De = np.einsum('ui,vi->uv', nc_arr[:, :(self.no_fz+self.no_occ)], nc_arr[:, :(self.no_fz+self.no_occ)])
            self.F = self.H + 2.0 * np.einsum('pqrs,rs->pq', I, De) - np.einsum('prqs,rs->pq', I, De)
            self.F_nfz = np.einsum('uj, vi, uv', new_C, new_C, self.F)
            self.F = self.F_nfz[self.no_fz:, self.no_fz:]
            print("Checking size of Fock matrix: {}".format(self.F.shape))
            self.F_ao = self.H + 2.0 * np.einsum('pqrs,rs->pq', I, De) - np.einsum('prqs,rs->pq', I, De)
            hf_e = np.einsum('pq,pq->', self.H + self.F_ao, De)
            print("Hartree-Fock energy: {}".format(hf_e +33.35807208233505))
        else:
            self.F = self.H + 2.0 * self.J - self.K
            self.F_nfz = np.einsum('uj, vi, uv', C, C, self.F)
            self.F = self.F_nfz[self.no_fz:, self.no_fz:]

        #test = self.H + 2.0 * self.J - self.K
        #test = np.einsum('uj, vi, uv', C, C, test)


        # Need to change ERIs to physicist notation
        self.MO = self.MO.swapaxes(1, 2)
        self.MO_nfz = self.MO_nfz.swapaxes(1, 2)

        # Need F_occ and F_vir separate (will need F_vir for semi-canonical basis later)
        self.F_occ = self.F[:self.no_occ, :self.no_occ]
        self.F_vir = self.F[self.no_occ:, self.no_occ:]

        #print("MO basis F_vir:\n{}\n".format(self.F_vir))
        print("MO basis F_occ:\n{}\n".format(self.F_occ))
        #self.eps_occ = self.eps[:self.no_occ]
        #self.eps_vir = self.eps[self.no_occ:]
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
            e_ij = ndot('ijab,ijab->ij', self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], self.t_ijab)
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
                self.D[ij] = np.einsum('ab,bc->ac', T_ij[ij], Tt_ij[ij].T) + np.einsum('ab,bc->ac', T_ij[ij].T, Tt_ij[ij])
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
                self.F_pno = np.einsum('pa,ab,bq->pq', tmp1.swapaxes(0, 1), self.F_vir, tmp1)
                self.eps_pno, L = np.linalg.eigh(self.F_pno)
                self.eps_pno_list.append(self.eps_pno)
                self.L_list.append(L)
            #print('Q x L:\n{}\n'.format(Q @ L))


    # Make intermediates, Staunton:1991 eqns 3-11
    # Spin-adapted, every TEI term is modified to include
    # antisymmetrized term
    def make_taut(self, t_ia, t_ijab):
        tau_t = t_ijab + 0.5 * (np.einsum('ia,jb->ijab', t_ia, t_ia))
        return tau_t


    def make_tau(self, t_ia, t_ijab):
        tau = t_ijab + (np.einsum('ia,jb->ijab', t_ia, t_ia))
        return tau


    def make_Fae(self, taut, t_ia, t_ijab):
        Fae = self.F_vir.copy()
        #Fae[np.diag_indices_from(Fae)] = 0
        Fae -= ndot('me,ma->ae', self.F[:self.no_occ, self.no_occ:], t_ia, prefactor=0.5)
        Fae += ndot('mf,mafe->ae', t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:], prefactor=2.0)
        Fae -= ndot('mf,maef->ae', t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        Fae -= ndot('mnaf,mnef->ae', taut, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], prefactor=2.0)
        Fae += ndot('mnaf,mnfe->ae', taut, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Fae


    def make_Fmi(self, taut, t_ia, t_ijab):
        Fmi = self.F_occ.copy()
        #Fmi[np.diag_indices_from(Fmi)] = 0
        Fmi += ndot('ie,me->mi', t_ia, self.F[:self.no_occ, self.no_occ:], prefactor=0.5)
        Fmi += ndot('ne,mnie->mi', t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:], prefactor=2.0)
        Fmi -= ndot('ne,mnei->mi', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        Fmi += ndot('inef,mnef->mi', taut, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], prefactor=2.0)
        Fmi -= ndot('inef,mnfe->mi', taut, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Fmi


    def make_Fme(self, t_ia, t_ijab):
        Fme = self.F[:self.no_occ, self.no_occ:].copy()
        Fme += ndot('nf,mnef->me', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], prefactor = 2.0)
        Fme -= ndot('nf,mnfe->me', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Fme


    def make_Wmnij(self, tau, t_ia, t_ijab):
        Wmnij = self.MO[:self.no_occ, :self.no_occ, :self.no_occ, :self.no_occ].copy()
        Wmnij += ndot('je,mnie->mnij', t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:])
        Wmnij += ndot('ie,mnej->mnij', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        Wmnij += ndot('ijef,mnef->mnij', tau, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Wmnij


    def make_Wmbej(self, t_ia, t_ijab):
        Wmbej = self.MO[:self.no_occ, self.no_occ:, self.no_occ:, :self.no_occ].copy()
        Wmbej += ndot('jf,mbef->mbej', t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        Wmbej -= ndot('nb,mnej->mbej', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        tmp = 0.5 * t_ijab.copy() + np.einsum('jf,nb->jnfb', t_ia, t_ia)
        Wmbej -= ndot('jnfb,mnef->mbej', tmp, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        Wmbej += ndot('njfb,mnef->mbej', t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        Wmbej -= ndot('njfb,mnfe->mbej', t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], prefactor=0.5)
        return Wmbej


    def make_Wmbje(self, t_ia, t_ijab):
        Wmbje = -1.0 * self.MO[:self.no_occ, self.no_occ:, :self.no_occ, self.no_occ:].copy()
        Wmbje -= ndot('jf,mbfe->mbje', t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        Wmbje += ndot('nb,mnje->mbje', t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:])
        tmp = 0.5 * t_ijab.copy() + np.einsum('jf,nb->jnfb', t_ia, t_ia)
        Wmbje += ndot('jnfb,mnfe->mbje', tmp, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Wmbje


    def make_Zmbij(self, tau):
        Zmbij = 0
        Zmbij += ndot('mbef,ijef->mbij', self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:], tau)
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
        Ria += ndot('ie,ae->ia', t_ia, Fae)
        Ria -= ndot('ma,mi->ia', t_ia, Fmi)
        Ria += ndot('imae,me->ia', t_ijab, Fme, prefactor=2.0)
        Ria -= ndot('imea,me->ia', t_ijab, Fme)
        Ria -= ndot('nf,naif->ia', t_ia, self.MO[:no_occ, no_occ:, :no_occ, no_occ:])
        Ria += ndot('nf,nafi->ia', t_ia, self.MO[:no_occ, no_occ:, no_occ:, :no_occ], prefactor=2.0)
        Ria += ndot('mief,maef->ia', t_ijab, self.MO[:no_occ, no_occ:, no_occ:, no_occ:], prefactor=2.0)
        Ria -= ndot('mife,maef->ia', t_ijab, self.MO[:no_occ, no_occ:, no_occ:, no_occ:])
        Ria -= ndot('mnae,nmei->ia', t_ijab, self.MO[:no_occ, :no_occ, no_occ:, :no_occ], prefactor=2.0)
        Ria += ndot('mnae,nmie->ia', t_ijab, self.MO[:no_occ, :no_occ, :no_occ, no_occ:])

        # Create residual T2s
        Rijab = self.MO[:no_occ, :no_occ, no_occ:, no_occ:].copy()
        # Term 2
        tmp = ndot('ijae,be->ijab', t_ijab, Fae)
        Rijab += tmp 
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)

        tmp = 0.5 * np.einsum('ijae,mb,me->ijab', t_ijab, t_ia, Fme)
        Rijab -= tmp
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 3
        tmp = ndot('imab,mj->ijab', t_ijab, Fmi)
        Rijab -= tmp
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        tmp = 0.5 * np.einsum('imab,je,me->ijab', t_ijab, t_ia, Fme)
        Rijab -= tmp 
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 4
        Rijab += ndot('mnab,mnij->ijab', tau, Wmnij)
        # Term 5
        Rijab += ndot('ijef,abef->ijab', tau, self.MO[no_occ:, no_occ:, no_occ:, no_occ:])
        # Extra term since Wabef is not formed
        tmp = ndot('ma,mbij->ijab', t_ia, Zmbij)
        Rijab -= tmp
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 6 # 1
        tmp = ndot('imae,mbej->ijab', t_ijab, Wmbej)
        tmp -= ndot('imea,mbej->ijab', t_ijab, Wmbej)
        Rijab += tmp
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
        tmp1 = np.einsum('ie,ma,mbej->ijab', t_ia, t_ia, self.MO[:no_occ, no_occ:, no_occ:, :no_occ])
        Rijab -= tmp1
        Rijab -= tmp1.swapaxes(0, 1).swapaxes(2, 3)
        # Term 6 # 2
        tmp = ndot('imae,mbej->ijab', t_ijab, Wmbej)
        tmp += ndot('imae,mbje->ijab', t_ijab, Wmbje)
        Rijab += tmp
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 6 # 3
        tmp = ndot('mjae,mbie->ijab', t_ijab, Wmbje)
        Rijab += tmp
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
        tmp1 = np.einsum('ie,mb,maje->ijab', t_ia, t_ia, self.MO[:no_occ, no_occ:, :no_occ, no_occ:])
        Rijab -= tmp1
        Rijab -= tmp1.swapaxes(0, 1).swapaxes(2, 3)

        # Term 7
        tmp = ndot('ie,abej->ijab', t_ia, self.MO[no_occ:, no_occ:, no_occ:, :no_occ])
        Rijab += tmp
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 8
        tmp = ndot('ma,mbij->ijab', t_ia, self.MO[:no_occ, no_occ:, :no_occ, :no_occ])
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
                R1Q = np.einsum('b,ba->a', Ria[i], tmp_Q)
                # Transform RQs using L
                R1QL = np.einsum('b,ba->a', R1Q, tmp_L)
                tmp3 = self.eps_pno_list[i*no_occ+i]
                # Use vir orb. energies from semicanonical
                d1_QL = np.zeros(tmp3.shape[0])
                for a in range(tmp3.shape[0]):
                    d1_QL[a] = self.F_occ[i, i] - tmp3[a]
                T1QL = R1QL / d1_QL
                # Back transform to TQs
                T1Q = np.einsum('ab,b->a', tmp_L, T1QL)
                # Back transform to Ts
                new_tia[i] += np.einsum('ab,b->a', tmp_Q, T1Q)
            '''
            # Update T2s
            new_tijab = t_ijab.copy()
            for ij in range(no_occ * no_occ):
                tmp1 = self.Q_list[ij]
                # Transform Rs using Q
                R2Q = np.einsum('ac,ab,bd->cd', tmp1, Rijab[ij // no_occ, ij % no_occ], tmp1)
                # Transform RQs using L
                tmp2 = self.L_list[ij]
                R2QL = np.einsum('ac,ab,bd->cd', tmp2, R2Q, tmp2)
                # Use vir orb. energies from semicanonical
                tmp3 = self.eps_pno_list[ij]
                d2_QL = np.zeros((tmp3.shape[0], tmp3.shape[0]))
                for a in range(tmp3.shape[0]):
                    for b in range(tmp3.shape[0]):
                        d2_QL[a, b] = self.F_occ[ij // no_occ, ij // no_occ ] + self.F_occ[ij % no_occ, ij % no_occ] - tmp3[a] - tmp3[b]
                #print('denom in semi-canonical PNO basis:\n{}\n'.format(d_QL.shape))
                T2QL = R2QL / d2_QL
                # Back transform to TQs
                T2Q = np.einsum('ca,ab,db->cd', tmp2, T2QL, tmp2)
                # Back transform to Ts
                new_tijab[ij // no_occ, ij % no_occ] += np.einsum('ca,ab,db->cd', tmp1, T2Q, tmp1)
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
        E_corr = ndot('ia,ia->', self.F[:no_occ, no_occ:], t_ia, prefactor=2.0)
        one = E_corr
        print("1 e- energy: {}".format(one))
        tmp_tau = self.make_tau(t_ia, t_ijab)
        two = ndot('ijab,ijab->', self.MO[:no_occ, :no_occ, no_occ:, no_occ:], tmp_tau, prefactor=2.0)
        two -= ndot('ijba,ijab->', self.MO[:no_occ, :no_occ, no_occ:, no_occ:], tmp_tau, prefactor=1.0)
        print("2 e- energy: {}".format(two))
        E_corr = one + two
        return E_corr

    def do_CC(self, local=False, e_conv=1e-8, r_conv=1e-7, maxiter=40, max_diis=8, start_diis=0):
        self.old_e = self.corr_energy(self.t_ia, self.t_ijab)
        print('Iteration\t\t CCSD Correlation energy\t\tDifference\t\tRMS\nMP2\t\t\t {}'.format(self.old_e))
    # Set up DIIS
        diis = HelperDIIS(self.t_ia, self.t_ijab, max_diis)

    # Iterate until convergence
        for i in range(maxiter):
            tau_t = self.make_taut(self.t_ia, self.t_ijab)
            tau = self.make_tau(self.t_ia, self.t_ijab)
            new_tia, new_tijab = self.update_ts(local, tau, tau_t, self.t_ia, self.t_ijab)
            if i == 0:
                np.save('t1.npy',new_tia)
                np.save('t2.npy',new_tijab)
            new_e = self.corr_energy(new_tia, new_tijab)
            rms = np.linalg.norm(new_tia - self.t_ia)
            rms += np.linalg.norm(new_tijab - self.t_ijab)
            print('CC Iteration: {}\t\t {}\t\t{}\t\t{}\tDIIS Size: {}'.format(i, new_e, abs(new_e - self.old_e), rms, diis.diis_size))
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
